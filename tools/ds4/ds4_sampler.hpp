// tools/ds4/ds4_sampler.hpp - (mab776 M5) Strata's sampler chain on the host, for ds4_serve.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// A port of the default path of src/kernels/cuda/sampler.cu (sampler_greedy_kernel; the top_k selection of the
// one-block / split kernels; sampled_tail_warp) following its host mirror in src/kernels/sampler_parity.cpp
// (mirror_select / mirror_pick).  Ds4Dense hands its logits back on the host, so the CPU and the CUDA engine share
// this one implementation.  What is kept exactly:
//   - ONE penalties stage, first (issue #53): count = occurrences in the window; repeat MULTIPLIES a logit <= 0 and
//     DIVIDES a positive one; then minus count * freq, minus present when count > 0 (presence is a flag);
//   - the window = the last min(penalty_last_n, 4096) tokens of prompt + generated-so-far, i.e. every token up to and
//     including the row's own position (generate.cpp's `consumed` + the fed-back head, penalty_rows());
//   - greedy (temperature <= 0) = the first maximum of the (penalised) logits: std::max_element, which is what
//     ds4_generate's greedy decode returns (and the kernel's lowest-index tie rule on finite rows);
//   - top_k: 1..63 as given, 0 (off) and 64+ keep the widest list, 64, never more than the vocabulary; the list is in
//     selection order (value descending, ties to the lower id); NaN and -inf are never picked, and a round that
//     finds nothing stores (-inf, id 0) - every later round then finds nothing too;
//   - top_p in double over that list (cut at the first i whose cumulative probability reaches top_p, at least
//     min_keep), then min_p as the prefix with logit >= logit[0] + logf(min_p), then 1/temperature in float, a
//     softmax in double, and one Philox draw: the first i with u < cumulative probability;
//   - Philox4x32-10 exactly as sampler.cu's philox_uniform (round keys = the round index, the seed in c2/c3), on
//     (seed, counter) with counter = the POSITION of the row being sampled (verify.cpp: "Row t's draw is
//     Philox(seed, pos0 + t)"), so a seed replays the same text whether the prompt was read fresh or continued.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ds4serve {

constexpr int kPenaltyWindowCap = 4096;   // generate.cpp's kPenaltyWindowCap
constexpr int kSelMax = 64;               // the widest top_k list (sampler.cu's kSelMax)

struct SampleParams {
    float    temperature = 0.0f;       // <= 0: greedy
    int      top_k = 20;               // upstream's default when a request gives none
    float    top_p = 1.0f;             // >= 1 disables
    float    min_p = 0.0f;             // the caller clamps it to [0, 1]
    int      min_keep = 1;
    int      penalty_last_n = 0;       // 0 = no penalties
    float    penalty_repeat = 1.0f;
    float    penalty_freq = 0.0f;
    float    penalty_present = 0.0f;
    uint64_t seed = 0;
};

// sampler.cu's philox4x32_round + philox_uniform, transcribed (sampler_parity.cpp's host_philox_uniform).
inline float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ i;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ 0u;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);   // 24 bits: [0, 1), never 1.0
}

// sampler.cu's apply_penalties
inline float apply_penalty(float logit, int count, const SampleParams & p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

inline int sampled_k(int top_k, int n_vocab) {
    const int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// The top_k list in selection order: one pass keeping the best k (value desc, id asc).  A later id never beats an
// earlier one of the same value (the scan is ascending and the comparison strict), which is the kernels' tie rule.
inline int select_top_k(const float * s, int nv, int k, int * ids, float * vals) {
    int m = 0;
    for (int v = 0; v < nv; ++v) {
        const float x = s[v];
        if (!(x > -std::numeric_limits<float>::infinity())) continue;   // NaN and -inf are never picked
        if (m == k && !(x > vals[m - 1])) continue;
        int j = m < k ? m++ : k - 1;   // the slot it takes (the worst goes when the list is full)
        while (j > 0 && x > vals[j - 1]) { vals[j] = vals[j - 1]; ids[j] = ids[j - 1]; --j; }
        vals[j] = x;
        ids[j] = v;
    }
    for (int i = m; i < k; ++i) { ids[i] = 0; vals[i] = -std::numeric_limits<float>::infinity(); }   // empty rounds
    return m;
}

// The pick over a selection-ordered list (sampled_tail_warp / mirror_pick).
inline int sampled_tail(const int * sel_ids, const float * sel_logit, int k, const SampleParams & p, uint64_t counter) {
    double ex[kSelMax];
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) ex[i] = std::exp((double) sel_logit[i] - (double) mx);
        for (int i = 0; i < k; ++i) sum += ex[i];
        for (int i = 0; i < k; ++i) ex[i] = ex[i] / sum;
        int cut = k;
        double cum = 0.0;
        for (int i = 0; i < k; ++i) {
            cum += ex[i];
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);   // logf
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, sel_logit[i] * inv_t);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) ex[i] = std::exp((double) (sel_logit[i] * inv_t) - (double) smx);
    for (int i = 0; i < n_keep; ++i) sum += ex[i];
    for (int i = 0; i < n_keep; ++i) ex[i] = ex[i] / sum;
    const float u = philox_uniform(p.seed, counter);
    double cum = 0.0;
    int pick = sel_ids[n_keep > 0 ? n_keep - 1 : 0];
    for (int i = 0; i < n_keep; ++i) {
        cum += ex[i];
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
    return pick;
}

// One token from one row of `nv` logits.  `hist` / `n_hist`: every token up to and including the row's own position
// (prompt + generated so far), most recent last; only its last min(penalty_last_n, 4096) count.  `counter` = the
// row's position.  `scratch` holds the penalised copy (reused across calls).
inline int sample_row(const float * logits, int nv, const SampleParams & p, uint64_t counter,
                      const int * hist, size_t n_hist, std::vector<float> & scratch) {
    const float * s = logits;
    const size_t win = std::min<size_t>(n_hist, (size_t) std::min(std::max(p.penalty_last_n, 0), kPenaltyWindowCap));
    if (win > 0) {
        scratch.assign(logits, logits + nv);
        std::vector<int> w(hist + (n_hist - win), hist + n_hist);
        std::sort(w.begin(), w.end());
        for (size_t i = 0; i < w.size();) {
            size_t j = i;
            while (j < w.size() && w[j] == w[i]) ++j;
            const int v = w[i];
            if (v >= 0 && v < nv) scratch[(size_t) v] = apply_penalty(logits[v], (int) (j - i), p);   // ids outside: ignored
            i = j;
        }
        s = scratch.data();
    }
    if (p.temperature <= 0.0f) return (int) (std::max_element(s, s + nv) - s);
    if (nv <= 0) return 0;
    const int k = sampled_k(p.top_k, nv);
    int ids[kSelMax] = {};
    float vals[kSelMax] = {};
    select_top_k(s, nv, k, ids, vals);
    return sampled_tail(ids, vals, k, p, counter);
}

}  // namespace ds4serve
