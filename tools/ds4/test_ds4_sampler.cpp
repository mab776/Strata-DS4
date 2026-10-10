// tools/ds4/test_ds4_sampler.cpp - (mab776 M5) the host sampler chain (tools/ds4/ds4_sampler.hpp) against
// independent references.  No ggml, no model: `test_ds4_sampler` exits 0 when every check passes.
//
//   test_ds4_sampler                     the checks
//   test_ds4_sampler --philox S C [C..]  philox_uniform(S, C) per counter, %.9g (test_ds4_serve.py compares them with
//                                        its own transcription of sampler.cu)
//
// The references are deliberately naive: `ref_select` is sampler_parity.cpp's mirror_select (k rounds of an argmax over
// the not-yet-taken, ties to the lower id, -inf/NaN never picked, an empty round = (-inf, id 0) and id 0 taken), and
// `ref_pick` its mirror_pick, ported line by line.  The chi-square check compares the draw frequencies of 100k
// counters with the exact distribution of the chain computed in double from first principles.
#include "ds4_sampler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using ds4serve::SampleParams;

int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++g_fail; } } while (0)

const float kInf = std::numeric_limits<float>::infinity();

std::vector<float> penalised(const std::vector<float> & l, const std::vector<int> & window, const SampleParams & p) {
    std::vector<float> s(l);
    for (size_t v = 0; v < l.size(); ++v) {
        int c = 0;
        for (int h : window) c += h == (int) v;
        float x = l[v];
        if (c > 0) {
            if (x <= 0.0f) x *= p.penalty_repeat; else x /= p.penalty_repeat;
            x -= (float) c * p.penalty_freq + (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
        }
        s[v] = x;
    }
    return s;
}

void ref_select(const std::vector<float> & s, int k, std::vector<int> & ids, std::vector<float> & vals) {
    const int nv = (int) s.size();
    std::vector<char> taken((size_t) nv, 0);
    ids.clear(); vals.clear();
    for (int i = 0; i < k; ++i) {
        int best = nv;
        float bv = -kInf;
        for (int v = 0; v < nv; ++v)
            if (!taken[(size_t) v] && s[(size_t) v] > bv) { bv = s[(size_t) v]; best = v; }
        const int id = best < nv ? best : 0;
        taken[(size_t) id] = 1;
        ids.push_back(id);
        vals.push_back(bv);
    }
}

int ref_pick(const std::vector<int> & ids, const std::vector<float> & vals, int k, const SampleParams & p, uint64_t counter) {
    int n_keep = k;
    float mx = vals[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, vals[(size_t) i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) vals[(size_t) i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) vals[(size_t) i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = vals[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (vals[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return vals[(size_t) i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    const float u = ds4serve::philox_uniform(p.seed, counter);
    double cum = 0.0;
    int pick = ids[(size_t) (n_keep > 0 ? n_keep - 1 : 0)];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = ids[(size_t) i]; break; }
    }
    return pick;
}

int ref_sample(const std::vector<float> & l, const std::vector<int> & hist, const SampleParams & p, uint64_t counter) {
    const size_t win = std::min<size_t>(hist.size(), (size_t) std::min(std::max(p.penalty_last_n, 0), ds4serve::kPenaltyWindowCap));
    const std::vector<int> window(hist.end() - (long) win, hist.end());
    const std::vector<float> s = win > 0 ? penalised(l, window, p) : l;
    if (p.temperature <= 0.0f) {   // the first maximum
        int best = 0;
        for (int v = 1; v < (int) s.size(); ++v) if (s[(size_t) v] > s[(size_t) best]) best = v;
        return best;
    }
    const int nv = (int) l.size();
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64);
    std::vector<int> ids;
    std::vector<float> vals;
    ref_select(s, k, ids, vals);
    return ref_pick(ids, vals, k, p, counter);
}

// the exact distribution of the sampled chain over token ids (double, from the chain's definition)
std::vector<double> exact_dist(const std::vector<float> & s, const SampleParams & p) {
    const int nv = (int) s.size();
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64);
    std::vector<int> ids;
    std::vector<float> vals;
    ref_select(s, k, ids, vals);
    int n_keep = k;
    if (p.top_p < 1.0f) {
        double z = 0;
        for (int i = 0; i < k; ++i) z += std::exp((double) vals[(size_t) i] - (double) vals[0]);
        double cum = 0;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) vals[(size_t) i] - (double) vals[0]) / z;
            if (cum >= (double) p.top_p) { n_keep = i + 1; break; }
        }
    }
    if (p.min_p > 0.0f)
        for (int i = 0; i < n_keep; ++i)
            if (vals[(size_t) i] < vals[0] + std::log(p.min_p)) { n_keep = i; break; }
    std::vector<double> d((size_t) nv, 0.0);
    double z = 0;
    for (int i = 0; i < n_keep; ++i) z += std::exp((double) (vals[(size_t) i] / p.temperature) - (double) (vals[0] / p.temperature));
    for (int i = 0; i < n_keep; ++i)
        d[(size_t) ids[(size_t) i]] = std::exp((double) (vals[(size_t) i] / p.temperature) - (double) (vals[0] / p.temperature)) / z;
    return d;
}

std::vector<float> random_row(std::mt19937 & rng, int nv, float scale, int n_ties, bool specials) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> l((size_t) nv);
    for (float & x : l) x = nd(rng);
    std::uniform_int_distribution<int> pick(0, nv - 1);
    for (int i = 0; i < n_ties; ++i) l[(size_t) pick(rng)] = l[(size_t) pick(rng)];   // exact ties
    if (specials) {
        l[(size_t) pick(rng)] = -kInf;
        l[(size_t) pick(rng)] = std::numeric_limits<float>::quiet_NaN();
        l[(size_t) pick(rng)] = -0.0f;
        l[(size_t) pick(rng)] = 0.0f;
    }
    return l;
}

void test_philox() {
    // [0, 1), and the 24-bit grid
    double mean = 0;
    for (uint64_t c = 0; c < 20000; ++c) {
        const float u = ds4serve::philox_uniform(12345, c);
        CHECK(u >= 0.0f && u < 1.0f, "philox out of [0,1): %g", u);
        CHECK(std::floor(u * 16777216.0f) == u * 16777216.0f, "philox not on the 24-bit grid");
        mean += u;
    }
    mean /= 20000;
    CHECK(std::fabs(mean - 0.5) < 0.01, "philox mean %g", mean);
    CHECK(ds4serve::philox_uniform(1, 5) != ds4serve::philox_uniform(2, 5), "seed ignored");
    CHECK(ds4serve::philox_uniform(1, 5) != ds4serve::philox_uniform(1, (uint64_t) 5 + ((uint64_t) 1 << 32)),
          "counter high word ignored");
}

void test_penalties() {
    SampleParams p;
    p.penalty_repeat = 2.0f; p.penalty_freq = 0.5f; p.penalty_present = 0.25f;
    // repeat: divides a positive logit, multiplies a non-positive one; freq x count; presence once
    CHECK(ds4serve::apply_penalty(4.0f, 3, p) == 4.0f / 2.0f - (3 * 0.5f + 0.25f), "positive logit");
    CHECK(ds4serve::apply_penalty(-4.0f, 1, p) == -4.0f * 2.0f - (0.5f + 0.25f), "negative logit");
    CHECK(ds4serve::apply_penalty(0.0f, 2, p) == 0.0f * 2.0f - (1.0f + 0.25f), "zero logit");
    CHECK(ds4serve::apply_penalty(4.0f, 0, p) == 4.0f, "count 0 untouched");
    // the window: only the last penalty_last_n tokens count, ids outside the vocabulary are ignored
    std::vector<float> l = { 1.0f, 3.0f, 2.9f, -1.0f };
    std::vector<float> scratch;
    SampleParams g;              // greedy
    g.penalty_last_n = 2; g.penalty_repeat = 2.0f;
    const std::vector<int> hist = { 1, 1, 1, 0, 7, -3 };   // window = { 7, -3 }: nothing in range
    CHECK(ds4serve::sample_row(l.data(), 4, g, 0, hist.data(), hist.size(), scratch) == 1, "window must be the tail");
    const std::vector<int> hist2 = { 0, 1 };                // id 1: 3.0 / 2 = 1.5 < 2.9
    CHECK(ds4serve::sample_row(l.data(), 4, g, 0, hist2.data(), hist2.size(), scratch) == 2, "greedy with penalties");
    g.penalty_last_n = 0;
    CHECK(ds4serve::sample_row(l.data(), 4, g, 0, hist2.data(), hist2.size(), scratch) == 1, "penalty_last_n 0 = off");
}

void test_greedy_ties() {
    std::vector<float> l = { 0.5f, 2.0f, -1.0f, 2.0f, 2.0f };
    std::vector<float> scratch;
    SampleParams g;
    CHECK(ds4serve::sample_row(l.data(), 5, g, 0, nullptr, 0, scratch) == 1, "greedy tie -> lowest id");
    std::vector<float> m = { -kInf, -kInf };
    CHECK(ds4serve::sample_row(m.data(), 2, g, 0, nullptr, 0, scratch) == 0, "all -inf -> 0");
}

void test_top_k_mapping() {
    CHECK(ds4serve::sampled_k(0, 1000) == 64, "top_k 0 -> 64");
    CHECK(ds4serve::sampled_k(64, 1000) == 64, "top_k 64 -> 64");
    CHECK(ds4serve::sampled_k(65, 1000) == 64, "top_k 65 -> 64");
    CHECK(ds4serve::sampled_k(-3, 1000) == 64, "top_k < 0 -> 64");
    CHECK(ds4serve::sampled_k(1, 1000) == 1, "top_k 1");
    CHECK(ds4serve::sampled_k(63, 1000) == 63, "top_k 63");
    CHECK(ds4serve::sampled_k(20, 10) == 10, "top_k capped by the vocabulary");
    // top_k 1 sampled == greedy
    std::mt19937 rng(3);
    std::vector<float> scratch;
    for (int it = 0; it < 50; ++it) {
        const std::vector<float> l = random_row(rng, 200, 2.0f, 5, false);
        SampleParams p; p.temperature = 0.8f; p.top_k = 1; p.seed = 9;
        SampleParams g;
        CHECK(ds4serve::sample_row(l.data(), 200, p, (uint64_t) it, nullptr, 0, scratch) ==
              ds4serve::sample_row(l.data(), 200, g, 0, nullptr, 0, scratch), "top_k 1 must be the argmax");
    }
}

// the whole chain against the line-by-line port of the kernels' host mirror, on rows with ties, NaN, +-inf, +-0
void test_against_reference() {
    std::mt19937 rng(20261010);
    std::vector<float> scratch;
    int n = 0;
    const int topks[] = { 0, 1, 2, 5, 20, 40, 63, 64, 65, 1000 };
    const float topps[] = { 1.0f, 0.95f, 0.5f, 0.1f, 0.0f, -1.0f };
    const float minps[] = { 0.0f, 0.05f, 0.3f, 1.0f };
    const float temps[] = { 0.0f, 0.3f, 0.7f, 1.0f, 2.5f };
    for (int it = 0; it < 4000; ++it) {
        const int nv = 1 + (int) (rng() % 300);
        const std::vector<float> l = random_row(rng, nv, (it % 3 == 0) ? 0.2f : 3.0f, (int) (rng() % 8), it % 4 == 0 && nv > 4);
        SampleParams p;
        p.top_k = topks[rng() % 10];
        p.top_p = topps[rng() % 6];
        p.min_p = minps[rng() % 4];
        p.temperature = temps[rng() % 5];
        p.seed = rng();
        if (it % 2) {
            p.penalty_last_n = (int) (rng() % 70);
            p.penalty_repeat = 1.0f + (float) (rng() % 100) / 50.0f;
            p.penalty_freq = (float) (rng() % 10) / 10.0f;
            p.penalty_present = (float) (rng() % 10) / 10.0f;
        }
        std::vector<int> hist((size_t) (rng() % 100));
        for (int & h : hist) h = (int) (rng() % (unsigned) (nv + 3)) - 1;   // a few ids outside the vocabulary
        const uint64_t counter = rng();
        const int want = ref_sample(l, hist, p, counter);
        const int got = ds4serve::sample_row(l.data(), nv, p, counter, hist.data(), hist.size(), scratch);
        if (got != want) {
            CHECK(false, "chain mismatch it %d nv %d k %d p %g minp %g t %g: want %d got %d", it, nv, p.top_k, p.top_p,
                  p.min_p, p.temperature, want, got);
            if (++n > 10) return;
        }
    }
}

// The order is observable: at T < 1 a temperature applied before top_p would cut sooner.  Pin that this fixture can
// see it, then that the chain draws from the specified (top_p first) distribution: chi-square over 100k counters.
void test_order_and_chi_square() {
    std::vector<float> l = { 2.0f, 1.6f, 1.3f, 1.1f, 0.4f, 0.2f, -0.5f, -1.0f, 0.9f, 1.7f };
    SampleParams p;
    p.top_k = 6; p.top_p = 0.8f; p.min_p = 0.05f; p.temperature = 0.5f; p.seed = 77;
    p.penalty_last_n = 3; p.penalty_repeat = 1.3f; p.penalty_freq = 0.1f; p.penalty_present = 0.2f;
    const std::vector<int> hist = { 4, 9, 0, 9 };   // window { 9, 0, 9 }
    const std::vector<float> s = penalised(l, { 9, 0, 9 }, p);
    const std::vector<double> d = exact_dist(s, p);
    // the wrong order (temperature first) keeps a different number of tokens on this row
    SampleParams q = p;
    std::vector<float> st(s);
    for (float & x : st) x /= p.temperature;
    q.temperature = 1.0f;
    const std::vector<double> dq = exact_dist(st, q);
    int keep_p = 0, keep_q = 0;
    for (double x : d) keep_p += x > 0;
    for (double x : dq) keep_q += x > 0;
    CHECK(keep_p != keep_q, "fixture cannot see the order (%d vs %d survivors)", keep_p, keep_q);
    const int N = 100000;
    std::vector<int> cnt(l.size(), 0);
    std::vector<float> scratch;
    for (int c = 0; c < N; ++c) ++cnt[(size_t) ds4serve::sample_row(l.data(), (int) l.size(), p, (uint64_t) c, hist.data(), hist.size(), scratch)];
    double chi = 0;
    int dof = -1;
    for (size_t v = 0; v < l.size(); ++v) {
        if (d[v] == 0) { CHECK(cnt[v] == 0, "token %zu outside the support drawn %d times", v, cnt[v]); continue; }
        const double e = d[v] * N;
        chi += (cnt[v] - e) * (cnt[v] - e) / e;
        ++dof;
    }
    // p = 0.001 critical values for dof 1..5: 10.8 13.8 16.3 18.5 20.5
    const double crit[] = { 0, 10.8, 13.8, 16.3, 18.5, 20.5 };
    CHECK(dof >= 1 && dof <= 5, "support size %d", dof + 1);
    if (dof >= 1 && dof <= 5) CHECK(chi < crit[dof], "chi-square %.2f >= %.1f (dof %d)", chi, crit[dof], dof);
    std::fprintf(stderr, "chi-square %.2f over %d tokens (dof %d), order fixture keeps %d vs %d\n", chi, dof + 1, dof,
                 keep_p, keep_q);
}

void test_min_p_prefix() {
    std::vector<float> l = { 3.0f, 2.9f, 1.0f, 0.5f };
    SampleParams p; p.top_k = 4; p.temperature = 1.0f; p.min_p = 0.5f; p.seed = 5;
    std::vector<float> scratch;
    for (uint64_t c = 0; c < 2000; ++c) {
        const int t = ds4serve::sample_row(l.data(), 4, p, c, nullptr, 0, scratch);
        CHECK(t == 0 || t == 1, "min_p 0.5 must keep only the head two, drew %d", t);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--philox") {
        const uint64_t seed = std::strtoull(argv[2], nullptr, 10);
        for (int i = 3; i < argc; ++i)
            std::printf("%.9g\n", ds4serve::philox_uniform(seed, std::strtoull(argv[i], nullptr, 10)));
        return 0;
    }
    test_philox();
    test_penalties();
    test_greedy_ties();
    test_top_k_mapping();
    test_against_reference();
    test_order_and_chi_square();
    test_min_p_prefix();
    if (g_fail) { std::fprintf(stderr, "test_ds4_sampler: %d FAILED\n", g_fail); return 1; }
    std::fprintf(stderr, "test_ds4_sampler: all checks passed\n");
    return 0;
}
