// tools/ds4/ds4_serve.cpp - (mab776 M5) the Strata DeepSeek-V4 engine as a resident server: Strata's engine line
// protocol on stdin/stdout, the model loaded once, prompt reuse across requests, STOP honoured between prompt chunks
// and between tokens, and the full sampler chain (tools/ds4/ds4_sampler.hpp).  serve/server.py's StrataEngine starts
// it as `[exe, "--serve", *args]`.
//
// The engine pieces are ds4_generate's (tools/ds4/ds4_generate.cpp, which this file does not touch - upstream rewrites
// it daily).  Every block copied from it says `= ds4_generate.cpp:A-B @b588dbd1`, so it can be re-diffed.
// Spec: bench/mab776-m2/work/m5/DESIGN.md (untracked); the protocol's consumer: serve/server.py (Strata v0.1.41).
//
//   out  INFO k=v ...                 once, before READY (context, engine, expert_slots, ...)
//   out  READY <ctx> stop             "stop": STOP is honoured mid-request
//   in   GEN <max_new>[ key=value ...] <id,id,...>    the WHOLE prompt every time; the engine finds what it can reuse
//                                     (keys: the sampler's, seed, ckpt, mtp_k = chained drafts per pass, 0 = plain
//                                     decoding; others skipped)
//   out  RESUME <reused>              before reading: the prompt tokens not read again (always <= n - 1)
//   out  PP <pos> <n> <ms> <tok/s>    after every prompt chunk (every 64 tokens without chunks), and once at n
//   out  T <id>                       per generated token - the stop token too
//   out  DONE <gen> <n> <prompt_ms> <decode_ms> <stop|length|cancel> <accepted> <drafted> <reused> <hits> <lookups> 0
//        <file_tier> 0
//        <prompt_read>                exactly once per accepted GEN (15 fields: serve/server.py _parse_done)
//   out  ERR <why>                    instead of DONE; the engine stays in step (nothing follows)
//   in   STOP                         ends the running request (DONE ... cancel); a STOP between requests is stale
//   in   QUIT / EOF                   exit (a running request is abandoned)
//   in   GENI / VRAM / SAVE / RESTORE refused in step: ERR / ERR / SERR
//
// Reuse (DESIGN.md section 4).  `live` = the tokens the state holds (positions [0, L)).  A request continues it when
// it extends it (append), redoes its last <= Ds4Dense::max_tokens() positions in place (rewind: the rings keep that
// many spare slots), restores a ring checkpoint whose tokens it starts with, or starts over (reset); the largest
// reuse wins.  Checkpoints sit at n - 1 - before the last prompt token, which DeepSeek's template turns from
// `<think>` into `</think>` when the turn becomes history - in two roles: `last` (requests that continued the live
// state: #123 / #1053 continuation passes) and `turn` (requests that restored or reset: new turns, resends), so a
// continuation pass never overwrites the checkpoint the next turn needs; plus, with --turn-token, `user` at the last
// turn token (an edited last message).  A checkpoint above the resume point is dropped: everything from there on is
// rewritten, and a checkpoint at P stays exact only while every write lands at >= P.
#include "ds4_dense.hpp"
#include "ds4_moe.hpp"
#include "ds4_sampler.hpp"
#include "mtp_experts.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "strata/artifact/ds4_geometry.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

namespace ds4 = strata::ds4;

constexpr const char * kEngineVersion = "ds4-serve-0.3";   // 0.2: M6 (GPU 1 tier, elastic cache); 0.3: M7c (K drafts)
constexpr int kPrefillChunkMax = 8192;   // a bigger chunk overflows the attention graph's 4096 nodes at long context

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------------------------------------ arguments

struct Args {
    std::string model, profile;
    std::string backend = "cuda";    // dense half
    std::string experts = "gpu";     // routed tier: gpu (cache + prefetch + CPU/PCIe) | cpu (pool only)
    int64_t slots = 0;               // --expert-slots: <= 0 = auto (ds4_generate's --slots auto, M1)
    double vram_margin_gib = 0.9;
    bool vram_margin_set = false;    // an explicit --vram-margin wins over the long-context default
    int64_t vram_reserve_mib = -1;   // --vram-reserve-mib (serve/server.py's Settings page may append it): the margin
    double pcie = 0.25, pf_b = 1.43, arena_gib = 60.0;
    int threads = 0;
    int64_t ctx = 0;                 // --ctx / --max-context (required)
    uint64_t seed = 0;               // sampled requests without seed=: 0 = the clock
    std::vector<int> stop;           // stop ids (default: the GGUF's tokenizer.ggml.eos_token_id)
    bool stop_set = false;
    float route_bias = 0.0f;
    bool vram_lru = false, arena_adapt = false, arena_skip = false;
    int prefill_chunk = 0;           // 0: the prompt through the decode loop
    int chunk_min = 32;              // a prompt span shorter than this goes through the decode loop
    bool chunk_mmq = false, chunk_prestage = false;
    float skip_miss = 0.0f;
    bool f16_q8 = false, comp_q8 = false, icomp_q8 = false, comp_host = false;
    std::string comp_type, dense_requant;
    int comp_rot = -1;
    int turn_token = -1;             // --turn-token: the `user` checkpoint at its last occurrence (-1 = off)
    bool no_ckpt = false;            // --no-ckpt: never take checkpoints (append / rewind reuse only)
    // mab776 M6 (= ds4_generate's M7a flags): GPU 1 as a static expert tier, and the elastic GPU 0 cache - the prompt
    // chunks run on the --vram-margin budget, the cache is grown to leave --vram-grow GiB free between prompts
    int gpu1 = -1;
    double gpu1_gib = 0.0, gpu1_reserve = 0.6;
    double vram_grow = -1.0;
    // mab776 M6: conversation parking - a conversation the server switches away from goes to host RAM (its rings,
    // checkpoints and compressed rows, ~0.3 GB at 127k), so coming back to it reads only what is new
    int64_t park_mib = 0;            // --park-mib: the host RAM parked conversations may use (0 = off)
    int park_min = 512;              // --park-min: shorter conversations are not worth parking
    // mab776 M7b: --mtp FILE = speculative decoding with the DeepSeek MTP head - one 2-token verify pass per step,
    // exact speculative sampling for sampled requests (ds4_sampler.hpp verify_row); its routed experts on the CPU
    // (MXFP4; --mtp-resident copies them into RAM, --mtp-keep K computes only the K heaviest)
    std::string mtp;
    bool verify = false;
    bool mtp_resident = false;
    int mtp_keep = 0;
    int mtp_device = -1;   // --mtp-device N: the MTP head's experts on GPU N (3.4 GB MXFP4) instead of the CPU
    // mab776 M7c: --mtp-k K = K chained drafts per verify pass (the head fed its own output for drafts 2..K), one pass
    // of K + 1 tokens; a request's `mtp_k` key overrides it.  1 = M7b's single draft.
    int mtp_k = 1;
};

bool to_i64(const std::string & s, int64_t & out) {
    if (s.empty()) return false;
    errno = 0;
    char * e = nullptr;
    const long long v = std::strtoll(s.c_str(), &e, 10);
    if (*e != '\0' || errno != 0) return false;
    out = v;
    return true;
}

bool to_f64(const std::string & s, double & out) {
    if (s.empty()) return false;
    errno = 0;
    char * e = nullptr;
    const double v = std::strtod(s.c_str(), &e);
    if (*e != '\0' || errno != 0 || !std::isfinite(v)) return false;
    out = v;
    return true;
}

bool to_csv(const std::string & s, std::vector<int> & out) {
    out.clear();
    std::string t = s;
    for (char & c : t) if (c == ',') c = ' ';
    std::istringstream in(t);
    std::string w;
    while (in >> w) {
        int64_t v = 0;
        if (!to_i64(w, v) || v < 0 || v > INT32_MAX) return false;
        out.push_back((int) v);
    }
    return true;
}

void usage() {
    std::fprintf(stderr,
        "usage: ds4_serve [--serve] -m model.gguf --ctx N [--backend cuda|cpu] [--experts gpu|cpu]\n"
        "       [--expert-slots auto|N] [--vram-margin GiB | --vram-reserve-mib MiB] [--pcie 0.25] [--pf-b 1.43]\n"
        "       [--profile ds4routes.bin]\n"
        "       [--arena-gib 60] [--threads N] [--prefill-chunk N [--chunk-min 32] [--chunk-mmq] [--chunk-prestage]]\n"
        "       [--comp-type f32|q8_0|q5_0|q4_0|iq4_nl] [--comp-q8] [--comp-rot 0|1] [--icomp-q8] [--comp-host]\n"
        "       [--f16-q8] [--dense-requant q4_k|q5_k|q6_k] [--route-bias F] [--vram-lru] [--arena-adapt]\n"
        "       [--arena-skip-resident] [--skip-miss T] [--seed N] [--stop id,id] [--turn-token ID] [--no-ckpt]\n"
        "       [--gpu1 N [--gpu1-gib G] [--gpu1-reserve G]] [--vram-grow G] [--park-mib MiB [--park-min tokens]]\n"
        "       [--mtp FILE [--mtp-resident] [--mtp-keep K] [--mtp-device N] [--mtp-k 1..7]]\n"
        "  Strata's engine line protocol on stdin/stdout (serve/server.py); see the head of tools/ds4/ds4_serve.cpp.\n");
}

// false + `why` on a bad command line (exit 2: the server shows it in the start error's log tail)
bool parse_args(int argc, char ** argv, Args & a, std::string & why) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        std::string v;
        auto next = [&]() -> bool {
            if (i + 1 >= argc) { why = k + " needs a value"; return false; }
            v = argv[++i];
            return true;
        };
        auto num_i = [&](int64_t lo, int64_t hi, int64_t & out) -> bool {
            if (!next()) return false;
            if (!to_i64(v, out) || out < lo || out > hi) {
                why = k + " " + v + ": expected an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]";
                return false;
            }
            return true;
        };
        auto num_f = [&](double & out) -> bool {
            if (!next()) return false;
            if (!to_f64(v, out)) { why = k + " " + v + ": expected a finite number"; return false; }
            return true;
        };
        int64_t iv = 0;
        double fv = 0;
        if (k == "--serve") continue;                                   // serve/server.py always passes it first
        else if (k == "-m" || k == "--model") { if (!next()) return false; a.model = v; }
        else if (k == "--backend") { if (!next()) return false; a.backend = v; }
        else if (k == "--experts") { if (!next()) return false; a.experts = v; }
        else if (k == "--expert-slots") {
            if (!next()) return false;
            if (v == "auto") a.slots = 0;
            else if (!to_i64(v, a.slots) || a.slots < 0) { why = "--expert-slots " + v + ": expected auto or N >= 0"; return false; }
        }
        else if (k == "--slots") {
            why = "--slots: serve/server.py reads --slots as its parallel-request count (it int()s the value and 'auto' "
                  "crashes it); use --expert-slots auto|N for the VRAM expert cache";
            return false;
        }
        else if (k == "--vram-margin") { if (!num_f(fv)) return false; a.vram_margin_gib = fv; a.vram_margin_set = true; }
        else if (k == "--vram-reserve-mib") { if (!num_i(0, (int64_t) 1 << 20, iv)) return false; a.vram_reserve_mib = iv; }
        else if (k == "--pcie") { if (!num_f(fv)) return false; a.pcie = fv; }
        else if (k == "--pf-b") { if (!num_f(fv)) return false; a.pf_b = fv; }
        else if (k == "--profile") { if (!next()) return false; a.profile = v; }
        else if (k == "--arena-gib") { if (!num_f(fv)) return false; a.arena_gib = fv; }
        else if (k == "--threads") { if (!num_i(0, 1024, iv)) return false; a.threads = (int) iv; }
        else if (k == "--ctx" || k == "--max-context") { if (!num_i(1, (int64_t) 1 << 30, iv)) return false; a.ctx = iv; }
        else if (k == "--seed") {
            if (!next()) return false;
            errno = 0;
            char * e = nullptr;
            const unsigned long long s = std::strtoull(v.c_str(), &e, 10);
            if (v.empty() || *e != '\0' || errno != 0) { why = "--seed " + v + ": expected an unsigned integer"; return false; }
            a.seed = s;
        }
        else if (k == "--stop" || k == "--eos-ids") {
            if (!next()) return false;
            if (!to_csv(v, a.stop)) { why = k + " " + v + ": expected token ids id,id,..."; return false; }
            a.stop_set = true;
        }
        else if (k == "--dense-requant") { if (!next()) return false; a.dense_requant = v; }
        else if (k == "--route-bias") { if (!num_f(fv)) return false; a.route_bias = (float) fv; }
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--arena-skip-resident") a.arena_skip = true;
        else if (k == "--prefill-chunk") { if (!num_i(0, kPrefillChunkMax, iv)) return false; a.prefill_chunk = (int) iv; }
        else if (k == "--chunk-min") { if (!num_i(1, (int64_t) 1 << 30, iv)) return false; a.chunk_min = (int) iv; }
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--chunk-prestage") a.chunk_prestage = true;
        else if (k == "--skip-miss") { if (!num_f(fv)) return false; a.skip_miss = (float) fv; }
        else if (k == "--f16-q8") a.f16_q8 = true;
        else if (k == "--comp-q8") a.comp_q8 = true;
        else if (k == "--comp-type") { if (!next()) return false; a.comp_type = v; }
        else if (k == "--comp-rot") { if (!num_i(0, 1, iv)) return false; a.comp_rot = (int) iv; }
        else if (k == "--icomp-q8") a.icomp_q8 = true;
        else if (k == "--comp-host") a.comp_host = true;
        else if (k == "--turn-token") { if (!num_i(-1, INT32_MAX, iv)) return false; a.turn_token = (int) iv; }
        else if (k == "--no-ckpt") a.no_ckpt = true;
        else if (k == "--gpu1") { if (!num_i(0, 64, iv)) return false; a.gpu1 = (int) iv; }
        else if (k == "--gpu1-gib") { if (!num_f(fv)) return false; a.gpu1_gib = fv; }
        else if (k == "--gpu1-reserve") { if (!num_f(fv)) return false; a.gpu1_reserve = fv; }
        else if (k == "--vram-grow") { if (!num_f(fv)) return false; a.vram_grow = fv; }
        else if (k == "--park-mib") { if (!num_i(0, (int64_t) 1 << 22, iv)) return false; a.park_mib = iv; }
        else if (k == "--park-min") { if (!num_i(1, INT32_MAX, iv)) return false; a.park_min = (int) iv; }
        else if (k == "--ids" || k == "--ids-file" || k == "-n" || k == "--n-predict" || k == "--ppl" ||
                 k == "--dump-logits" || k == "--pos-offset" || k == "--quiet") {
            why = k + ": a one-shot ds4_generate option; ds4_serve takes its prompts from GEN lines";
            return false;
        }
        else if (k == "--temp") {
            why = "--temp: requests carry their own temperature (an absent one means greedy - an engine default would "
                  "sample the requests that asked for greedy)";
            return false;
        }
        else if (k == "--mtp") { if (!next()) return false; a.mtp = v; a.verify = true; }
        else if (k == "--mtp-resident") a.mtp_resident = true;
        else if (k == "--mtp-keep") { if (!num_i(0, 64, iv)) return false; a.mtp_keep = (int) iv; }
        else if (k == "--mtp-device") { if (!num_i(-1, 64, iv)) return false; a.mtp_device = (int) iv; }
        else if (k == "--mtp-k") { if (!num_i(1, 7, iv)) return false; a.mtp_k = (int) iv; }
        else if (k == "--verify") {}   // implied by --mtp: a server drafts only to verify
        else { why = "unknown argument " + k; return false; }
    }
    // Strata's own name for the VRAM left free beside the expert cache: the margin, unless --vram-margin says it
    if (a.vram_reserve_mib >= 0) {
        if (a.vram_margin_set) {
            std::fprintf(stderr, "ds4_serve: --vram-reserve-mib %lld ignored: --vram-margin %.2f GiB is given\n",
                         (long long) a.vram_reserve_mib, a.vram_margin_gib);
        } else {
            a.vram_margin_gib = (double) a.vram_reserve_mib / 1024.0;
            a.vram_margin_set = true;
            std::fprintf(stderr, "ds4_serve: --vram-reserve-mib %lld -> --vram-margin %.2f GiB\n",
                         (long long) a.vram_reserve_mib, a.vram_margin_gib);
        }
    }
    if (a.model.empty()) { why = "-m model.gguf is required"; return false; }
    if ((a.mtp_resident || a.mtp_keep > 0 || a.mtp_device >= 0) && a.mtp.empty()) {
        why = "--mtp-resident / --mtp-keep / --mtp-device need --mtp FILE";
        return false;
    }
    if (a.ctx <= 0) { why = "--ctx N (or --max-context N) is required"; return false; }
    if (a.backend != "cuda" && a.backend != "cpu") { why = "--backend cuda|cpu"; return false; }
    if (a.experts != "gpu" && a.experts != "cpu") { why = "--experts gpu|cpu"; return false; }
    return true;
}

// ------------------------------------------------------------------------------------------------ protocol output

// The protocol stream: a dup of the original stdout.  main() then points fd 1 at stderr, so a stray printf from a
// library lands in the log instead of between two protocol lines.  Only the main thread writes here.
struct Proto {
    std::FILE * f = nullptr;
    void line(const char * fmt, ...) __attribute__((format(printf, 2, 3))) {
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(f, fmt, ap);
        va_end(ap);
        std::fputc('\n', f);
        std::fflush(f);
    }
    void err(const std::string & why) {   // one line, whatever the text holds
        std::string one = why;
        for (char & c : one) if (c == '\n' || c == '\r') c = ' ';
        line("ERR %s", one.c_str());
    }
};

// ------------------------------------------------------------------------------------------------ stdin

bool is_gen(const std::string & l) { return l.rfind("GEN ", 0) == 0 || l.rfind("GENI ", 0) == 0; }

// The reader thread.  read(2) on the descriptor, not std::cin: glibc's exit() waits for stdin's lock, which a
// blocked getline holds (upstream generate.cpp's note).  STOP is matched to the request it was meant for by number,
// not by clearing a flag: the reader counts GEN/GENI lines; a STOP records that count; request #k (the k-th GEN the
// main loop takes) is stopped iff the recorded count is >= k.  A STOP the server sends after DONE (it always does
// when a reply ends on a stop id) carries the old number and is ignored; a STOP right behind a GEN the main loop has
// not taken yet still counts (upstream clears its flag when it takes the GEN and loses that STOP).  QUIT and EOF stop
// everything.  Heap-allocated and never freed: the detached thread may outlive main().
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<int64_t> gens_read{0}, stop_seq{0};
    std::atomic<bool> quit{false}, eof{false};

    void run() {
        std::string buf, l;
        char chunk[4096];
        auto getline_fd = [&](std::string & out) -> bool {
            for (;;) {
                const size_t nl = buf.find('\n');
                if (nl != std::string::npos) {
                    out.assign(buf, 0, nl);
                    buf.erase(0, nl + 1);
                    return true;
                }
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_seq.store(gens_read.load()); continue; }
            if (l == "QUIT") quit.store(true);
            std::lock_guard<std::mutex> lk(mu);
            if (is_gen(l)) gens_read.fetch_add(1);
            q.push_back(l);
            cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            eof.store(true);
            quit.store(true);
        }
        cv.notify_all();
    }
    bool next(std::string & out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof.load(); });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// ------------------------------------------------------------------------------------------------ GEN parsing

struct Gen {
    long long max_new = 0;
    std::vector<int> ids;
    ds4serve::SampleParams sp;
    unsigned long long seed = 0;   // 0 = none given
    bool ckpt = true;
    int mtp_k = -1;                // mab776 M7c: this request's chained drafts per pass (-1 = --mtp-k, 0 = none)
};

// upstream generate.cpp's parse_i64_list: commas and blanks separate, every id an int32 >= 0
bool parse_ids(const char * s, std::vector<int> & out, std::string & err) {
    out.clear();
    std::string text(s);
    for (char & c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

// GEN <max_new>[ key=value ...] <ids> -> "" or the ERR text (upstream's words, generate.cpp ~9021-9189).  Nothing here
// touches the engine: a refused request leaves the conversation state as it was.
std::string parse_gen(const std::string & line, Gen & g, int n_vocab, int64_t ctx) {
    char * endp = nullptr;
    errno = 0;
    g.max_new = std::strtoll(line.c_str() + 4, &endp, 10);
    // optional keys between max_new and the ids; the ids start at the first token without '='; unknown keys
    // (pin, cvec, pcie_frac, spec_min_p, ...) are skipped
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f, pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    long long top_k = 20, pen_last_n = 0;
    std::string bad;
    for (;;) {
        while (*endp == ' ') ++endp;
        const char * start = endp;
        while (*endp != '\0' && *endp != ' ') ++endp;
        if (endp == start) break;
        const std::string tok(start, (size_t) (endp - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) { endp = const_cast<char *>(start); break; }
        const std::string key = tok.substr(0, eq);
        const char * val = tok.c_str() + eq + 1;
        auto fval = [&](float & out) {
            const float f = std::strtof(val, nullptr);
            if (!std::isfinite(f)) bad = key + "=" + val + " is not a finite number";
            else out = f;
        };
        auto ival = [&](long long & out) {
            const long long v = std::strtoll(val, nullptr, 10);   // saturates, never throws
            out = std::clamp<long long>(v, INT32_MIN, INT32_MAX);
        };
        if (key == "temperature") fval(temperature);
        else if (key == "top_p") fval(top_p);
        else if (key == "top_k") ival(top_k);
        else if (key == "min_p") fval(min_p);
        else if (key == "penalty_last_n") ival(pen_last_n);
        else if (key == "penalty_repeat") fval(pen_repeat);
        else if (key == "penalty_freq") fval(pen_freq);
        else if (key == "penalty_present") fval(pen_present);
        else if (key == "seed") g.seed = std::strtoull(val, nullptr, 10);   // a 30-digit seed saturates
        else if (key == "ckpt") g.ckpt = std::strtoll(val, nullptr, 10) != 0;
        else if (key == "mtp_k") { long long v = 0; ival(v); g.mtp_k = (int) std::clamp<long long>(v, 0, 64); }
    }
    std::string pe;
    if (g.max_new < 1 || !parse_ids(endp, g.ids, pe)) return "bad request: " + (pe.empty() ? std::string("max_new") : pe);
    if (!bad.empty()) return "bad request: " + bad;
    const long long n = (long long) g.ids.size();
    if (g.max_new > ctx || n + g.max_new + 8 > ctx)   // the first test keeps the sum from overflowing
        return "prompt (" + std::to_string(n) + " tokens) + max_new (" + std::to_string(g.max_new) +
               ") exceeds the context (" + std::to_string(ctx) + ")";
    for (int t : g.ids)
        if (t >= n_vocab) return "a token id is outside the vocabulary";
    g.sp.temperature = temperature;
    g.sp.top_p = top_p;
    g.sp.top_k = (int) top_k;
    g.sp.min_p = std::clamp(min_p, 0.0f, 1.0f);
    g.sp.penalty_last_n = (int) std::max(pen_last_n, 0LL);
    g.sp.penalty_repeat = pen_repeat;
    g.sp.penalty_freq = pen_freq;
    g.sp.penalty_present = pen_present;
    return "";
}

// ------------------------------------------------------------------------------------------------ the server

enum Slot { kLast = 0, kTurn = 1, kUser = 2, kSlots = 3 };
const char * slot_name(int s) { return s == kLast ? "last" : s == kTurn ? "turn" : "user"; }

enum class Kind { Append, Rewind, Restore, Reset };
const char * kind_name(Kind k) {
    return k == Kind::Append ? "append" : k == Kind::Rewind ? "rewind" : k == Kind::Restore ? "checkpoint" : "reset";
}

struct Ckpt {
    int pos = -1;            // the state held positions [0, pos) when it was taken; -1 = none
    std::vector<int> ids;    // those tokens
};

struct Choice {
    Kind kind = Kind::Reset;
    int R = 0;               // reused prompt tokens
    int slot = -1;           // Restore: the checkpoint
    size_t lcp = 0;
};

// mab776 M6: a parked conversation - everything the server tracks about a state (its tokens and checkpoints) and the
// state itself (Ds4Dense::park), in host RAM
struct Parked {
    std::vector<int> live;
    Ckpt ck[kSlots];
    std::vector<uint8_t> blob;
    int64_t tick = 0;        // last use (LRU)
};

// the reuse a state (tokens `live`, checkpoints `ck`) offers a prompt: append, rewind in place (the rings keep K spare
// slots) or restore a checkpoint; the largest wins, the earlier offer on a tie
Choice best_reuse(const std::vector<int> & live, const Ckpt * ck, int slots, const std::vector<int> & ids, int K) {
    const int n = (int) ids.size(), L = (int) live.size();
    Choice best;
    while ((int) best.lcp < L && (int) best.lcp < n && live[best.lcp] == ids[best.lcp]) ++best.lcp;
    const int lcp = (int) best.lcp;
    auto offer = [&](Kind k, int R, int slot) {
        if (R > best.R) { best.kind = k; best.R = R; best.slot = slot; }
    };
    if (L >= 1 && lcp == L && L <= n - 1) offer(Kind::Append, L, -1);
    const int m = std::min(lcp, n - 1);
    if (m >= 1 && L - m >= 1 && L - m <= K) offer(Kind::Rewind, m, -1);
    for (int s = 0; s < slots; ++s) {
        const Ckpt & c = ck[s];
        if (c.pos >= 1 && c.pos <= n - 1 && (int) c.ids.size() == c.pos && std::equal(c.ids.begin(), c.ids.end(), ids.begin()))
            offer(Kind::Restore, c.pos, s);
    }
    return best;
}

struct Server {
    Args a;
    Proto & proto;
    Input & in;

    ggml_backend_t be = nullptr;
    Ds4Dense dense;
    ds4::Ds4MoeTier tier;
    ds4::Ds4MoeConfig mc;
    int64_t ctx = 0;                 // what READY says
    int n_layer = 0, top_k = 0, n_vocab = 0, slots_ck = 0;
    int64_t n_embd = 0;
    static constexpr int kPredW = ds4::Ds4MoeConfig::kPredW;
    std::vector<int> ids_i, pred_i, cids;
    std::vector<int32_t> ids32, pred32, cids32;
    std::vector<float> w, routed, rb, cw, scratch;
    std::string err;                 // the last engine error

    std::vector<int> live;           // the tokens the state holds, positions [0, live.size())
    Ckpt ck[kSlots];
    int64_t gens = 0;                // GEN/GENI lines taken (request numbers, see Input)
    int test_delay_ms = 0;           // DS4_SERVE_TEST_DELAY_MS
    int test_fail_pos = -1;          // DS4_SERVE_TEST_FAIL=<pos>: the engine fails at that position (the ERR path's test)
    bool shrunk = false;             // mab776 M6: this prompt gave the grown cache's tail back (end_prompt regrows it)
    double t_shrink = 0, t_grow = 0; // ms, the last prompt's
    int il_mtp = -1, NT = 1;         // mab776 M7b: the MTP block (-1 = off), tokens a multi-token pass carries
    ds4::MtpExperts mtp_x;
    std::vector<int> mids, ids_n;
    std::vector<int32_t> ids32_n;
    std::vector<float> mw, mrouted, w_n, routed_n;
    int64_t drafts = 0, accepted = 0;   // this request's
    // mab776 M7c: this request's drafts examined / accepted by chain position (row j of a pass is examined when the
    // drafts before it passed), its verify passes and their ms (step_n + logits_n)
    std::vector<int64_t> d_off, d_acc;
    int64_t passes = 0;
    double t_pass = 0;
    // DS4_SERVE_TEST_ORACLE=<file> (re-read per request): drafts from a plain greedy run's tokens (the draft for the
    // n-th emitted token = line n), DS4_SERVE_TEST_CORRUPT=k spoils every k-th - the mini MTP's random weights never
    // draft right, so without this only the reject path would run (= ds4_generate's DS4_VERIFY_ORACLE)
    std::vector<int> oracle;
    int corrupt_every = 0, n_fixed = 0;
    // DS4_SERVE_TEST_FORCE=1 (with the oracle, mab776 M7c diagnostics): teacher forcing - every emitted token is the
    // oracle's (plain and verify loops), so runs with different K follow the SAME tokens and every logits row
    // (DS4_DUMP_TOKLOGITS) can be compared; drafts are accepted iff they equal it
    bool force = false;
    double t_mtp_win = 0;               // ms: the MTP window over this request's prompt
    double t_mtp_all = 0;               // ms in mtp_draft_n since the request began (window + decode drafts)
    int64_t n_mtp_calls = 0;
    std::vector<Parked> parked;      // mab776 M6: conversations in host RAM (--park-mib)
    size_t park_used = 0;
    int64_t park_tick = 0;
    std::string park_note;           // the last request's parking moves, for its log line

    Server(const Args & args, Proto & p, Input & i) : a(args), proto(p), in(i) {}

    bool init(std::string & why);
    bool step(int tid, int pos);
    bool step_n(const int * toks, int n, int pos0);
    bool mtp_draft_n(const int * next, int n, int pos0, int * draft);
    bool mtp_chain(int next, int src, int pos, int * draft);
    bool mtp_fill_chunk(const std::vector<int> & ids, int c0, int m, int W, int n);
    bool chunk(const int * toks, int n, int pos0, bool & quit);
    void before_chunk();
    void end_prompt();
    Choice choose(const std::vector<int> & ids) const;
    void park_live(const char * why);
    bool unpark(Parked & P);
    bool park_room(size_t need);
    int apply(Choice & c);
    void drop_above(int R);
    void forget_all();
    void take_ckpt(const std::vector<int> & ids, int pos, int slot);
    void delay(const std::function<bool()> & stopped) const;
    void serve_gen(const std::string & line, int64_t seq);
    void loop();
    void close();
};

bool Server::init(std::string & why) {
    test_delay_ms = std::getenv("DS4_SERVE_TEST_DELAY_MS") ? std::atoi(std::getenv("DS4_SERVE_TEST_DELAY_MS")) : 0;
    test_fail_pos = std::getenv("DS4_SERVE_TEST_FAIL") ? std::atoi(std::getenv("DS4_SERVE_TEST_FAIL")) : -1;
    // the model's own limits, before Ds4Dense sizes its compressed caches from them
    strata::Ds4Geometry g;
    try {
        const strata::GgufModel gm = strata::GgufModel::open(a.model);
        if (!(why = strata::ds4_read_geometry(gm, g)).empty()) return false;
        if (!a.stop_set)
            if (const strata::MetaValue * eos = gm.meta().get("tokenizer.ggml.eos_token_id"))
                a.stop.push_back((int) eos->u);
    } catch (const std::exception & e) {
        why = std::string("cannot read ") + a.model + ": " + e.what();
        return false;
    }
    // the context READY reports: --ctx, never past the model's context_length nor what a compressed cache holds
    // (comp_max = min(ctx/4 + 2, context_length/ratio) rows, Ds4Dense::init with ds4_generate's comp_cap_max), so a
    // request the server lets through (prompt + max_new + 8 <= ctx) never meets "beyond the compressed cache capacity"
    ctx = std::min<int64_t>(a.ctx, g.context_length);
    for (int64_t r : g.compress_ratios)
        if (r > 0) ctx = std::min<int64_t>(ctx, std::min<int64_t>(ctx / 4 + 2, std::max<int64_t>(1, g.context_length / r)) * r);
    if (ctx < 16) { why = "context " + std::to_string(ctx) + " is too small"; return false; }
    if (ctx < a.ctx)
        std::fprintf(stderr, "ds4_serve: --ctx %lld -> %lld (the model's context_length / compressed cache capacity)\n",
                     (long long) a.ctx, (long long) ctx);

    // = ds4_generate.cpp:205-210 @b588dbd1 ---- dense half on the chosen backend
    if (a.backend == "cuda") be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    else be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { why = "no " + a.backend + " backend in this build"; return false; }
    std::fprintf(stderr, "dense backend: %s\n", ggml_backend_name(be));
    const bool cpu_backend = ggml_backend_is_cpu(be);
    // (M5) the chunk buffers are allocated per prompt beside the resident cache (5-9 GiB at chunk 2048-4096, u-engine-
    // vram 1.3): with slots auto only an explicit margin can say how much of the card they may have
    if (!cpu_backend && a.prefill_chunk > 0 && a.slots <= 0 && !a.vram_margin_set && a.experts == "gpu") {
        why = "--prefill-chunk with --expert-slots auto on the GPU needs an explicit --vram-margin (the prompt chunks' "
              "buffers are allocated per prompt beside the expert cache: ~5.5-9.5 GiB at chunk 2048-4096)";
        return false;
    }

    // = ds4_generate.cpp:212-251 @b588dbd1 (no MTP, no --pos-offset; ctx from above; + the snapshot slots)
    const double t_load0 = now_ms();
    Ds4DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.skip_routed_experts = true;
    if (!a.dense_requant.empty()) {
        const std::string q = a.dense_requant;
        dc.requant_type = q == "q4_k" ? GGML_TYPE_Q4_K : q == "q5_k" ? GGML_TYPE_Q5_K : q == "q6_k" ? GGML_TYPE_Q6_K : -1;
        if (dc.requant_type < 0) { why = "--dense-requant q4_k|q5_k|q6_k"; return false; }
    }
    dc.comp_cap_max = ctx / 4 + 2;
    dc.mtp_path = a.mtp;   // mab776 M7b
    dc.prefill_chunk = a.prefill_chunk;
    dc.f16_q8 = a.f16_q8;
    dc.comp_q8 = a.comp_q8;
    if (!a.comp_type.empty()) {
        const std::string t = a.comp_type;
        dc.comp_type = t == "f32" ? GGML_TYPE_F32 : t == "q8_0" ? GGML_TYPE_Q8_0 : t == "q5_0" ? GGML_TYPE_Q5_0
                     : t == "q4_0" ? GGML_TYPE_Q4_0 : t == "iq4_nl" ? GGML_TYPE_IQ4_NL : -1;
        if (dc.comp_type < 0) { why = "--comp-type f32|q8_0|q5_0|q4_0|iq4_nl"; return false; }
    }
    dc.comp_rot = a.comp_rot;
    dc.icomp_q8 = a.icomp_q8;
    dc.comp_host = a.comp_host;
    slots_ck = a.no_ckpt ? 0 : (a.turn_token >= 0 ? 3 : 2);
    dc.snapshot_slots = slots_ck;
    std::fprintf(stderr, "context cap %lld tokens\n", (long long) ctx);
    if (!dense.init(a.model, dc, err)) { why = "dense init: " + err; return false; }
    if (!dense.reserve_graphs(a.verify ? a.mtp_k + 1 : 1)) { why = dense.last_error(); return false; }
    // = ds4_generate.cpp:372-384 (mab776 M7b): the MTP head's routed experts - on the CPU, or on GPU --mtp-device.
    // On a GPU they go in NOW, before GPU 0's cache is sized from its free VRAM and before GPU 1's tier is filled, so
    // the tier that card holds is what gives the room up (Marc's placement sweep: 1 = GPU 1, 0 = GPU 0)
    auto init_mtp = [&]() -> bool {
        il_mtp = dense.mtp_layer();
        if (il_mtp < 0) return true;
        mtp_x.keep = a.mtp_keep;
        mtp_x.device = a.mtp_device;
        if (!mtp_x.init(a.mtp, il_mtp, dense.geom().n_embd, dense.geom().n_expert_used,
                        (float) dense.geom().swiglu_clamp_exp[(size_t) il_mtp], a.threads > 0 ? a.threads : 8, err,
                        a.mtp_resident)) {
            why = err;
            return false;
        }
        std::fprintf(stderr, "MTP head: layer %d from %s (experts on %s, MXFP4%s%s)\n", il_mtp, a.mtp.c_str(),
                     a.mtp_device >= 0 ? ("GPU " + std::to_string(a.mtp_device)).c_str() : "the CPU",
                     a.mtp_resident && a.mtp_device < 0 ? ", resident" : "",
                     a.mtp_keep > 0 ? (", keep " + std::to_string(a.mtp_keep)).c_str() : "");
        return true;
    };
    if (a.mtp_device >= 0 && !init_mtp()) return false;

    // = ds4_generate.cpp:252-298 @b588dbd1 (M1: byte-sized slots auto) - verbatim but for return values / the name
    double slot_gib_auto = 0.0;   // mab776: the auto cache budget in GiB when blob sizes differ per layer
    double slot_gib_max_auto = 0.0;   // mab776 M6: --vram-grow's decode budget
    if (a.slots <= 0) {
        // mab776: long contexts get more headroom after the dense half (2.0 GiB from 65,536 tokens; the M2 depth runs
        // at 262,144 calibrate it); an explicit --vram-margin wins
        if (!a.vram_margin_set && ctx >= 65536) a.vram_margin_gib = 2.0;
        // CUDA-graph instances take VRAM when first captured (after this point): the MTP block and the 2-token
        // verify graphs add many (a 0.9 GiB margin OOMed in cudaGraphInstantiate with --mtp, 2026-10-07)
        if (!a.mtp.empty()) a.vram_margin_gib += 0.25;
        if (a.verify) a.vram_margin_gib += 0.75;
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        const double budget = std::max(0.0, (double) fr - a.vram_margin_gib * 1073741824.0);
        // mab776: size by the file's REAL expert bytes, per layer.  A fixed 6.75 MiB (the author's IQ2_XXS + Q2_K
        // file) over-asks on unsloth UD files, whose experts are 7.19-9.38 MiB (UD-IQ2_M: IQ2_XXS/IQ2_S gate/up,
        // IQ3_XXS/MXFP4 down): 44.7 GB asked of 39.4 free.  Mixed sizes -> a byte budget (Ds4MoeConfig::slot_gib, every slot sized
        // to its own layer's blob) with an uncapped count, the way Strata-GLM's glm_generate sizes its cache;
        // uniform sizes -> the slot count that fits (the tier ignores slot_gib there).
        strata::ds4::Ds4MoeGeom mg;
        if (!strata::ds4::ds4_moe_geom_from_gguf(a.model, mg, err)) {
            why = "slots auto: " + err;
            return false;
        }
        int64_t bmin = 0, bmax = 0;
        for (int64_t l = 0; l < mg.n_layers; ++l) {
            if (!mg.routed(l)) continue;
            strata::kernels::cpu::NativeFmt lf;
            if (!strata::ds4::ds4_moe_layer_layout(mg, l, lf, err)) {
                why = "slots auto: " + err;
                return false;
            }
            const int64_t b = (int64_t) lf.bytes;
            bmin = bmin == 0 ? b : std::min(bmin, b);
            bmax = std::max(bmax, b);
        }
        if (bmax <= 0) { why = "slots auto: no routed experts in " + a.model; return false; }
        if (bmin == bmax) {
            a.slots = (int64_t) (budget / (double) bmax);
        } else {
            slot_gib_auto = budget / 1073741824.0;
            a.slots = mg.n_routed() * mg.n_experts;   // the byte budget binds, not the count
        }
        std::fprintf(stderr, "slots auto: %.2f GiB free after the dense half -> %.2f GiB of expert slots (margin %.2f GiB; "
                             "expert %.2f-%.2f MiB -> %lld-%lld experts)\n", fr / 1073741824.0, budget / 1073741824.0,
                     a.vram_margin_gib, bmin / 1048576.0, bmax / 1048576.0, (long long) (budget / (double) bmax),
                     (long long) (budget / (double) bmin));
        if (budget <= 0) a.slots = 0;
        // mab776 M6 (= ds4_generate's M7a --vram-grow): the prompt chunks run on the budget above, the cache is grown
        // between prompts to leave --vram-grow GiB free (mixed blob sizes only: the elastic open)
        if (a.vram_grow >= 0) {
            if (bmin == bmax || a.vram_grow >= a.vram_margin_gib) {
                why = "--vram-grow needs mixed expert sizes and G < --vram-margin";
                return false;
            }
            // the decode graphs the MTP head / verify add are captured after the grow (ds4_generate's extras)
            a.vram_grow += (a.mtp.empty() ? 0.0 : 0.25) + (a.verify ? 0.75 : 0.0);
            slot_gib_max_auto = std::max(0.0, (double) fr - a.vram_grow * 1073741824.0) / 1073741824.0;
            std::fprintf(stderr, "vram grow: prompts on %.2f GiB of slots, decode on up to %.2f GiB (leaving %.2f GiB)\n",
                         slot_gib_auto, slot_gib_max_auto, a.vram_grow);
        }
    }
    if (a.vram_grow >= 0 && slot_gib_max_auto <= 0) { why = "--vram-grow needs --expert-slots auto"; return false; }
    if (a.gpu1 >= 0 && (a.experts != "gpu" || a.profile.empty())) {
        why = "--gpu1 needs --experts gpu and a --profile (its experts come from the ranking)";
        return false;
    }

    // = ds4_generate.cpp:300-331 @b588dbd1 ---- routed-expert tier (the cache seeded at load: a server prefills with
    // the cache resident, as DS4_SEED_FIRST=1 - never deferred to after "the" prompt)
    mc.slots = a.slots;
    mc.slot_gib = slot_gib_auto;   // mab776: 0 = slots x the largest blob (an explicit --slots N, or uniform blobs)
    mc.pcie_frac = a.pcie;
    mc.pf_b = a.pf_b;
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;   // the explicit flag is the ceiling; mem_floor_gib + memguard still guard RAM
    mc.cpu_only = a.experts == "cpu";
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.arena_skip_resident = a.arena_skip;
    mc.defer_cache = false;
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prestage = a.chunk_prestage;
    mc.skip_miss = a.skip_miss;
    mc.slot_gib_max = slot_gib_max_auto;   // mab776 M6
    mc.gpu1_device = a.gpu1;
    mc.gpu1_gib = a.gpu1_gib;
    mc.gpu1_reserve_gib = a.gpu1_reserve;
    mc.gpu1_chunk_tokens = a.prefill_chunk;
    if (!tier.init(a.model, mc, err)) { why = "tier init: " + err; return false; }
    if (!a.profile.empty() && !mc.cpu_only && !tier.seed_from_routes(a.profile, err)) { why = "profile: " + err; return false; }
    std::fprintf(stderr, "tier: %s, %lld resident slots, arena %.1f GiB (%lld experts), file tier %lld; load %.1f s\n",
                 tier.mode(), (long long) tier.resident(), tier.arena_gib(), (long long) tier.arena_experts(),
                 (long long) tier.file_tier(), (now_ms() - t_load0) / 1000.0);
    if (tier.gpu1_device() >= 0)
        std::fprintf(stderr, "gpu1: device %d holds %lld experts (%.2f GiB, filled in %.1f s; no arena copies)\n",
                     tier.gpu1_device(), (long long) tier.resident1(), tier.gpu1_gib(), tier.gpu1_seed_s());
    // mab776 M6: between requests the cache is grown (decode's budget); a prompt that runs chunks shrinks it first
    if (a.vram_grow >= 0 && tier.grow_cache(a.vram_grow, err, true) < 0) { why = err; return false; }
    if (a.mtp_device < 0 && !init_mtp()) return false;

    // = ds4_generate.cpp:333-359 @b588dbd1 (the step buffers)
    n_layer = (int) dense.n_layer();
    top_k = (int) tier.geom().top_k;
    n_embd = tier.geom().n_embd;
    n_vocab = (int) dense.geom().vocab_size;
    NT = dense.max_tokens();
    mids.resize((size_t) (top_k * NT)); ids_n.resize((size_t) (top_k * NT)); ids32_n.resize((size_t) (top_k * NT));
    mw.resize((size_t) (top_k * NT)); w_n.resize((size_t) (top_k * NT));
    mrouted.resize((size_t) (n_embd * NT)); routed_n.resize((size_t) (n_embd * NT));
    ids_i.resize((size_t) top_k); pred_i.resize((size_t) kPredW);
    ids32.resize((size_t) top_k); pred32.resize((size_t) kPredW);
    w.resize((size_t) top_k); routed.resize((size_t) n_embd);
    rb.resize((size_t) tier.geom().n_experts);
    if (a.prefill_chunk > 0) {   // = ds4_generate.cpp:476-479 (the chunk's router buffers)
        cids.resize((size_t) a.prefill_chunk * (size_t) top_k);
        cids32.resize(cids.size());
        cw.resize(cids.size());
    }
    for (int t : a.stop)
        if (t < 0 || t >= n_vocab) { why = "--stop id " + std::to_string(t) + " is outside the vocabulary"; return false; }
    if (a.turn_token >= n_vocab) { why = "--turn-token is outside the vocabulary"; return false; }
    return true;
}

// = ds4_generate.cpp:360-390 @b588dbd1 ---- one token through every layer (no --pos-offset, no phase timing)
bool Server::step(int tid, int pos) {
    if (pos == test_fail_pos) { err = "test failure (DS4_SERVE_TEST_FAIL)"; return false; }
    if (a.route_bias != 0.0f && !mc.cpu_only)
        for (int l = 0; l < n_layer; ++l) {   // the residency the cache has NOW (admissions move it per token)
            for (int64_t e = 0; e < (int64_t) rb.size(); ++e) rb[(size_t) e] = tier.resident(l, e) ? a.route_bias : 0.0f;
            dense.set_route_bias(l, rb.data());
        }
    if (!dense.begin_token(tid)) { err = dense.last_error(); return false; }
    for (int l = 0; l < n_layer; ++l) {
        if (!mc.cpu_only && mc.pf_b > 0 && dense.predict(l, pred_i.data(), kPredW)) {
            int n = 0;
            for (int i = 0; i < kPredW && pred_i[(size_t) i] >= 0; ++i) pred32[(size_t) n++] = pred_i[(size_t) i];
            if (n > 0) tier.prefetch(l, pred32.data(), n);
        }
        const float * x = nullptr;
        ggml_tensor * x_dev = nullptr;
        if (!dense.attn_router(l, pos, tid, ids_i.data(), w.data(), &x, &x_dev)) { err = dense.last_error(); return false; }
        for (int k = 0; k < top_k; ++k) ids32[(size_t) k] = ids_i[(size_t) k];
        if (!tier.run(l, ids32.data(), w.data(), x, routed.data())) {
            err = "tier.run refused at layer " + std::to_string(l);
            return false;
        }
        if (!dense.finish_layer(l, routed.data())) { err = dense.last_error(); return false; }
    }
    return true;
}

// = ds4_generate.cpp step_n (mab776 M7b) ---- n <= max_tokens() consecutive tokens through every trunk layer in one pass
bool Server::step_n(const int * toks, int n, int pos0) {
    if (a.route_bias != 0.0f && !mc.cpu_only)
        for (int l = 0; l < n_layer; ++l) {
            for (int64_t e = 0; e < (int64_t) rb.size(); ++e) rb[(size_t) e] = tier.resident(l, e) ? a.route_bias : 0.0f;
            dense.set_route_bias(l, rb.data());
        }
    if (!dense.begin_tokens(toks, n)) { err = dense.last_error(); return false; }
    for (int l = 0; l < n_layer; ++l) {
        if (!mc.cpu_only && mc.pf_b > 0 && dense.predict(l, pred_i.data(), kPredW)) {   // token 0's hint
            int m = 0;
            for (int i = 0; i < kPredW && pred_i[(size_t) i] >= 0; ++i) pred32[(size_t) m++] = pred_i[(size_t) i];
            if (m > 0) tier.prefetch(l, pred32.data(), m);
        }
        const float * x = nullptr;
        if (!dense.attn_router_n(l, pos0, n, ids_n.data(), w_n.data(), &x)) { err = dense.last_error(); return false; }
        for (int k = 0; k < top_k * n; ++k) ids32_n[(size_t) k] = ids_n[(size_t) k];
        if (!tier.run_multi(l, n, ids32_n.data(), w_n.data(), x, routed_n.data())) {
            err = "tier.run_multi refused at layer " + std::to_string(l);
            return false;
        }
        if (!dense.finish_layer_n(l, n, routed_n.data())) { err = dense.last_error(); return false; }
    }
    return true;
}

// = ds4_generate.cpp mtp_draft_n (mab776 M7b): the MTP block at positions pos0..pos0+n-1 (the decode state's last pass
// rows) with the tokens that follow them -> the last position's argmax, the draft for pos0 + n + 1
bool Server::mtp_draft_n(const int * next, int n, int pos0, int * draft) {
    struct Timer { Server & s; double t0; ~Timer() { s.t_mtp_all += now_ms() - t0; ++s.n_mtp_calls; } } tm{ *this, now_ms() };
    const float * x = nullptr;
    if (!dense.mtp_begin(next, n)) { err = dense.last_error(); return false; }
    if (!dense.attn_router_n(il_mtp, pos0, n, mids.data(), mw.data(), &x)) { err = dense.last_error(); return false; }
    if (a.mtp_device >= 0) {   // one batched call on the GPU (the CPU keeps its per-token graph: the verified one)
        if (!mtp_x.run_n(x, mids.data(), mw.data(), mrouted.data(), n)) { err = "MTP experts failed"; return false; }
    } else {
        for (int t = 0; t < n; ++t)
            if (!mtp_x.run(x + (size_t) t * n_embd, mids.data() + t * top_k, mw.data() + t * top_k,
                           mrouted.data() + (size_t) t * n_embd)) { err = "MTP experts failed"; return false; }
    }
    if (!dense.finish_layer_n(il_mtp, n, mrouted.data())) { err = dense.last_error(); return false; }
    // the head for the last position only, its argmax on the device (mab776 M7c; the window needs no head at all)
    if (draft && !dense.mtp_argmax(n - 1, draft)) { err = dense.last_error(); return false; }
    return true;
}

// mab776 M7c: one chained draft - the MTP block at position `pos` for the token `next` (the previous draft), its h the
// MTP block's own output for pos - 1 in state column `src` (the trunk has not seen these tokens yet) -> the draft for
// pos + 2.  The MTP row it writes at pos is speculative: the next verified pass's mtp_draft_n rewrites it.
bool Server::mtp_chain(int next, int src, int pos, int * draft) {
    struct Timer { Server & s; double t0; ~Timer() { s.t_mtp_all += now_ms() - t0; ++s.n_mtp_calls; } } tm{ *this, now_ms() };
    const float * x = nullptr;
    if (!dense.mtp_chain_begin(next, src)) { err = dense.last_error(); return false; }
    if (!dense.attn_router_n(il_mtp, pos, 1, mids.data(), mw.data(), &x)) { err = dense.last_error(); return false; }
    if (a.mtp_device >= 0) {
        if (!mtp_x.run_n(x, mids.data(), mw.data(), mrouted.data(), 1)) { err = "MTP experts failed"; return false; }
    } else if (!mtp_x.run(x, mids.data(), mw.data(), mrouted.data())) {
        err = "MTP experts failed";
        return false;
    }
    if (!dense.finish_layer_n(il_mtp, 1, mrouted.data())) { err = dense.last_error(); return false; }
    if (!dense.mtp_argmax(0, draft)) { err = dense.last_error(); return false; }
    return true;
}

// mab776 M7b: the MTP head's sliding window over a prompt chunk's positions [c0, c0 + m) that fall in [W, n - 1) - the
// positions the first decode drafts attend to (W = max(the reuse point, n - 1 - SWA)); from the chunk's own trunk
// states (prefill_state_to_decode), before the next chunk overwrites them.  Positions below the reuse point keep the
// MTP rows the restored state holds (the one at R - 1 may have seen another next token: drafts only, every draft is
// verified).
bool Server::mtp_fill_chunk(const std::vector<int> & ids, int c0, int m, int W, int n) {
    const double t = now_ms();
    int p = std::max(W, c0);
    const int end = std::min(c0 + m, n - 1);
    while (p < end) {
        const int cnt = std::min(NT, end - p);
        if (!dense.prefill_state_to_decode(p - c0, cnt)) { err = dense.last_error(); return false; }
        if (!mtp_draft_n(ids.data() + p + 1, cnt, p, nullptr)) return false;
        p += cnt;
    }
    t_mtp_win += now_ms() - t;
    return true;
}

// = ds4_generate.cpp:475-543 @b588dbd1 ---- one prompt chunk, n <= --prefill-chunk tokens at positions pos0..
// (no --ppl, no logits: the last prompt token always goes through step()).  QUIT / EOF are checked between layers:
// a chunk of the real model takes seconds, and the server's close() allows 20 s; `quit` then says the state is half
// written (the caller exits).
bool Server::chunk(const int * toks, int n, int pos0, bool & quit) {
    quit = false;
    if (test_fail_pos >= pos0 && test_fail_pos < pos0 + n) { err = "test failure (DS4_SERVE_TEST_FAIL)"; return false; }
    float * crouted = dense.prefill_routed_buffer();   // pinned on CUDA; fetched per chunk: prefill_release frees it
    void * croute_dev = mc.cpu_only ? nullptr : dense.prefill_routed_device();   // the tier writes the sums there
    if (!crouted) { err = dense.last_error(); return false; }
    if (!dense.prefill_begin(toks, n, pos0)) { err = dense.last_error(); dense.prefill_end(); return false; }
    for (int l = 0; l < n_layer; ++l) {
        if (in.quit.load()) { dense.prefill_end(); quit = true; return false; }
        const float * fn = nullptr;
        if (!dense.prefill_attn(l, cids.data(), cw.data(), &fn)) { err = dense.last_error(); dense.prefill_end(); return false; }
        for (int k = 0; k < n * top_k; ++k) cids32[(size_t) k] = cids[(size_t) k];
        if (!(croute_dev ? tier.run_chunk_dev(l, n, cids32.data(), cw.data(), fn, croute_dev)
                         : tier.run_chunk(l, n, cids32.data(), cw.data(), fn, crouted))) {
            err = "tier.run_chunk refused at layer " + std::to_string(l);
            dense.prefill_end();
            return false;
        }
        if (!dense.prefill_finish(l, croute_dev ? nullptr : crouted)) { err = dense.last_error(); dense.prefill_end(); return false; }
    }
    dense.prefill_end();
    return true;
}

// mab776 M6: the elastic GPU 0 cache.  Between requests it is grown to the decode budget (--vram-grow GiB left free);
// a prompt that reads chunks first gives the tail back (its experts each keep an arena copy), so the chunk buffers
// have the --vram-margin room the one-shot runs measured, and end_prompt grows it again from the arena (~0.1 s for
// ~900 slots on the M7a runs).  A short follow-up that reads no chunk keeps the grown cache.
void Server::before_chunk() {
    if (shrunk || !tier.cache_grown()) return;
    const double t = now_ms();
    std::string e;
    if (tier.shrink_cache(e) < 0)
        std::fprintf(stderr, "ds4_serve: the cache did not shrink before the prompt (%s): its chunk buffers may not fit\n",
                     e.c_str());
    t_shrink = now_ms() - t;
    shrunk = true;   // even after a failed shrink: the grow below maps whatever is free again
}

// = ds4_generate.cpp:544-545 @b588dbd1.  mab776 M6: the chunk buffers' device side goes back after every prompt (the
// pinned host side stays: page-locking 2 GiB again would cost a short turn more than its prompt), then the cache
// grows back to the decode budget.
void Server::end_prompt() {
    tier.release_chunk(/*keep_host=*/true);
    dense.prefill_release();
    t_grow = 0;
    if (shrunk) {
        const double t = now_ms();
        std::string e;
        if (tier.grow_cache(a.vram_grow, e, /*reseed_lru=*/false) < 0)
            std::fprintf(stderr, "ds4_serve: the cache did not grow back after the prompt (%s): decode on the prompt "
                                 "budget\n", e.c_str());
        t_grow = now_ms() - t;
        shrunk = false;
    }
}

Choice Server::choose(const std::vector<int> & ids) const {
    return best_reuse(live, ck, slots_ck, ids, dense.max_tokens());
}

// ---- conversation parking (mab776 M6)
// The RAM a park may take: the budget (oldest parked conversations evicted first) and what the container and the
// host have left - parking must never be what pushes the engine into the OOM killer (cgroup memory.max is the hard
// wall; 2 GiB stays free under it, and 4 GiB of the host's MemAvailable).
bool Server::park_room(size_t need) {
    size_t budget = (size_t) a.park_mib << 20;
    if (const char * tb = std::getenv("DS4_SERVE_TEST_PARK_BYTES")) budget = (size_t) std::strtoull(tb, nullptr, 10);
    if (need > budget) return false;
    auto evict_oldest = [&]() {
        if (parked.empty()) return false;
        size_t o = 0;
        for (size_t i = 1; i < parked.size(); ++i) if (parked[i].tick < parked[o].tick) o = i;
        std::fprintf(stderr, "ds4_serve: park: evicting a parked conversation of %zu tokens (%.0f MiB)\n",
                     parked[o].live.size(), parked[o].blob.size() / 1048576.0);
        park_used -= parked[o].blob.size();
        parked.erase(parked.begin() + (std::ptrdiff_t) o);
        return true;
    };
    while (park_used + need > budget)
        if (!evict_oldest()) return false;
    auto room = [&]() -> double {   // GiB the container / host could still give
        double r = 1e18;
        auto rd = [](const char * f, double & v) {   // one number from a cgroup file ("max" = no limit)
            std::FILE * fp = std::fopen(f, "r");
            if (!fp) return false;
            char b[64] = {0};
            const bool ok = std::fgets(b, sizeof b, fp) != nullptr;
            std::fclose(fp);
            if (!ok || std::strncmp(b, "max", 3) == 0) return false;
            v = std::strtod(b, nullptr);
            return true;
        };
        double mx = 0, cur = 0;
        if (rd("/sys/fs/cgroup/memory.max", mx) && rd("/sys/fs/cgroup/memory.current", cur)) {
            // memory.current counts the page cache the cgroup's reads filled (the model file: tens of GB, reclaimed
            // under pressure) - only what cannot be reclaimed is in use
            double file = 0;
            if (std::FILE * fp = std::fopen("/sys/fs/cgroup/memory.stat", "r")) {
                char line[256];
                while (std::fgets(line, sizeof line, fp))
                    if (std::strncmp(line, "active_file ", 12) == 0) file += std::strtod(line + 12, nullptr);
                    else if (std::strncmp(line, "inactive_file ", 14) == 0) file += std::strtod(line + 14, nullptr);
                std::fclose(fp);
            }
            r = std::min(r, (mx - std::max(0.0, cur - file)) / 1073741824.0 - 2.0);
        }
        if (std::FILE * fp = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, fp))
                if (std::strncmp(line, "MemAvailable:", 13) == 0) { r = std::min(r, std::strtod(line + 13, nullptr) / 1048576.0 - 4.0); break; }
            std::fclose(fp);
        }
        return r;
    };
    while (room() < need / 1073741824.0)
        if (!evict_oldest()) return false;
    return true;
}

// the live conversation into host RAM (when parking is on and it is worth it); its state stays in the engine too,
// so a caller that then fails to use the slot it freed loses nothing
void Server::park_live(const char * why) {
    if (a.park_mib <= 0 || (int) live.size() < a.park_min) return;
    int max_pos = (int) live.size();
    for (int s = 0; s < slots_ck; ++s) max_pos = std::max(max_pos, ck[s].pos);
    const size_t need = dense.park_bytes(max_pos);
    if (!park_room(need)) {
        std::fprintf(stderr, "ds4_serve: park: no room for a conversation of %zu tokens (%.0f MiB): not parked\n",
                     live.size(), need / 1048576.0);
        return;
    }
    const double t = now_ms();
    Parked P;
    P.live = live;
    for (int s = 0; s < kSlots; ++s) P.ck[s] = ck[s];
    P.tick = ++park_tick;
    if (!dense.park(max_pos, P.blob)) {
        std::fprintf(stderr, "ds4_serve: park failed: %s\n", dense.last_error().c_str());
        return;
    }
    park_used += P.blob.size();
    char b[192];
    std::snprintf(b, sizeof b, " parked %zu tokens (%s, %.1f MiB = %zu B, %.0f ms);", live.size(), why,
                  P.blob.size() / 1048576.0, P.blob.size(), now_ms() - t);
    park_note += b;
    parked.push_back(std::move(P));
}

// a parked conversation (already taken out of the list) becomes the live state
bool Server::unpark(Parked & P) {
    const double t = now_ms();
    const bool ok = dense.unpark(P.blob);
    if (ok) {
        live = P.live;
        for (int s = 0; s < kSlots; ++s) ck[s] = P.ck[s];
        char b[128];
        std::snprintf(b, sizeof b, " unparked %zu tokens (%.0f MiB, %.0f ms);", live.size(), P.blob.size() / 1048576.0,
                      now_ms() - t);
        park_note += b;
    } else {
        std::fprintf(stderr, "ds4_serve: unpark failed (%s): the conversation is read again\n", dense.last_error().c_str());
        dense.reset();
        live.clear();
        forget_all();
    }
    return ok;
}

void Server::forget_all() {
    for (Ckpt & c : ck) { c.pos = -1; c.ids.clear(); }
}

// a checkpoint above the resume point describes positions about to be rewritten: it is gone
void Server::drop_above(int R) {
    for (int s = 0; s < kSlots; ++s)
        if (ck[s].pos > R) { ck[s].pos = -1; ck[s].ids.clear(); }
}

// -> R (the reused tokens); the state and `live` now hold exactly ids[0, R)
int Server::apply(Choice & c) {
    switch (c.kind) {
        case Kind::Append: break;
        case Kind::Rewind: live.resize((size_t) c.R); break;
        case Kind::Restore:
            if (dense.restore(c.slot)) { live = ck[c.slot].ids; break; }
            std::fprintf(stderr, "ds4_serve: restore of checkpoint %s failed (%s): starting over\n", slot_name(c.slot),
                         dense.last_error().c_str());
            c.kind = Kind::Reset;
            c.R = 0;
            [[fallthrough]];
        case Kind::Reset:
            dense.reset();
            live.clear();
            forget_all();
            break;
    }
    drop_above(c.R);
    return c.R;
}

void Server::take_ckpt(const std::vector<int> & ids, int pos, int slot) {
    if (slot >= slots_ck || pos < 1 || (int) live.size() != pos) return;
    Ckpt & c = ck[slot];
    if (c.pos == pos && std::equal(c.ids.begin(), c.ids.end(), ids.begin())) return;   // already there
    if (!dense.snapshot(pos, slot)) {
        std::fprintf(stderr, "ds4_serve: checkpoint %s at %d failed: %s\n", slot_name(slot), pos, dense.last_error().c_str());
        c.pos = -1;
        c.ids.clear();
        return;
    }
    c.pos = pos;
    c.ids.assign(ids.begin(), ids.begin() + pos);
}

// DS4_SERVE_TEST_DELAY_MS: a pause after each chunk / prompt token / decode step, cut short by STOP or QUIT (the STOP
// tests on the millisecond-per-token mini model)
void Server::delay(const std::function<bool()> & stopped) const {
    if (test_delay_ms <= 0) return;
    const double end = now_ms() + test_delay_ms;
    while (!stopped()) {
        const double left = end - now_ms();
        if (left <= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds((int) std::min(5.0, left) + 1));
    }
}

void Server::serve_gen(const std::string & line, int64_t seq) {
    Gen g;
    const std::string bad = parse_gen(line, g, n_vocab, ctx);
    if (!bad.empty()) { proto.err(bad); return; }
    const std::vector<int> & ids = g.ids;
    const int n = (int) ids.size();
    const bool ckpt = g.ckpt && slots_ck > 0;
    g.sp.seed = g.seed ? g.seed
              : a.seed ? a.seed : (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    auto stopped = [&]() { return in.stop_seq.load() >= seq || in.quit.load(); };
    const double t0 = now_ms();

    // ---- reuse: the live state, or a parked conversation when it offers clearly more (mab776 M6)
    park_note.clear();
    Choice c = choose(ids);
    if (!parked.empty()) {
        constexpr int kParkGain = 64;   // a swap costs two host copies (~0.1 s at 127k): worth ~64 tokens of prompt
        size_t pick = parked.size();
        Choice best;
        for (size_t i = 0; i < parked.size(); ++i) {
            const Choice ci = best_reuse(parked[i].live, parked[i].ck, slots_ck, ids, dense.max_tokens());
            if (ci.R > best.R) { best = ci; pick = i; }
        }
        if (pick < parked.size() && best.R > c.R + kParkGain) {
            // out of the list first: parking the live conversation may evict parked ones, never this one
            Parked P = std::move(parked[pick]);
            parked.erase(parked.begin() + (std::ptrdiff_t) pick);
            park_used -= P.blob.size();
            park_live("switched away");
            if (unpark(P)) c = choose(ids);
            else c = Choice{};
        }
    }
    // the request starts over: the conversation the state holds is kept for a return to it.  (A restore of the turn
    // checkpoint is every new turn's normal path - DeepSeek's template rewrites the last <think> - and parks nothing.)
    if (c.kind == Kind::Reset) park_live("new conversation");
    const int L0 = (int) live.size();
    const int R = apply(c);
    std::fprintf(stderr, "ds4_serve: GEN #%lld: %d tokens, max_new %lld; live %d, common %zu -> %s%s%s, reused %d;%s parked "
                         "%zu (%.0f MiB)\n",
                 (long long) seq, n, g.max_new, L0, c.lcp, kind_name(c.kind), c.kind == Kind::Restore ? " " : "",
                 c.kind == Kind::Restore ? slot_name(c.slot) : "", R, park_note.c_str(), parked.size(),
                 park_used / 1048576.0);
    proto.line("RESUME %d", R);

    // ---- the prompt: [R, n-1), split at the checkpoint positions; then the last token through step()
    const int ck_slot = (c.kind == Kind::Append || c.kind == Kind::Rewind) ? kLast : kTurn;
    int user_at = -1;
    if (ckpt && a.turn_token >= 0)
        for (int i = n - 2; i > R; --i)
            if (ids[(size_t) i] == a.turn_token) { user_at = i; break; }
    bool cancelled = false, failed = false, quit = false;
    std::string ferr;
    int pp_mark = R;   // the decode-loop PP cadence
    // mab776 M7b: the MTP window = the positions the first drafts attend to, of those this request reads
    const int W = il_mtp >= 0 ? std::max(R, n - 1 - (int) dense.geom().n_swa) : INT32_MAX;
    t_mtp_win = 0;
    t_mtp_all = 0;
    n_mtp_calls = 0;
    drafts = accepted = 0;
    d_off.clear();
    d_acc.clear();
    passes = 0;
    t_pass = 0;
    auto pp = [&](int pos) {
        const double ms = now_ms() - t0;
        proto.line("PP %d %d %.0f %.1f", pos, n, ms, ms > 0 ? 1000.0 * (pos - R) / ms : 0.0);
    };
    std::vector<int> stops;
    if (user_at > R) stops.push_back(user_at);
    stops.push_back(n - 1);
    for (int to : stops) {
        const int slot = to == user_at ? kUser : ck_slot;
        int pos = (int) live.size();
        if (pos == to && ckpt) take_ckpt(ids, to, slot);   // nothing to read before it (R == n - 1)
        while (pos < to) {
            const int left = to - pos;
            if (a.prefill_chunk > 0 && left >= a.chunk_min) {
                const int m = std::min(a.prefill_chunk, left);
                before_chunk();
                if (!chunk(ids.data() + pos, m, pos, quit)) {
                    failed = !quit;
                    ferr = "prefill failed: " + err;
                    break;
                }
                if (il_mtp >= 0 && !mtp_fill_chunk(ids, pos, m, W, n)) {   // mab776 M7b
                    failed = true;
                    ferr = "MTP window failed: " + err;
                    break;
                }
                live.insert(live.end(), ids.begin() + pos, ids.begin() + pos + m);
                pos += m;
                pp(pos);
                pp_mark = pos;
            } else {
                if (!step(ids[(size_t) pos], pos)) { failed = true; ferr = "prefill failed at " + std::to_string(pos) + ": " + err; break; }
                if (il_mtp >= 0 && pos >= W && pos < n - 1) {   // mab776 M7b: the MTP window, token by token
                    const double tm = now_ms();
                    if (!mtp_draft_n(&ids[(size_t) pos + 1], 1, pos, nullptr)) { failed = true; ferr = "MTP window failed: " + err; break; }
                    t_mtp_win += now_ms() - tm;
                }
                live.push_back(ids[(size_t) pos]);
                ++pos;
                if (pos - pp_mark >= 64) { pp(pos); pp_mark = pos; }
            }
            if (pos == to && ckpt) take_ckpt(ids, to, slot);   // at the boundary, before a STOP can end the read
            delay(stopped);
            if (stopped()) { cancelled = true; break; }
        }
        if (failed || quit || cancelled) break;
    }
    end_prompt();
    if (quit) { live.clear(); return; }   // half a chunk written: we exit (QUIT / EOF)
    if (!failed && !cancelled && stopped()) cancelled = true;
    if (!failed && !cancelled) {
        if (!step(ids[(size_t) n - 1], n - 1)) { failed = true; ferr = "prefill failed at " + std::to_string(n - 1) + ": " + err; }
        else live.push_back(ids[(size_t) n - 1]);
    }
    const int reached = std::min((int) live.size(), n);   // prompt tokens the state holds
    const double prompt_ms = now_ms() - t0;
    if (!failed && !cancelled) pp(n);

    // ---- decode (= ds4_generate.cpp:682-709 @b588dbd1, with the sampler chain and STOP)
    int gen = 0;
    const char * finish = cancelled ? "cancel" : "length";
    ds4::Ds4MoeStats st;
    double dec_ms = 0;
    std::FILE * tokl = nullptr;   // DS4_DUMP_TOKLOGITS=<prefix>: this request's logits rows in <prefix>.<seq>
    if (const char * f = std::getenv("DS4_DUMP_TOKLOGITS"))
        if (!failed && !cancelled) tokl = std::fopen((std::string(f) + "." + std::to_string(seq)).c_str(), "wb");
    if (!failed && !cancelled) {
        tier.reset_stats();
        std::vector<int> hist;   // the penalty window's source: the prompt, then every generated token
        if (g.sp.penalty_last_n > 0) {
            const int keep = std::min(n, ds4serve::kPenaltyWindowCap);
            hist.assign(ids.end() - keep, ids.end());
        }
        const double td0 = now_ms();
        int pos = n;   // the position the next generated token is fed at
        // emits one token; true when the request ends with it (a stop id, or max_new)
        auto emit = [&](int tok, const float * row, int nv) -> bool {
            if (tokl) std::fwrite(row, 4, (size_t) nv, tokl);
            proto.line("T %d", tok);
            ++gen;
            if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end()) { finish = "stop"; return true; }
            if (gen >= g.max_new) { finish = "length"; return true; }
            return false;
        };
        oracle.clear();
        n_fixed = 0;
        if (const char * f = std::getenv("DS4_SERVE_TEST_ORACLE")) {
            if (std::FILE * fp = std::fopen(f, "r")) {
                int v = 0;
                while (std::fscanf(fp, "%d", &v) == 1) oracle.push_back(v);
                std::fclose(fp);
            }
            corrupt_every = std::getenv("DS4_SERVE_TEST_CORRUPT") ? std::atoi(std::getenv("DS4_SERVE_TEST_CORRUPT")) : 0;
        }
        force = std::getenv("DS4_SERVE_TEST_FORCE") != nullptr && !oracle.empty();
        auto forced = [&](int tok) { return force && gen < (int) oracle.size() ? oracle[(size_t) gen] : tok; };
        if (il_mtp >= 0 && a.verify && g.mtp_k != 0) {
            // mab776 M7b/M7c, speculative decoding (= ds4_generate's --verify loop, with the sampler): `cur` = the newest
            // token (emitted, not yet fed) at position `pos`; dr[0..K-1] = the MTP head's guesses for pos + 1 .. pos + K
            // - dr[0] from the trunk's states (the MTP block over the positions the last pass verified), dr[j] from the
            // MTP block's own output one position earlier (chained: the trunk has not seen those tokens).  One pass
            // feeds cur and the K drafts; rows 0..K-1 decide the drafts in order (ds4serve::verify_row: exact
            // speculative sampling) up to the first rejection, whose row gives the replacement token, and when all K
            // pass row K gives one more.  Counters = the rows' positions, as in plain decoding.
            const int K = std::clamp(g.mtp_k > 0 ? g.mtp_k : a.mtp_k, 1, NT - 1);   // mtp_k=0: the plain loop below
            const float * lg = nullptr;
            int nv = 0;
            bool done = false;
            int cur = -1;
            std::vector<int> dr((size_t) K), toks((size_t) K + 1), nxt((size_t) K + 1);
            auto fix_draft = [&](int j) {   // dr[j] guesses the token emitted at index gen + j
                if (oracle.empty() || gen + j >= (int) oracle.size()) return;
                dr[(size_t) j] = oracle[(size_t) (gen + j)];
                if (corrupt_every > 0 && ++n_fixed % corrupt_every == 0) dr[(size_t) j] = (dr[(size_t) j] + 1) % n_vocab;
            };
            // the MTP block at the verified positions p0 .. p0+nn-1 with the tokens after them (its last column guesses
            // p0 + nn + 1), then K - 1 chained drafts at the positions after it
            auto draft_chain = [&](const int * next, int nn, int p0) -> bool {
                if (!mtp_draft_n(next, nn, p0, &dr[0])) return false;
                fix_draft(0);
                for (int j = 1; j < K; ++j) {
                    if (!mtp_chain(dr[(size_t) j - 1], j == 1 ? nn - 1 : 0, p0 + nn - 1 + j, &dr[(size_t) j])) return false;
                    fix_draft(j);
                }
                return true;
            };
            d_off.assign((size_t) K, 0);
            d_acc.assign((size_t) K, 0);
            passes = 0;
            t_pass = 0;
            if (!dense.logits(&lg, &nv)) { failed = true; ferr = "decode failed at pos " + std::to_string(n - 1) + ": " + dense.last_error(); }
            else {
                cur = forced(ds4serve::sample_row(lg, nv, g.sp, (uint64_t) (n - 1), hist.data(), hist.size(), scratch));
                done = emit(cur, lg, nv);
                if (g.sp.penalty_last_n > 0) hist.push_back(cur);
                if (!done && !draft_chain(&cur, 1, n - 1)) { failed = true; ferr = "MTP draft failed: " + err; }
            }
            while (!failed && !done) {
                if (stopped()) { finish = "cancel"; break; }
                toks[0] = cur;
                for (int j = 0; j < K; ++j) toks[(size_t) j + 1] = dr[(size_t) j];
                const double tp = now_ms();
                if (!step_n(toks.data(), K + 1, pos)) { failed = true; ferr = "verify pass failed at pos " + std::to_string(pos) + ": " + err; break; }
                if (!dense.logits_n(K + 1, &lg, &nv)) { failed = true; ferr = "verify logits failed: " + dense.last_error(); break; }
                t_pass += now_ms() - tp;
                ++passes;
                live.push_back(cur);   // position pos holds `cur` now
                drafts += K;
                int A = 0;             // drafts accepted: positions pos + 1 .. pos + A hold them
                int last = -1;         // the token after them (emitted, not fed): the next pass's `cur`
                for (int j = 0; j < K && !done; ++j) {
                    ++d_off[(size_t) j];
                    bool acc = false;
                    const float * row = lg + (size_t) j * nv;
                    last = ds4serve::verify_row(row, nv, g.sp, (uint64_t) (pos + j), hist.data(), hist.size(), scratch,
                                                dr[(size_t) j], &acc);
                    if (force && gen < (int) oracle.size()) { last = oracle[(size_t) gen]; acc = last == dr[(size_t) j]; }
                    if (acc) {
                        ++d_acc[(size_t) j];
                        ++accepted;
                        ++A;
                        live.push_back(last);   // position pos + 1 + j holds the draft, which is `last`
                    }
                    done = emit(last, row, nv);
                    if (!done && g.sp.penalty_last_n > 0) hist.push_back(last);
                    if (!acc) break;   // positions above pos + A hold rejected drafts: later passes rewrite them
                }
                if (done) break;
                if (A == K) {          // every draft passed: row K's token for free
                    const float * row = lg + (size_t) K * nv;
                    last = forced(ds4serve::sample_row(row, nv, g.sp, (uint64_t) (pos + K), hist.data(), hist.size(), scratch));
                    if ((done = emit(last, row, nv))) break;
                    if (g.sp.penalty_last_n > 0) hist.push_back(last);
                }
                for (int j = 0; j < A; ++j) nxt[(size_t) j] = dr[(size_t) j];
                nxt[(size_t) A] = last;
                if (!draft_chain(nxt.data(), A + 1, pos)) { failed = true; ferr = "MTP draft failed: " + err; break; }
                cur = last;
                pos += A + 1;
                delay(stopped);
            }
        } else {
            for (;;) {
                if (stopped()) { finish = "cancel"; break; }
                const float * lg = nullptr;
                int nv = 0;
                if (!dense.logits(&lg, &nv)) { failed = true; ferr = "decode failed at pos " + std::to_string(pos - 1) + ": " + dense.last_error(); break; }
                const int tok = forced(ds4serve::sample_row(lg, nv, g.sp, (uint64_t) (pos - 1), hist.data(), hist.size(), scratch));
                if (emit(tok, lg, nv)) break;   // the last token needs no forward pass
                if (!step(tok, pos)) { failed = true; ferr = "decode failed at pos " + std::to_string(pos) + ": " + err; break; }
                live.push_back(tok);
                if (g.sp.penalty_last_n > 0) hist.push_back(tok);
                ++pos;
                delay(stopped);
            }
        }
        dec_ms = now_ms() - td0;
        st = tier.stats();
    }
    if (tokl) std::fclose(tokl);
    if (failed) {   // the state is unknown: start over; exactly one terminal line, and never "ERR verify..."
        dense.reset();
        live.clear();
        forget_all();
        std::fprintf(stderr, "ds4_serve: GEN #%lld failed: %s - the state was reset\n", (long long) seq, ferr.c_str());
        proto.err(ferr);
        return;
    }
    const int read_n = std::max(0, reached - R);
    proto.line("DONE %d %d %.1f %.1f %s %lld %lld %d %lld %lld 0 %lld 0 %d", gen, n, prompt_ms, dec_ms, finish,
               (long long) accepted, (long long) drafts, R, (long long) st.hits, (long long) st.lookups(),
               (long long) st.file_tier, read_n);
    // serve/server.py's ENGINE_REQUEST shape (STRATA_REQUEST_LINES echoes it)
    char read_txt[64];
    if (std::strcmp(finish, "cancel") == 0 && reached < n)
        std::snprintf(read_txt, sizeof read_txt, "%d of %d", read_n, n - R);
    else
        std::snprintf(read_txt, sizeof read_txt, "%d", n - R);
    int nck = 0;
    for (int s = 0; s < slots_ck; ++s) nck += ck[s].pos > 0;
    char htxt[128] = "";   // the decode's expert lookups by tier (the slot effect, apart from draft acceptance)
    if (st.lookups() > 0) {
        const double L = (double) st.lookups();
        std::snprintf(htxt, sizeof htxt, "; experts GPU0 %.1f%% GPU1 %.1f%% CPU %.1f%% PCIe %.1f%%", 100.0 * st.hits / L,
                      100.0 * st.hits1 / L, 100.0 * st.cpu / L, 100.0 * st.pcie / L);
    }
    char dtxt[384] = "";
    if (il_mtp >= 0 && d_off.empty()) {
        std::snprintf(dtxt, sizeof dtxt, "; drafts off (mtp_k=0)");
    } else if (il_mtp >= 0) {
        // mab776 M7c: acceptance by chain position (of the rows examined: draft j is examined when 0..j-1 passed), the
        // tokens a pass gives, and a pass's ms (step_n + logits_n) apart from the drafts'
        std::string by;
        for (size_t j = 0; j < d_off.size(); ++j) {
            char b[48];
            std::snprintf(b, sizeof b, "%s%lld/%lld", j ? " " : "", (long long) d_acc[j], (long long) d_off[j]);
            by += b;
        }
        std::snprintf(dtxt, sizeof dtxt, "; drafts %lld/%lld accepted (K=%zu, by position %s; %lld passes of %.1f ms, "
                                         "%.2f tokens each), MTP window %.0f ms, %.2f ms/draft",
                      (long long) accepted, (long long) drafts, d_off.size(), by.c_str(), (long long) passes,
                      passes > 0 ? t_pass / (double) passes : 0.0, passes > 0 ? (double) gen / (double) passes : 0.0,
                      t_mtp_win, drafts > 0 ? (t_mtp_all - t_mtp_win) / (double) drafts : 0.0);
    }
    std::fprintf(stderr, "ds4_serve: prompt %d tokens = %d reused + %s read in %.0f ms (%.1f tok/s; cache shrink %.0f + "
                         "grow %.0f ms), %d generated in %.0f ms (%.1f tok/s), %s%s%s; live %zu, %d checkpoints [last %d, "
                         "turn %d, user %d]\n",
                 n, R, read_txt, prompt_ms, prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, t_shrink, t_grow, gen,
                 dec_ms, dec_ms > 0 ? 1000.0 * gen / dec_ms : 0.0, finish, dtxt, htxt, live.size(), nck, ck[kLast].pos,
                 ck[kTurn].pos, ck[kUser].pos);
    t_shrink = t_grow = 0;
}

void Server::loop() {
    std::string line;
    for (;;) {
        if (in.quit.load() || !in.next(line)) break;
        if (line == "QUIT") break;
        if (line.empty()) continue;   // nothing to answer (and never a stray terminal line)
        try {
            if (line.rfind("GEN ", 0) == 0) {
                serve_gen(line, ++gens);
            } else if (line.rfind("GENI ", 0) == 0) {
                ++gens;   // counted like a GEN: the STOP numbering stays in step with the reader's
                proto.err("this engine was started without --vision");
            } else if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
                proto.err("this engine was started without --vram-elastic");
            } else if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
                // a bare ERR here would make the server end the engine as out of step (session_file)
                proto.line("SERR invalid 0 this engine keeps no disk sessions");
            } else {
                proto.err("expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>");
            }
        } catch (const std::exception & e) {
            dense.reset();
            live.clear();
            forget_all();
            std::fprintf(stderr, "ds4_serve: internal error: %s - the state was reset\n", e.what());
            proto.err(std::string("internal: ") + e.what());
        }
    }
}

void Server::close() {
    tier.close();
}

}  // namespace

int main(int argc, char ** argv) {
    // = ds4_generate.cpp:181-186 @b588dbd1: ggml's DEBUG lines cost a stderr write each on the decode path
    if (!std::getenv("DS4_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) {
            if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
        }, nullptr);
    Args a;
    std::string why;
    if (!parse_args(argc, argv, a, why)) {
        std::fprintf(stderr, "ds4_serve: %s\n", why.c_str());
        usage();
        return 2;
    }
    // stdout = the protocol only: keep a private handle on it and point fd 1 at the log
    const int pfd = ::dup(1);
    std::FILE * pf = pfd >= 0 ? ::fdopen(pfd, "w") : nullptr;
    if (!pf || ::dup2(2, 1) < 0) { std::perror("ds4_serve: stdout"); return 1; }
    std::setvbuf(pf, nullptr, _IOFBF, 1 << 16);
    Proto proto{ pf };
    Input * in = new Input();   // never freed: the detached reader may outlive main()
    std::thread([in] { in->run(); }).detach();

    // the server object lives on the heap and is never destroyed: Ds4Dense's buffers go with the process (the
    // Impl destructor's frees would only cost time at exit), and tier.close() below joins the CPU pool first
    Server * srv = new Server(a, proto, *in);
    if (!srv->init(why)) {
        std::fprintf(stderr, "ds4_serve: %s\n", why.c_str());
        return 1;
    }
    if (in->quit.load()) { srv->close(); return 0; }   // the server gave up while the model loaded
    {
        std::string extra;
        if (!ggml_backend_is_cpu(srv->be)) {
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(srv->be), &fr, &tot);
            extra = " vram_free_mib=" + std::to_string((long long) (fr >> 20));
        }
        if (srv->tier.gpu1_device() >= 0)
            extra += " gpu1_experts=" + std::to_string((long long) srv->tier.resident1()) +
                     " gpu1_mib=" + std::to_string((long long) (srv->tier.gpu1_gib() * 1024.0));
        if (srv->a.vram_grow >= 0) extra += " elastic=1";
        if (srv->a.park_mib > 0) extra += " park_mib=" + std::to_string((long long) srv->a.park_mib);
        proto.line("INFO context=%lld engine=%s expert_slots=%lld arena_mib=%lld arena_experts=%lld file_tier=%lld "
                   "prefill_chunk=%d chunk_min=%d checkpoints=%d snapshot_kib=%lld turn_token=%d%s",
                   (long long) srv->ctx, kEngineVersion, (long long) srv->tier.resident(),
                   (long long) (srv->tier.arena_gib() * 1024.0), (long long) srv->tier.arena_experts(),
                   (long long) srv->tier.file_tier(), a.prefill_chunk, a.chunk_min, srv->slots_ck,
                   (long long) (srv->dense.snapshot_bytes() >> 10), a.turn_token, extra.c_str());
        proto.line("READY %lld stop", (long long) srv->ctx);
        std::fprintf(stderr, "ds4_serve: ready (context %lld, prefill chunk %d, %d checkpoint slots of %.1f MiB, stop ids",
                     (long long) srv->ctx, a.prefill_chunk, srv->slots_ck, srv->dense.snapshot_bytes() / 1048576.0);
        for (int t : srv->a.stop) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "%s)\n", srv->a.stop.empty() ? " none" : "");
    }
    srv->loop();
    std::fprintf(stderr, "ds4_serve: %s - exiting\n", in->eof.load() ? "stdin closed" : "QUIT");
    srv->close();
    std::fflush(pf);
    return 0;
}
