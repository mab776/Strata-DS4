// tools/ds4/mtp_experts.hpp - the DS4 MTP block's routed experts on the CPU (shared by ds4_generate and
// ds4_mtp_bench).
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "strata/artifact/gguf_reader.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4 {

// The MTP block's routed experts, v1: a CPU ggml mul_mat_id straight off the MTP file (MXFP4, bound zero-copy on its
// mmap) - ~13 MB per expert, 6 per draft.  Not in the expert tier (whose CUDA kernels do not do MXFP4).
// `resident`: copy the three expert tensors (3.2 GB) into a CPU backend buffer instead - pages the kernel cannot evict
// (warm mmap and the copy both run at ~23 GB/s; the mmap is slow only when the file's pages are not in RAM).
struct MtpExperts {
    std::unique_ptr<strata::GgufModel> model;
    ggml_context* wctx = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_t be = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_buffer_t rbuf = nullptr;   // `resident`: the copied experts
    std::vector<std::pair<const uint8_t*, size_t>> spans;   // the expert tensors' bytes in the file mapping
    ggml_gallocr_t allo = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor *t_x = nullptr, *t_id = nullptr, *t_w = nullptr, *out = nullptr;
    int64_t D = 0, K = 0;
    int64_t KIN = 0;        ///< the router's top-k (entries run() receives)
    /// > 0: only the `keep` highest-weight of the KIN routed experts are computed (a cheaper draft: the MTP head only
    /// has to rank the next token).  Set before init.  0 = all.
    int64_t keep = 0;
    /// mab776 M7b: >= 0 = the experts live on that GPU (ggml's GPU device index: CUDA_VISIBLE_DEVICES order) and run
    /// there - the three tensors (3.4 GB MXFP4) are copied in at init; -1 = the CPU (the default).  Set before init.
    int device = -1;
    std::vector<float> host;
    float clamp_ = 0.0f;
    ggml_tensor *gate_w_ = nullptr, *up_w_ = nullptr, *down_w_ = nullptr;
    /// run_n's graphs, one per token count (1..kMaxN): their own context, inputs, output and allocator
    static constexpr int kMaxN = 8;
    struct G {
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_gallocr_t allo = nullptr;
        ggml_tensor *x = nullptr, *id = nullptr, *w = nullptr, *out = nullptr;
    } gn[kMaxN + 1];

    bool init(const std::string& path, int il, int64_t n_embd, int64_t topk, float clamp, int threads, std::string& err,
              bool resident = false) {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(path));
        if (model->size() != 1) { err = "MTP experts: cannot open " + path + " (single file expected)"; return false; }
        D = n_embd; KIN = topk; K = keep > 0 ? std::min<int64_t>(keep, topk) : topk;
        clamp_ = clamp;
        if (device >= 0) {   // mab776 M7b: the device-th GPU of ggml's registry
            int seen = 0;
            for (size_t i = 0; i < ggml_backend_dev_count() && !be; ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
                if (seen++ == device) be = ggml_backend_dev_init(d, nullptr);
            }
            if (!be) { err = "MTP experts: no GPU device " + std::to_string(device) + " in this build"; return false; }
            resident = true;   // the device copy
        } else {
            be = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(be, threads);
        }
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        wctx = ggml_init(ip);
        ctx = ggml_init(ip);
        const strata::GgufFile& sh = model->shard(0);
        const auto& ts = sh.tensors();
        uint8_t* base = const_cast<uint8_t*>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
        wbuf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
        auto bind = [&](const std::string& n) -> ggml_tensor* {
            const strata::TensorInfo* ti = model->find(n);
            if (!ti) { err = "MTP experts: no " + n; return nullptr; }
            int64_t ne[4] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti->shape.size(); ++d) ne[d] = (int64_t) ti->shape[d];
            ggml_tensor* t = ggml_new_tensor(wctx, (ggml_type) ti->type, (int) ti->shape.size(), ne);
            spans.push_back({ sh.tensor_data(*ti), ggml_nbytes(t) });
            if (resident) return t;   // allocated and filled below
            if (ggml_backend_tensor_alloc(wbuf, t, const_cast<uint8_t*>(sh.tensor_data(*ti))) != GGML_STATUS_SUCCESS) {
                err = "MTP experts: cannot bind " + n; return nullptr;
            }
            return t;
        };
        const std::string p = "blk." + std::to_string(il) + ".";
        ggml_tensor *gate_w = bind(p + "ffn_gate_exps.weight"), *up_w = bind(p + "ffn_up_exps.weight"),
                    *down_w = bind(p + "ffn_down_exps.weight");
        if (!gate_w || !up_w || !down_w) return false;
        if (resident) {
            rbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_get_default_buffer_type(be));
            if (!rbuf) { err = "MTP experts: cannot allocate the resident copy"; return false; }
            ggml_tensor* ws[3] = { gate_w, up_w, down_w };
            for (int i = 0; i < 3; ++i) ggml_backend_tensor_set(ws[i], spans[(size_t) i].first, 0, spans[(size_t) i].second);
            for (auto& sp : spans)   // the file's pages are not needed again
                madvise(const_cast<uint8_t*>(sp.first) - ((uintptr_t) sp.first % 4096), sp.second, MADV_DONTNEED);
        }
        gate_w_ = gate_w; up_w_ = up_w; down_w_ = down_w;
        t_x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
        t_id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, 1);
        t_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, K, 1);
        for (ggml_tensor* t : { t_x, t_id, t_w }) ggml_set_input(t);
        ggml_tensor* up = ggml_clamp(ctx, ggml_mul_mat_id(ctx, up_w, t_x, t_id), -clamp, clamp);
        ggml_tensor* gate = ggml_clamp(ctx, ggml_mul_mat_id(ctx, gate_w, t_x, t_id), -INFINITY, clamp);
        ggml_tensor* down = ggml_mul(ctx, ggml_mul_mat_id(ctx, down_w, ggml_swiglu_split(ctx, gate, up), t_id), t_w);
        for (int64_t k = 0; k < K; ++k) {
            ggml_tensor* v = ggml_view_2d(ctx, down, D, 1, down->nb[2], (size_t) k * down->nb[1]);
            out = out ? ggml_add(ctx, out, v) : v;
        }
        ggml_set_output(out);
        gf = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(gf, out);
        allo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!allo || !ggml_gallocr_alloc_graph(allo, gf)) { err = "MTP experts: graph alloc failed"; return false; }
        host.resize((size_t) D);
        return true;
    }
    /// mab776 M7b: run_n's graph for n tokens (built on first use): the same ops as init's, batched over n
    bool build_n(int n) {
        G& g = gn[n];
        if (g.gf) return true;
        ggml_init_params ip = { 4 * 1024 * 1024, nullptr, true };
        g.ctx = ggml_init(ip);
        g.x  = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, D, 1, n);
        g.id = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, K, n);
        g.w  = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, K, n);
        for (ggml_tensor* t : { g.x, g.id, g.w }) ggml_set_input(t);
        ggml_tensor* up = ggml_clamp(g.ctx, ggml_mul_mat_id(g.ctx, up_w_, g.x, g.id), -clamp_, clamp_);
        ggml_tensor* gate = ggml_clamp(g.ctx, ggml_mul_mat_id(g.ctx, gate_w_, g.x, g.id), -INFINITY, clamp_);
        ggml_tensor* down = ggml_mul(g.ctx, ggml_mul_mat_id(g.ctx, down_w_, ggml_swiglu_split(g.ctx, gate, up), g.id), g.w);
        for (int64_t k = 0; k < K; ++k) {   // [D, K, n] -> [D, n]: the K weighted rows, added in order (as run's graph)
            ggml_tensor* v = ggml_view_2d(g.ctx, down, D, n, down->nb[2], (size_t) k * down->nb[1]);
            g.out = g.out ? ggml_add(g.ctx, g.out, v) : v;
        }
        ggml_set_output(g.out);
        g.gf = ggml_new_graph_custom(g.ctx, 256, false);
        ggml_build_forward_expand(g.gf, g.out);
        g.allo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        return g.allo && ggml_gallocr_alloc_graph(g.allo, g.gf);
    }
    /// mab776 M7b: n <= kMaxN tokens in one call - x [n][D], ids / w [n][KIN] -> o [n][D]
    bool run_n(const float* x, const int* ids, const float* w, float* o, int n) {
        if (n < 1 || n > kMaxN) return false;
        if (n == 1) return run(x, ids, w, o);
        if (!build_n(n)) return false;
        std::vector<int32_t> iv((size_t) (K * n));
        std::vector<float> wv((size_t) (K * n));
        std::vector<int> ord((size_t) KIN);
        for (int t = 0; t < n; ++t) {
            const int* it = ids + (size_t) t * KIN;
            const float* wt = w + (size_t) t * KIN;
            for (int64_t i = 0; i < KIN; ++i) ord[(size_t) i] = (int) i;
            if (K < KIN)
                std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) { return wt[a] > wt[b]; });
            for (int64_t i = 0; i < K; ++i) {
                iv[(size_t) (t * K + i)] = it[ord[(size_t) i]];
                wv[(size_t) (t * K + i)] = wt[ord[(size_t) i]];
            }
        }
        G& g = gn[n];
        ggml_backend_tensor_set(g.x, x, 0, (size_t) D * n * 4);
        ggml_backend_tensor_set(g.id, iv.data(), 0, iv.size() * 4);
        ggml_backend_tensor_set(g.w, wv.data(), 0, wv.size() * 4);
        if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(g.out, o, 0, (size_t) D * n * 4);
        return true;
    }
    // the share of the expert bytes in RAM: 1 with the resident copy, else the file mapping's mincore residency
    double resident_frac() const {
        if (rbuf) return 1.0;
        const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
        size_t in = 0, all = 0;
        for (const auto& sp : spans) {
            const uintptr_t a0 = (uintptr_t) sp.first / pg * pg, a1 = (uintptr_t) sp.first + sp.second;
            std::vector<unsigned char> v((a1 - a0 + pg - 1) / pg);
            if (mincore((void*) a0, a1 - a0, v.data()) != 0) return -1.0;
            for (unsigned char c : v) in += c & 1;
            all += v.size();
        }
        return all ? (double) in / (double) all : 0.0;
    }
    // ids/w: K entries; x: D floats -> out D floats (the weighted routed sum)
    bool run(const float* x, const int* ids, const float* w, float* o) {
        std::vector<int32_t> iv(ids, ids + KIN);
        std::vector<float> wv(w, w + KIN);
        if (K < KIN) {   // keep the K heaviest (the weights stay as routed: the draft only needs the argmax)
            std::vector<int> ord((size_t) KIN);
            for (int64_t i = 0; i < KIN; ++i) ord[(size_t) i] = (int) i;
            std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) { return w[a] > w[b]; });
            for (int64_t i = 0; i < K; ++i) { iv[(size_t) i] = ids[ord[(size_t) i]]; wv[(size_t) i] = w[ord[(size_t) i]]; }
        }
        ggml_backend_tensor_set(t_x, x, 0, (size_t) D * 4);
        ggml_backend_tensor_set(t_id, iv.data(), 0, (size_t) K * 4);
        ggml_backend_tensor_set(t_w, wv.data(), 0, (size_t) K * 4);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(out, o, 0, (size_t) D * 4);
        return true;
    }
    ~MtpExperts() {
        for (G& g : gn) {
            if (g.allo) ggml_gallocr_free(g.allo);
            if (g.ctx) ggml_free(g.ctx);
        }
        if (allo) ggml_gallocr_free(allo);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (rbuf) ggml_backend_buffer_free(rbuf);
        if (ctx) ggml_free(ctx);
        if (wctx) ggml_free(wctx);
        if (be) ggml_backend_free(be);
    }
};

}  // namespace strata::ds4
