// tools/ds4/test_hc_fused.cpp - (mab776 M13) the fused hyper-connection ops against the ops they replace, on the GPU:
//   ggml_dsv4_hc_pre_norm  vs  rms_norm(dsv4_hc_pre(x, w)) * norm_w   -> must be BIT-identical (any n, strided weights)
//   ggml_dsv4_hc_mix       vs  mul_mat(fn F16, rms_norm(x)) (F32 acc) -> float rounding only (max rel. diff reported)
// Random inputs at the model's scale (hc 4, n_embd 4096: the flattened row 16384, 24 mixes).  Exit 0 = pass.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(1234);
static void fill(std::vector<float> & v, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (float & x : v) x = d(rng);
}

struct Run {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~Run() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
};

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::printf("no CUDA device\n"); return 1; }
    const int64_t D = 4096, HC = 4, M = 24;
    bool ok = true;
    for (int64_t n : { 1, 2, 3, 4, 8, 37, 2048 }) {
        Run r;
        ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() * 2, nullptr, true };
        r.ctx = ggml_init(ip);
        ggml_tensor * x   = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, D, HC, n);
        ggml_tensor * gt  = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, 2 * HC, n);   // pre | post, as hc_gates writes it
        ggml_tensor * nw  = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, D);
        ggml_tensor * fn  = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, HC * D, M);
        ggml_tensor * pre = ggml_view_2d(r.ctx, gt, HC, n, gt->nb[1], 0);           // strided, as in the graph
        const float eps = 1e-6f;
        // the replaced ops
        ggml_tensor * a_pn = ggml_mul(r.ctx, ggml_rms_norm(r.ctx, ggml_dsv4_hc_pre(r.ctx, x, pre), eps), nw);
        ggml_tensor * flat = ggml_reshape_2d(r.ctx, x, HC * D, n);
        ggml_tensor * a_mx = ggml_mul_mat(r.ctx, fn, ggml_rms_norm(r.ctx, flat, eps));
        ggml_mul_mat_set_prec(a_mx, GGML_PREC_F32);
        // the fused ones
        ggml_tensor * b_pn = ggml_dsv4_hc_pre_norm(r.ctx, x, pre, nw, eps);
        ggml_tensor * b_mx = n <= 8 ? ggml_dsv4_hc_mix(r.ctx, fn, flat, eps) : nullptr;
        ggml_cgraph * g = ggml_new_graph(r.ctx);
        for (ggml_tensor * t : { a_pn, a_mx, b_pn }) { ggml_set_output(t); ggml_build_forward_expand(g, t); }
        if (b_mx) { ggml_set_output(b_mx); ggml_build_forward_expand(g, b_mx); }
        r.buf = ggml_backend_alloc_ctx_tensors(r.ctx, be);
        std::vector<float> hx((size_t) (D * HC * n)), hg((size_t) (2 * HC * n)), hn((size_t) D), hf((size_t) (HC * D * M));
        fill(hx, -3.0f, 3.0f);
        fill(hg, 0.0f, 1.0f);
        fill(hn, 0.5f, 1.5f);
        fill(hf, -0.05f, 0.05f);
        std::vector<ggml_fp16_t> hf16(hf.size());
        ggml_fp32_to_fp16_row(hf.data(), hf16.data(), (int64_t) hf.size());
        ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * 4);
        ggml_backend_tensor_set(gt, hg.data(), 0, hg.size() * 4);
        ggml_backend_tensor_set(nw, hn.data(), 0, hn.size() * 4);
        ggml_backend_tensor_set(fn, hf16.data(), 0, hf16.size() * 2);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, g) || ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
            std::printf("n %lld: compute failed\n", (long long) n);
            return 1;
        }
        std::vector<float> va((size_t) (D * n)), vb((size_t) (D * n));
        ggml_backend_tensor_get(a_pn, va.data(), 0, va.size() * 4);
        ggml_backend_tensor_get(b_pn, vb.data(), 0, vb.size() * 4);
        const bool same = std::memcmp(va.data(), vb.data(), va.size() * 4) == 0;
        size_t ndiff = 0;
        for (size_t i = 0; i < va.size(); ++i) ndiff += std::memcmp(&va[i], &vb[i], 4) != 0;
        std::printf("n %4lld: hc_pre_norm %s (%zu of %zu floats differ)", (long long) n, same ? "BIT-IDENTICAL" : "DIFFERENT",
                    ndiff, va.size());
        ok = ok && same;
        if (b_mx) {
            std::vector<float> ma((size_t) (M * n)), mb((size_t) (M * n));
            ggml_backend_tensor_get(a_mx, ma.data(), 0, ma.size() * 4);
            ggml_backend_tensor_get(b_mx, mb.data(), 0, mb.size() * 4);
            double worst = 0, scale = 0;
            for (size_t i = 0; i < ma.size(); ++i) scale = std::max(scale, (double) std::fabs(ma[i]));
            for (size_t i = 0; i < ma.size(); ++i) worst = std::max(worst, std::fabs((double) ma[i] - mb[i]) / scale);
            std::printf("; hc_mix max |diff| / max |mix| = %.3g (max |mix| %.3g)", worst, scale);
            ok = ok && worst < 1e-5;
        }
        std::printf("\n");
        ggml_gallocr_free(ga);
    }
    ggml_backend_free(be);
    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
