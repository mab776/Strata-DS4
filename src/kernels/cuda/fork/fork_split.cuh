// src/kernels/cuda/fork/fork_split.cuh - Split<11> (Q3_K) and Split<143> (PTQ1_0): the decode-once form of the fork
// formats' dots, so DeepSeek-V4.1's GSQ experts (gate/up 47 % PTQ1_0 + 18 % Q3_K, down 35 % + 17 %) take the
// multi-entry grouped kernels (each weight block decoded once for up to GRP_NC entries) instead of the per-entry
// native_gu_kernel / native_down_kernel, which redo the decode for every token routed to the expert.
// Fork code kept out of upstream's iq_kernels.cu (one #include line there, after its last Split): it uses that
// translation unit's Split / kSplit templates and helpers (not a standalone header).
//
// As upstream's Splits: `load` = everything that depends only on the weight block, `apply` = the activation loads and
// the dot's integer and float operations in the same order on the same values, so apply(load(...)) is BITWISE the
// dot (Fmt<T>::dot) and a column of the multi kernels equals the per-entry kernels' column.  Integer sums (dp4a chains)
// are exact in any order; every float operation keeps the original's expression and order.
// Runtime switch: STRATA_SPLIT_FORK=1 takes the multi kernels for these types (g_split_fork in the launchers);
// off (the default) = the per-entry kernels exactly as before.
// Measured 2026-10-16 (nsys, 8 s of greedy verify passes, GPU 0): bitwise the same output; Q3_K gate/up 52.3 -> 42.9 us a
// call (-18 %), Q3_K down 20.5 -> 21.8 (no gain: kForkSplitDown off), PTQ1_0 gate/up 85.8 -> 85.5 and down 57.1 -> 60.9
// - PTQ1_0's cost is its trit decode (~2.3x Q2_0's instructions per weight) and a verify pass routes ~1 token per
// expert, so decoding once saves nothing: Split<143> stays below for reference but kSplit<143> is off.
#pragma once

template<int TY> inline constexpr bool kForkSplit = false;       // gate/up (and mmvq) types STRATA_SPLIT_FORK gates
template<int TY> inline constexpr bool kForkSplitDown = false;   // ... of those, the ones whose down takes it too

// Q3_K (11): vec_dot_q3_K_q8_1 (ggml's MMVQ dot, VDR 1): one int of qs + its hmask bits -> 4 signed 3-bit words for the
// QR3_K = 4 q8_1 sub-blocks, their 6-bit scales, the fp16 block scale
template<> inline constexpr bool kSplit<11> = true;
template<> inline constexpr bool kForkSplit<11> = true;
template<> struct Split<11> {
    struct W { int vi[QR3_K]; int sc[QR3_K]; float d; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q3_K* bq3_K = (const block_q3_K*) vbq + kbx;
        const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
        W r;
        r.d = bq3_K->d;
        const int vl = get_int_b2(bq3_K->qs, iqs);
        const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;   // as the dot: 0/1 -> 4/0
        const uint8_t* scales = bq3_K->scales;
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            const int isc = scale_offset + 2 * i;
            const int isc_low = isc % (QK_K / 32);
            const int sc_shift_low = 4 * (isc / (QK_K / 32));
            const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
            const int isc_high = isc % (QK_K / 64);
            const int sc_shift_high = 2 * (isc / (QK_K / 64));
            const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
            r.sc[i] = (sc_low | sc_high) - 32;
            const int vil = (vl >> (2 * i)) & 0x03030303;
            const int vih = ((vh >> i) << 2) & 0x04040404;
            r.vi[i] = __vsubss4(vil, vih);
        }
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            const int u = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
            const float d8 = __low2float(bq8_1[bq8_offset + i].ds);
            sumf += d8 * (ggml_cuda_dp4a(r.vi[i], u, 0) * r.sc[i]);   // vec_dot_q3_K_q8_1_impl_mmvq's expression
        }
        return r.d * sumf;
    }
};

#ifndef STRATA_PTQ1_QUARTER   // the whole-block Fmt<143> (ipb 1) - the quarter dot sums a block in another order
// PTQ1_0 (143): vec_dot_ptq1_0_q8_1 - the block's 128 trits as 30 dp4a digit words (qs[0..23], 5 trits a byte) and 8
// scalar digits (qh[0..1], 4 a byte), the fp16 d; apply keeps the dot's sumi / sumu per q8_1 sub-block (integers)
// and its float sum over the 4 sub-blocks, in order
template<> struct Split<143> {   // (not enabled: kSplit<143> stays false - see the header)
    struct W { int q[30]; int qh[8]; float d; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int, const uint32_t* __restrict__ = nullptr) {
        const strata_block_ptq1_0* bq = (const strata_block_ptq1_0*) vbq + kbx;
        const uint32_t* qs32 = (const uint32_t*) bq->qs;
        W r;
        int n = 0;
#pragma unroll
        for (int w = 0; w < 6; ++w) {   // qs[0..15] (w 0..3: elements 16t + 4w) then qs[16..23] (w 4..5: 80 + 8t + 4(w-4))
            const uint32_t packed = qs32[w];
            uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
            uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
            for (int t = 0; t < 5; ++t) {
                const uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
                v_lo = w_lo & 0x00FF00FFu;
                v_hi = w_hi & 0x00FF00FFu;
                r.q[n++] = __byte_perm(w_lo, w_hi, 0x7531);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t v = bq->qh[h];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t w = v * 3;
                r.qh[h * 4 + t] = (int) (w >> 8);
                v = w & 0xFF;
            }
        }
        r.d = __half2float(bq->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[4] = {0, 0, 0, 0};
        int sumu[4] = {0, 0, 0, 0};
        int n = 0;
#pragma unroll
        for (int w = 0; w < 6; ++w) {
#pragma unroll
            for (int t = 0; t < 5; ++t) {
                const int e = w < 4 ? t * 16 + 4 * w : 80 + t * 8 + 4 * (w - 4);
                const int u = get_int_b4(bq8_1[iqs + (e >> 5)].qs, (e & 31) >> 2);
                sumi[e >> 5] = ggml_cuda_dp4a(r.q[n++], u, sumi[e >> 5]);
                sumu[e >> 5] = ggml_cuda_dp4a(0x01010101, u, sumu[e >> 5]);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = 120 + t * 2 + h;
                const int a = (int) bq8_1[iqs + (e >> 5)].qs[e & 31];
                sumi[e >> 5] += r.qh[h * 4 + t] * a;
                sumu[e >> 5] += a;
            }
        }
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) acc += __low2float(bq8_1[iqs + k].ds) * (float) (sumi[k] - sumu[k]);
        return r.d * acc;
    }
};
#endif
