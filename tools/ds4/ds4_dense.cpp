// tools/ds4/ds4_dense.cpp - see ds4_dense.hpp for the API and docs/ds4/ENGINE_DENSE.md for the state layout.
//
// The op sequence mirrors tools/ds4/ds4_ref.cpp (the Phase-3 verified reference forward) one token at a time.
// Everything the reference does over the whole sequence is done here over the *visible* keys only: the raw
// sliding window is the last n_swa tokens, the compressed source is the completed blocks, and the reference's
// visibility masks become the shape of the decode graph.  ggml-cpu's flash-attn skips a masked key outright
// (`if (mv == -INFINITY) continue;`, ops.cpp) and the visible keys keep their relative order, so the attention
// numerics are the reference's rather than an approximation of them.

#include "ds4_dense.hpp"

#include "ggml-alloc.h"
#include "ggml-impl.h"   // ggml_cgraph::uid + ggml_graph_next_uid (graph reuse, see alloc_graph)
#include "ggml-backend-impl.h"   // ggml_backend_event::context (= a cudaEvent_t on CUDA; mab776 M11b)
#include "ggml-cpu.h"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace strata;

namespace {

constexpr float NEG_INF  = -INFINITY;
constexpr float SENTINEL = -1.0e30f;   // "compressor state row not written yet" score: exp() underflows to 0

// Tokens one verify pass carries (the main token + up to 7 chained drafts; was 4 before mab776 M7c).  Every ring is
// this many - 1 slots longer than its gather window, so a later token of a pass never overwrites a row an earlier token
// of the same pass reads.
constexpr int64_t kNtMax = 8;

// Every graph here has its own gallocr and never changes after it is built, so it needs allocating once.  Stamping it
// with a uid then lets ggml-cuda skip its per-call node-property memcpy/memcmp (ggml_cuda_graph_update_required) and
// reuse the captured CUDA graph as is: ~90 graph launches per token, 165-243 nodes each, all CPU time the GPU waits
// on (the layers are serial).  The inputs are uploaded into fixed tensors, so nothing a reused graph reads moves.
// DS4_GRAPH_REUSE=0: allocate and compare every call, as before (A/B).
// With one allocator shared by every decode graph (Impl::make_allo), a graph that needs more than the buffer holds
// makes ggml-alloc reallocate it, which leaves every graph allocated earlier pointing into freed memory: the
// allocator's epoch then advances and each graph re-allocates (and ggml-cuda re-captures it) on its next use.
// reserve_graphs reserves every variant first, so in practice the buffer reaches its size before any graph runs.
bool async_on();
// be (mab776 M11, DS4_ASYNC): the decode backend - a real allocation (first use, or after the shared buffer moved) may
// grow the shared arena and FREE the old buffer while an async graph still runs on its intermediates (cudaFree only
// "may" synchronize), so the stream is drained first.  The cached path (every other call) never waits.
// mab776 M12c (DS4_F16_ACC32, default on): the decode graphs' F16-weight matmuls - the hyper-connection mixes, the
// compressors' and the indexer's projections, the MTP head's eh_proj (the loader makes F16 of our file's F32 / Q8_0
// tensors because the geometry wants F16) - ask ggml for fp32 accumulation.  ggml-cuda's default for an F16 weight in
// its vector kernel (every one-token step, and every verify pass with GGML_MMVF_F16_MAX) is half2 accumulation with
// the activations rounded to half; llama.cpp multiplies these from the GGUF's F32 tensors in F32.  The router already
// asked for F32.  Prompt chunks (their own allocator, never this function) keep their kernels.  DS4_F16_ACC32=0: the
// default precision as before.
bool f16_acc32_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_F16_ACC32"); return !e || std::atoi(e) != 0; }();
    return on;
}
void f16_acc32(ggml_cgraph * g) {
    const int nn = ggml_graph_n_nodes(g);
    for (int i = 0; i < nn; ++i) {
        ggml_tensor * t = ggml_graph_node(g, i);
        if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[0]->type == GGML_TYPE_F16) ggml_prec_set_acc(t, GGML_PREC_F32);
    }
}
bool alloc_graph(ggml_gallocr_t a, ggml_cgraph * g, ggml_backend_t be = nullptr) {
    static const bool reuse = [] { const char * e = std::getenv("DS4_GRAPH_REUSE"); return !e || std::atoi(e) != 0; }();
    static std::map<const void *, uint64_t> epoch;                  // per allocator
    static std::map<const ggml_cgraph *, uint64_t> graph_epoch;     // the epoch each graph was allocated in
    uint64_t & ep = epoch[a];
    auto it = graph_epoch.find(g);
    if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == ep) return true;
    if (it != graph_epoch.end() && it->second != ep) {
        // allocated before the shared buffer moved: ggml-alloc never re-binds a tensor that has data, so unbind this
        // graph's compute-buffer tensors (and views of them) first.  Persistent state, inputs and weights live in
        // other buffers and keep theirs.
        auto computed = [](const ggml_tensor * t) {
            return t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        };
        auto unbind = [&](ggml_tensor * t) {
            if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
        };
        const int nn = ggml_graph_n_nodes(g);
        for (int i = 0; i < nn; ++i) {
            ggml_tensor * t = ggml_graph_node(g, i);
            for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
            unbind(t);
        }
        for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
    }
    if (be && async_on()) ggml_backend_synchronize(be);
    if (f16_acc32_on()) f16_acc32(g);   // mab776 M12c: op params only, before the graph's first (re)capture
    const size_t before = ggml_gallocr_get_buffer_size(a, 0);
    if (!ggml_gallocr_alloc_graph(a, g)) return false;
    if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++ep;   // the buffer moved
    graph_epoch[g] = ep;
    if (reuse) g->uid = ggml_graph_next_uid();
    return true;
}

// mab776 M11 (DS4_ASYNC, default on): decode graphs and their hand-offs ride the backend's stream with no host wait per
// graph.  ggml_backend_graph_compute is compute_async + a synchronize, and ggml-cuda's plain tensor_get/set copy on
// cudaStreamPerThread + a synchronize: a layer waited 6 times on the dense half alone (predict, its readback,
// attention, its readback, the routed-sum upload, finish), the GPU idle while the host launched the next graph (nsys
// @127k: GPU 0 busy 71 %).  Async: one synchronize where the host needs a result (the predict and router readbacks);
// finish and the next predict queue behind each other.  Stream order keeps every dependency (one stream, and the
// graphs that share the decode arena run one after the other on it).  DS4_ASYNC=0: the waits as before.
bool async_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_ASYNC"); return !e || std::atoi(e) != 0; }();
    return on;
}
bool compute_dec(ggml_backend_t be, ggml_cgraph * g) {
    return (async_on() ? ggml_backend_graph_compute_async(be, g) : ggml_backend_graph_compute(be, g)) == GGML_STATUS_SUCCESS;
}
void get_dec(ggml_backend_t be, const ggml_tensor * t, void * dst, size_t n) {   // waits for the stream
    if (async_on()) { ggml_backend_tensor_get_async(be, t, dst, 0, n); ggml_backend_synchronize(be); }
    else ggml_backend_tensor_get(t, dst, 0, n);
}
// mab776 M11b (DS4_OVL, default on; needs DS4_ASYNC): the two host waits a verify layer kept that the card had to idle
// through - (1) the predict readback: the host picked the prefetch, then launched the attention graph; (2) the expert
// tier's stream sync before finish was launched.  Now predict's readback lands in pinned memory behind an event and
// the attention is queued before the host waits for it (predict_launch / predict_collect + attn_router_n's `queued`
// hook), and the tier records an event after its mix that finish waits on ON THE CARD (decode_sum_event).  Same
// kernels in the same stream order: exact.  DS4_OVL=0: the waits as before.
bool ovl_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_OVL"); return !e || std::atoi(e) != 0; }();
    return on && async_on();
}

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ------------------------------------------------------------------ weights (zero-copy on CPU, uploaded off-CPU)

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;

    ~WStore() {
        for (ggml_backend_buffer_t b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "[ds4_dense] missing weight tensor %s\n", n.c_str());
            std::abort();
        }
        return it->second;
    }
};

bool is_routed_expert(const std::string & n) {
    return n.size() > 12 && n.compare(n.size() - 12, 12, "_exps.weight") == 0;
}

// Off-CPU, `token_embd` stays on the host (begin_token looks the row up there: 1 GiB of VRAM the expert cache gets
// instead) and, with `skip_experts`, the routed experts are not loaded at all.
// --dense-requant: the big Q8_0 dense matrices, by name
bool requant_target(const std::string & n) {
    for (const char * s : { "attn_q_b.weight", "attn_output_a.weight", "attn_output_b.weight", "ffn_gate_shexp.weight",
                            "ffn_up_shexp.weight", "ffn_down_shexp.weight" })
        if (n.size() > std::strlen(s) && n.compare(n.size() - std::strlen(s), std::string::npos, s) == 0 &&
            n.rfind("blk.", 0) == 0 && n.find("indexer") == std::string::npos)
            return true;
    if (n.size() > 21 && n.compare(n.size() - 21, 21, ".nextn.eh_proj.weight") == 0) return true;   // MTP block (BF16)
    return n == "output.weight";
}

// `prefix` non-empty: only the tensors whose name starts with it (the MTP file: its own block - its token_embd and
// output duplicate the trunk's), plus those named in `alias` (file name -> store name: the MTP file's own hc_head
// weights, which differ from the trunk's).  A second call appends to the same store.
// --f16-q8: the F16 attention-side matrices that are big and not precision-critical (compressors, indexer query and
// projection: ~0.9 GiB F16, ~2 ms/token of F16 matvecs).  The router (ffn_gate_inp) and the hyper-connection mixers
// stay F16: they pick experts and drive the Sinkhorn mix.
bool f16_q8_target(const std::string & n) {
    for (const char * s : { "attn_compressor_kv.weight", "attn_compressor_gate.weight", "indexer_compressor_kv.weight",
                            "indexer_compressor_gate.weight", "indexer.attn_q_b.weight", "indexer.proj.weight" })
        if (n.size() > std::strlen(s) && n.compare(n.size() - std::strlen(s), std::string::npos, s) == 0) return true;
    return false;
}

bool load_weights(const GgufModel & model, ggml_backend_t backend, WStore & w, bool skip_experts, int requant,
                  std::string & err, const std::string & prefix = "",
                  const std::map<std::string, std::string> & alias = {}, bool f16_q8 = false,
                  const std::map<std::string, uint32_t> * want_type = nullptr) {
    if (!w.ctx) {
        ggml_init_params ip = { /*mem_size*/ 256ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        w.ctx = ggml_init(ip);
        if (!w.ctx) { err = "ggml_init(weights) failed"; return false; }
    }
    ggml_set_no_alloc(w.ctx, true);
    auto wanted = [&](const std::string & n) { return prefix.empty() || n.rfind(prefix, 0) == 0 || alias.count(n); };
    auto sname = [&](const std::string & n) { auto it = alias.find(n); return it == alias.end() ? n : it->second; };

    const bool cpu = ggml_backend_get_default_buffer_type(backend) == ggml_backend_cpu_buffer_type();

    if (cpu) {
        // bind every tensor straight onto the GGUF mmap (ds4_ref's loader)
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & tensors = sh.tensors();
            if (tensors.empty()) continue;
            const uint8_t * any = sh.tensor_data(tensors[0]);
            uint8_t * base = const_cast<uint8_t *>(any) - sh.data_start() - tensors[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            if (!buf) { err = "cannot wrap " + sh.path(); return false; }
            w.bufs.push_back(buf);
            for (const auto & ti : tensors) {
                if (!wanted(ti.name)) continue;
                if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
                int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
                for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
                ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
                if (!t) { err = "cannot create " + ti.name; return false; }
                ggml_set_name(t, sname(ti.name).c_str());
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[sname(ti.name)] = t;
            }
        }
        return true;
    }

    // non-CPU: create the tensor set, allocate one backend buffer for it, upload the payloads
    auto skip = [&](const std::string & n) {
        return !wanted(n) || n == "token_embd.weight" || (skip_experts && is_routed_expert(n));
    };
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip(ti.name)) continue;
            if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_type ty = (ggml_type) ti.type;
            // mab776: a file with another quant mix (unsloth UD-*) - dense tensors take the type this engine's graphs
            // and fused kernels were written for (converted below by the requantize path); experts keep theirs
            if (want_type && !is_routed_expert(ti.name)) {
                auto wt = want_type->find(ti.name);
                if (wt != want_type->end() && (int) wt->second != (int) ty) ty = (ggml_type) wt->second;
            }
            if (requant >= 0 && (ty == GGML_TYPE_Q8_0 || ty == GGML_TYPE_BF16) && requant_target(ti.name) &&
                ne[0] % ggml_blck_size((ggml_type) requant) == 0)
                ty = (ggml_type) requant;
            // every other BF16 matrix (the MTP file's attn_kv, attn_q_a) -> Q8_0, like the trunk's: a BF16 mul_mat
            // goes through cuBLAS on CUDA, whose first use loads its kernels into VRAM after --slots auto has sized the
            // expert cache (cudaGraphInstantiate OOM with --mtp, 2026-10-07)
            else if (ty == GGML_TYPE_BF16 && ti.shape.size() >= 2 && ne[0] % 32 == 0)
                ty = GGML_TYPE_Q8_0;
            else if (f16_q8 && ty == GGML_TYPE_F16 && ti.shape.size() >= 2 && ne[0] % 32 == 0 && f16_q8_target(ti.name))
                ty = GGML_TYPE_Q8_0;
            ggml_tensor * t = ggml_new_tensor(w.ctx, ty, (int) ti.shape.size(), ne);
            if (!t) { err = "cannot create " + ti.name; return false; }
            ggml_set_name(t, sname(ti.name).c_str());
            w.t[sname(ti.name)] = t;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buf) { err = "cannot allocate the weight buffer on the backend"; return false; }
    w.bufs.push_back(buf);
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip(ti.name)) continue;
            ggml_tensor * t = w.t[sname(ti.name)];
            if ((int) t->type == (int) ti.type) {
                ggml_backend_tensor_set(t, model.shard(s).tensor_data(ti), 0, ggml_nbytes(t));
                continue;
            }
            // requantize: rows dequantized with the file type's to_float, re-quantized to t->type, in parallel
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            const uint8_t * src = model.shard(s).tensor_data(ti);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        if (ti.type == GGML_TYPE_F32)   // mab776: F32 has no to_float (unsloth's F32 hc mixers)
                            std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * sizeof(float));
                        else
                            ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
        }
    }
    return true;
}

}  // namespace

// ================================================================== Impl

struct Ds4Dense::Impl {
    std::string  path;
    Ds4Geometry  g;
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;
    int  n_threads = 8;

    WStore          w;
    std::unique_ptr<GgufModel> mtp_model; // the MTP (nextn) head's own GGUF, when loaded
    int64_t         il_mtp = -1;          // its block index (= n_trunk), or -1
    int64_t         n_trunk = 0;          // trunk layers (ly holds n_trunk + 1 entries with the MTP block)
    ggml_context *  sctx = nullptr;     // persistent decode state
    // Per-step inputs and per-layer router outputs live at fixed offsets in two small buffers so a token costs ONE
    // upload of every layer's inputs and one readback per layer (was ~11 tensor_set + 3 tensor_get per layer, each a
    // stream sync on CUDA: 1283 copies + 905 syncs per token measured by nsys).
    ggml_context *  ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr, obuf = nullptr;
    ggml_tensor *   i_span = nullptr;   // I8 over the whole input buffer
    std::vector<uint8_t> in_host;       // its host mirror
    uint8_t *       in_base = nullptr;  // device base of ibuf (offsets are t->data - in_base)
    int             in_pos = -1;        // the first position whose inputs are uploaded
    int             in_n   = 0;         // ... and the pass's token count
    int             cur_tid = -1;       // the token id in i_tid
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_context *  hctx = nullptr;     // --comp-host: the compressed K caches' own context ...
    ggml_backend_buffer_t hbuf = nullptr;   // ... in a pinned host buffer, read zero-copy by the device
    bool            comp_rot = false;   // comp rows Hadamard-rotated around their quantization
    ggml_context *  gctx = nullptr;     // graph node structs (data comes from the gallocrs)
    std::string     err;

    // ---- prompt chunks (prefill_*).  While `pf` is set the builders build a chunk graph: the rings are read as
    // [ring | the chunk's own rows] and written back with the chunk's last rows, the gate taps are skipped and the
    // graph is not registered.  pf_swap() points the decode hand-off / input tensors at the chunk-sized ones in P for
    // the duration of one build (the graph keeps the pointers it was built with).
    bool    pf = false;
    int64_t pf_cap = 0;                  // Ds4DenseConfig::prefill_chunk
    int     pf_pos0 = 0, pf_n = 0;       // the chunk begun by prefill_begin
    ggml_tensor * dbg_hcab_d2 = nullptr, * dbg_hcab_r2 = nullptr;   // DS4_CHECK_HCAB=1: [1, n] |hcab - blocked|^2, |blocked|^2
    int dbg_hcab_il = -1;
    int64_t pf_row0 = 0;                 // build_head in pf mode: first row of the chunk
    ggml_context * pf_ctx = nullptr;     // the chunk tensors (P)
    ggml_backend_buffer_t pf_buf = nullptr;
    ggml_backend_buffer_t pf_obuf = nullptr;   // the chunk's router output blocks (o_f / o_i alias one buffer)
    ggml_gallocr_t pf_allo = nullptr;    // one allocator for every chunk graph: each is built, run once and dropped
    std::vector<uint8_t> pf_meta;        // the chunk graph's node structs (one reused buffer)
    ggml_tensor * pf_logits = nullptr;   // the last chunk head graph's output
    struct PfRatio {                     // per compress ratio (4 = CSA, 128 = HCA): the layers of a ratio share inputs
        ggml_tensor * i_slot_comp = nullptr, * i_slot_state = nullptr, * i_comp_pos = nullptr, * i_state_pos = nullptr,
                    * i_idx_state = nullptr, * vis_full = nullptr, * i_nvis = nullptr;
        int64_t n_emit = 0;              // DS4_PF_EMIT: the chunk's compressed rows that land in a slot (see prefill_begin)
    };
    struct PfSet {
        ggml_tensor * x_state = nullptr, * routed_sum = nullptr, * i_tid = nullptr, * i_emb = nullptr, * i_pos = nullptr;
        ggml_tensor * hap = nullptr, * post_f = nullptr, * comb_f = nullptr, * shexp = nullptr, * o_f = nullptr,
                    * o_i = nullptr;
        ggml_tensor * i_slot_raw = nullptr, * i_idx_raw = nullptr, * i_mask_raw = nullptr;
        PfRatio r4, r128;
    } P;
    // host side of a chunk: router output blocks [fn | ids | wts] x n, the ffn_norm rows the expert tier reads and the
    // routed sums it writes - pinned when the backend has a host buffer type (CUDA): pageable copies are staged by
    // the driver in pieces that queue behind the tier's expert DMAs on the one host-to-device copy engine
    ggml_backend_buffer_t pf_hbuf = nullptr;
    std::vector<uint8_t> pf_hvec;        // the same, pageable (CPU backend)
    // mab776 M11 (DS4_ASYNC): the decode finish's routed sums, pinned (CUDA), so their upload is queued on the stream
    // with no host wait - a pageable source has the driver stage it synchronously
    ggml_backend_buffer_t dec_hbuf = nullptr;
    float * dec_routed = nullptr;
    // mab776 M11b (DS4_OVL): the split predict's pinned readback and its event, and the event the expert tier records
    // after mixing into routed_sum (finish_layer_n waits on it on the card) - set up once, on CUDA only
    static constexpr size_t kPredHost = 64 * 1024;
    ggml_backend_buffer_t pred_hbuf = nullptr;
    void * pred_host = nullptr;
    ggml_backend_event_t ev_pred = nullptr, ev_sum = nullptr;
    ggml_backend_event_t ev_attn = nullptr;   // mab776 M12b: recorded after a queued attention graph (DS4_DEV_ROUTE)
    int  pred_il = -1;          // the layer whose predict readback is in flight
    bool ovl_tried = false;
    bool ovl_setup() {
        if (ovl_tried) return ev_pred != nullptr;
        ovl_tried = true;
        if (!ovl_on() || !backend || ggml_backend_is_cpu(backend)) return false;
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!dev) return false;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) return false;   // the tier records raw CUDA events
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev);
        if (!props.caps.events || !hbt) return false;
        pred_hbuf = ggml_backend_buft_alloc_buffer(hbt, kPredHost);
        ev_pred = ggml_backend_event_new(dev);
        ev_sum = ggml_backend_event_new(dev);
        ev_attn = ggml_backend_event_new(dev);
        if (!pred_hbuf || !ev_pred || !ev_sum || !ev_attn) { ovl_free(); return false; }
        pred_host = ggml_backend_buffer_get_base(pred_hbuf);
        return true;
    }
    void ovl_free() {
        if (pred_hbuf) ggml_backend_buffer_free(pred_hbuf);
        if (ev_pred) ggml_backend_event_free(ev_pred);
        if (ev_sum) ggml_backend_event_free(ev_sum);
        if (ev_attn) ggml_backend_event_free(ev_attn);
        pred_hbuf = nullptr; pred_host = nullptr; ev_pred = nullptr; ev_sum = nullptr; ev_attn = nullptr;
    }
    uint8_t * pf_out_host = nullptr;
    float *   pf_fn = nullptr;
    float *   pf_routed = nullptr;
    void pf_free() {
        if (pf_allo) ggml_gallocr_free(pf_allo);
        if (pf_buf) ggml_backend_buffer_free(pf_buf);
        if (pf_obuf) ggml_backend_buffer_free(pf_obuf);
        if (pf_ctx) ggml_free(pf_ctx);
        pf_allo = nullptr; pf_buf = nullptr; pf_obuf = nullptr; pf_ctx = nullptr; P = PfSet();
        std::vector<uint8_t>().swap(pf_meta);
        if (pf_hbuf) ggml_backend_buffer_free(pf_hbuf);
        pf_hbuf = nullptr;
        std::vector<uint8_t>().swap(pf_hvec);
        pf_out_host = nullptr; pf_fn = nullptr; pf_routed = nullptr;
    }

    // One graph allocator per graph, never a shared one.  ggml-alloc binds a tensor's data pointer once
    // and skips re-binding it on later allocations, while a reserve triggered by a *different* graph
    // frees and reallocates the chunks it had bound - a shared allocator therefore leaves the tensors of
    // every other graph dangling into freed memory as soon as one graph grows.  Per-graph allocators also
    // give each decode graph the stable, dedicated buffer ggml-cuda needs to capture it as a CUDA graph.
    bool ensure_n(int64_t n);           // init/head/finish graphs for n-token passes (defined after the builders)

    // DS4_SHARED_ALLO (default on): every decode graph on ONE allocator - they run one at a time and everything that
    // outlives a graph lives in sbuf / ibuf / obuf, so their intermediates may share memory (was ~850 MiB of VRAM for
    // ~500 per-graph arenas at p600).  Needs alloc_graph's epoch (a growth reallocates the shared buffer).
    // DS4_SHARED_ALLO=0: one allocator per graph, as before.
    mutable ggml_gallocr_t shared_allo = nullptr;
    static bool shared_on() {
        static const bool on = [] { const char * e = std::getenv("DS4_SHARED_ALLO"); return !e || std::atoi(e) != 0; }();
        return on;
    }
    ggml_gallocr_t make_allo() const {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        if (!shared_on()) return ggml_gallocr_new(buft);
        if (!shared_allo) shared_allo = ggml_gallocr_new(buft);
        return shared_allo;
    }
    void free_allo(ggml_gallocr_t a) { if (a && a != shared_allo) ggml_gallocr_free(a); }

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, NHPG = 0, RQ = 0, SWA = 0;
    int64_t RAW = 0;   // raw window ring storage = SWA + kNtMax - 1
    int64_t IDXH = 0, IDXK = 0, IDXTOPK = 0;
    int64_t NEXP = 0, NUSED = 0, NFF = 0, NSHEXP = 0;

    // A pass carries n <= kNtMax tokens (one per column); every per-token tensor below has a kNtMax axis and the
    // graphs for n tokens view its first n columns, so the n = 1 graphs are the single-token graphs exactly.
    ggml_tensor * x_state    = nullptr;   // [D, hc, kNtMax]  layer input / output state
    ggml_tensor * routed_sum = nullptr;   // [D, kNtMax]      routed-expert sums fed to finish_layer
    ggml_tensor * i_tid      = nullptr;   // I32[kNtMax]
    ggml_tensor * i_emb      = nullptr;   // F32[D, kNtMax]   embd[tid], looked up on the host (begin_tokens)
    const uint8_t * embd_data = nullptr;  // token_embd on the GGUF mmap
    int           embd_type  = -1;
    size_t        embd_row   = 0;         // bytes per row
    std::vector<float> host_emb;
    ggml_tensor * i_pos      = nullptr;   // I32[kNtMax]
    ggml_tensor * rot        = nullptr;   // [idx_k, idx_k] F32 Walsh-Hadamard (lightning indexer)
    ggml_tensor * neg_ramp   = nullptr;   // F32 [max comp_max]: -i, the DS4_VIS_DEV mask's constant half
    std::vector<float> host_logits;       // [V, kNtMax]
    int64_t       o_blk = 0;              // bytes per token in a layer's router output block [fn | ids | wts]
    int64_t       o_ids_off = 0, o_wts_off = 0;

    struct Var {
        int64_t cap = 0;
        int64_t n = 1;                    // tokens per pass
        // false: the pass completes no compressed block (one token, (pos+1) % ratio != 0) - the compressor and
        // indexer ring states are written, but the pooled row (invisible until its block completes) is not built
        bool emit = true;
        ggml_tensor * vis = nullptr;      // F32 [cap, n] 0 visible / -inf hidden, per query
        ggml_tensor * dbg_isc = nullptr;  // F32 [cap, n] indexer scores + visibility (DS4_DBG_INDEXER=1)
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t allo = nullptr;
    };
    struct Layer {
        int64_t ratio = 0;
        bool    csa = false;
        int64_t ring = 0, sdim = 0, comp_max = 0;
        int64_t ring_sz = 0;              // compressor state ring storage = ring + kNtMax - 1

        ggml_tensor * raw = nullptr;      // [DH, RAW]              raw sliding-window K (=V, post-RoPE)
        // compressed caches have one spare row at comp_max: when two tokens of a pass fall in the same block, the
        // earlier one writes its (partial, never visible) row there, so set_rows never sees a duplicate index
        ggml_tensor * comp = nullptr;     // [DH, comp_max + 1]     compressed K (post-RoPE)

        ggml_tensor * st_kv = nullptr;    // [sdim, ring]           compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [sdim, ring]           compressor state (scores, SENTINEL unwritten)
        ggml_tensor * icomp = nullptr;    // [IDXK, comp_max + 1]   lightning-indexer compressed K (post-WHT)
        ggml_tensor * ist_kv = nullptr;   // [2*IDXK, ring_sz]      indexer compressor state (values)
        ggml_tensor * ist_sc = nullptr;   // [2*IDXK, ring_sz]      indexer compressor state (scores)

        ggml_tensor * hap = nullptr;      // [D, HC, kNtMax]        attn hc_post result (finish residual)
        ggml_tensor * post_f = nullptr;   // [HC, kNtMax]
        ggml_tensor * comb_f = nullptr;   // [HC, HC, kNtMax]
        ggml_tensor * shexp = nullptr;    // [D, kNtMax]            shared expert output
        ggml_tensor * fn = nullptr;       // [D]                    ffn_norm activation of token 0 (device)
        ggml_tensor * t_attn_raw = nullptr;  // [NH*DH, kNtMax]     flash-attn output, pre output-LoRA
        ggml_tensor * t_attn = nullptr;      // [D, kNtMax]         attention output, post output-LoRA
        ggml_tensor * t_llast = nullptr;     // [D, HC, kNtMax]     l_last

        // per-step inputs, laid out for the pass's n (the host fills the first entries)
        ggml_tensor * i_slot_raw = nullptr;   // I32[kNtMax]               raw ring slot per token
        ggml_tensor * i_idx_raw = nullptr;    // I32[SWA + kNtMax - 1]     union window, oldest -> newest
        ggml_tensor * i_mask_raw = nullptr;   // F16[(SWA + kNtMax - 1) * kNtMax]  [n_kv, n] per-query window mask
        ggml_tensor * i_slot_comp = nullptr;  // I32[kNtMax]
        ggml_tensor * i_slot_state = nullptr; // I32[kNtMax]
        ggml_tensor * i_comp_pos = nullptr;   // I32[kNtMax]
        ggml_tensor * i_state_pos = nullptr;  // I32[kNtMax]
        ggml_tensor * i_idx_state = nullptr;  // I32[ring * kNtMax]        [ring, n] per-token state gather

        std::vector<Var> vars;
        ggml_cgraph * gf_predict = nullptr;
        ggml_tensor * predict_out = nullptr;
        ggml_gallocr_t allo_predict = nullptr;
        ggml_cgraph * gf_finish[kNtMax + 1] = {};      // by n
        ggml_gallocr_t allo_finish[kNtMax + 1] = {};
        std::vector<float> host_fn, host_attn_raw, host_attn, host_llast;   // [.., kNtMax]
        int  tap_n = 0;                   // the n of the pass the taps / host_fn hold
        bool have_attn = false, have_finish = false;
        ggml_tensor * vis_full = nullptr; // F32 [comp_max * kNtMax]  compressed-row visibility (graphs view [cap, n])
        ggml_tensor * i_nvis = nullptr;   // F32[kNtMax]  complete compressed blocks per query (DS4_VIS_DEV, s22)
        ggml_tensor * i_idx_all = nullptr; // I32[comp_max + 1]  0..comp_max: the masked path gathers with these when
                                           // `comp` is quantized (--comp-q8); filled once, never uploaded per token
        ggml_tensor * rbias = nullptr;    // F32 [NEXP]       cache-aware routing: added to the SELECTION score only
        // router outputs: per token a block of o_blk bytes [fn | ids | wts], kNtMax blocks, one readback per layer
        ggml_tensor * o_ids = nullptr;    // I32 [NUSED]      token 0's router ids
        ggml_tensor * o_wts = nullptr;    // F32 [NUSED]      token 0's router weights
        ggml_tensor * o_f = nullptr;      // F32 alias of the whole layer block (graphs view fn / wts columns)
        ggml_tensor * o_i = nullptr;      // I32 alias of the same bytes (graphs view the ids columns)
        ggml_tensor * o_span = nullptr;   // I8 over the whole layer block
        std::vector<uint8_t> host_out;
    };
    std::vector<Layer> ly;

    ggml_cgraph * gf_init[kNtMax + 1] = {};       // by n
    ggml_cgraph * gf_mtp_in[kNtMax + 1] = {};     // MTP input: eh_proj(enorm(embd) | hnorm(trunk state)), by n
    ggml_cgraph * gf_mtp_head[kNtMax + 1] = {};   // MTP head: hc_head -> shared_head_norm -> output, by n
    ggml_tensor * mtp_logits_t[kNtMax + 1] = {};
    ggml_gallocr_t allo_mtp_in[kNtMax + 1] = {};
    ggml_gallocr_t allo_mtp_head[kNtMax + 1] = {};
    // mab776 M7c, chained drafts: the MTP input for one token whose h is x_state column c (the MTP block's own output
    // one position earlier), and the MTP head on column c reduced to its argmax on the device - both by c
    ggml_cgraph * gf_mtp_chain[kNtMax] = {};
    ggml_gallocr_t allo_mtp_chain[kNtMax] = {};
    ggml_cgraph * gf_mtp_arg[kNtMax] = {};
    ggml_tensor * mtp_arg_t[kNtMax] = {};         // I32 [1]
    ggml_gallocr_t allo_mtp_arg[kNtMax] = {};
    ggml_cgraph * gf_head[kNtMax + 1] = {};
    ggml_tensor * logits_t[kNtMax + 1] = {};      // [V, n] head graph outputs
    ggml_gallocr_t allo_init[kNtMax + 1] = {};
    ggml_gallocr_t allo_head[kNtMax + 1] = {};

    // ---- ring snapshots (mab776 M5, Ds4Dense::snapshot): per slot, a copy of every tensor rings() lists
    ggml_context *        snapctx = nullptr;
    ggml_backend_buffer_t snapbuf = nullptr;
    std::vector<std::vector<ggml_tensor *>> snaps;   // [slot][i] = the copy of rings()[i]
    std::vector<int>      snap_pos;                  // [slot]: the positions it holds, -1 = empty
    size_t                snap_bytes = 0;            // one slot
    // the position-indexed rings, in a fixed order (every layer, the MTP block included): raw, st_kv, st_sc, ist_kv,
    // ist_sc where the layer has them.  Everything else in the state is either rewritten by every pass (x_state, the
    // hand-offs) or indexed by position and read only once complete (comp / icomp rows).
    std::vector<ggml_tensor *> rings() const {
        std::vector<ggml_tensor *> r;
        for (const Layer & L : ly)
            for (ggml_tensor * t : { L.raw, L.st_kv, L.st_sc, L.ist_kv, L.ist_sc })
                if (t) r.push_back(t);
        return r;
    }
    bool snap_alloc(int slots);   // init(), after the state; defined with Ds4Dense::snapshot

    ~Impl() {
        if (own_backend && backend) ggml_backend_synchronize(backend);   // mab776 M11: no async upload still reading
        // dec_hbuf (an outside owner frees - and so drains - its backend first: ds4_generate does, before this runs)
        if (dec_hbuf) ggml_backend_buffer_free(dec_hbuf);
        ovl_free();
        if (snapbuf) ggml_backend_buffer_free(snapbuf);
        if (snapctx) ggml_free(snapctx);
        pf_free();
        for (Var & v : all_vars()) free_allo(v.allo);
        for (Layer & L : ly) {
            free_allo(L.allo_predict);
            for (ggml_gallocr_t a : L.allo_finish) free_allo(a);
        }
        for (ggml_gallocr_t a : allo_init) free_allo(a);
        for (ggml_gallocr_t a : allo_head) free_allo(a);
        for (ggml_gallocr_t a : allo_mtp_in) free_allo(a);
        for (ggml_gallocr_t a : allo_mtp_head) free_allo(a);
        for (ggml_gallocr_t a : allo_mtp_chain) free_allo(a);
        for (ggml_gallocr_t a : allo_mtp_arg) free_allo(a);
        if (shared_allo) ggml_gallocr_free(shared_allo);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (hbuf) ggml_backend_buffer_free(hbuf);
        if (hctx) ggml_free(hctx);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (obuf) ggml_backend_buffer_free(obuf);
        if (ictx) ggml_free(ictx);
        if (sctx) ggml_free(sctx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }

    // every attn graph variant across every layer, for the destructor above
    std::vector<Var> all_vars() {
        std::vector<Var> r;
        for (Layer & L : ly) for (Var & v : L.vars) r.push_back(v);
        return r;
    }
};

// ================================================================== graph builder helpers (ds4_ref's Graph)

namespace {

struct B {
    Ds4Dense::Impl * im = nullptr;
    ggml_context *   c  = nullptr;
    const Ds4Geometry * g = nullptr;

    ggml_tensor * W(const std::string & n) const { return im->w.get(n); }
    ggml_tensor * BL(int il, const std::string & s) const {
        return im->w.get("blk." + std::to_string(il) + "." + s);
    }
    ggml_tensor * fill_f32(std::initializer_list<int64_t> ne, float v) const {
        ggml_tensor * tmpl = nullptr;
        auto it = ne.begin();
        switch (ne.size()) {
            case 1: tmpl = ggml_new_tensor_1d(c, GGML_TYPE_F32, it[0]); break;
            case 2: tmpl = ggml_new_tensor_2d(c, GGML_TYPE_F32, it[0], it[1]); break;
            case 3: tmpl = ggml_new_tensor_3d(c, GGML_TYPE_F32, it[0], it[1], it[2]); break;
            default: std::abort();
        }
        return ggml_fill(c, tmpl, v);
    }
    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * v2(ggml_tensor * a, int64_t n0, int64_t n1, size_t nb1, size_t off) const {
        return ggml_view_2d(c, a, n0, n1, nb1, off);
    }

    // hc affine: x*scale + base  (deepseek4.cpp:279-285)
    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) const {
        return ggml_add(c, ggml_mul(c, x, scale), base);
    }
    // deepseek4.cpp:314-349
    ggml_tensor * sinkhorn(ggml_tensor * comb) const {
        comb = ggml_soft_max(c, comb);
        ggml_tensor * eps = fill_f32({ 1 }, (float) g->hc_eps);
        comb = ggml_add(c, comb, eps);
        auto norm_cols = [&]() {
            ggml_tensor * cd = ggml_cont(c, ggml_permute(c, comb, 1, 0, 2, 3));
            ggml_tensor * cs = ggml_add(c, ggml_sum_rows(c, cd), eps);
            cs = ggml_permute(c, cs, 1, 0, 2, 3);
            comb = ggml_div(c, comb, cs);
        };
        norm_cols();
        for (int64_t i = 1; i < g->hc_sinkhorn_iters; ++i) {
            ggml_tensor * rs = ggml_add(c, ggml_sum_rows(c, comb), eps);
            comb = ggml_div(c, comb, rs);
            norm_cols();
        }
        return comb;
    }
    // deepseek4.cpp:287-312
    // The fused hyper-connection ops (ggml_dsv4_hc_comb/pre/post, llama.cpp #25585): one kernel each instead of the
    // Sinkhorn loop's ~140 tiny ops per call - on CUDA that loop alone was ~12k kernels/token.  DS4_HC_UNFUSED=1 keeps
    // the op-by-op form (the original deepseek4.cpp graph) for A/B.
    static bool unfused() { static const bool u = std::getenv("DS4_HC_UNFUSED") != nullptr; return u; }
    ggml_tensor * hc_mean(ggml_tensor * x, ggml_tensor * pre) const {
        if (!unfused()) return ggml_dsv4_hc_pre(c, x, pre);
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * acc = nullptr;
        for (int64_t h = 0; h < hc; ++h) {
            ggml_tensor * xh = v2(x, D, nt, x->nb[2], (size_t) h * x->nb[1]);
            ggml_tensor * wh = v2(pre, 1, nt, pre->nb[1], (size_t) h * pre->nb[0]);
            ggml_tensor * cur = ggml_mul(c, xh, wh);
            acc = acc ? ggml_add(c, acc, cur) : cur;
        }
        return acc;
    }
    // deepseek4.cpp:351-407
    ggml_tensor * hc_pre(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                         ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * flat = ggml_reshape_2d(c, x, hc * D, nt);
        ggml_tensor * flat_norm = ggml_rms_norm(c, flat, (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat_norm);

        ggml_tensor * scale_pre  = v1(sc, 1, 0);
        ggml_tensor * scale_post = v1(sc, 1, 1 * sizeof(float));
        ggml_tensor * base_pre   = v1(bs, hc, 0);
        ggml_tensor * base_post  = v1(bs, hc, (size_t) hc * 4);

        ggml_tensor * pre = nullptr;
        // mab776 M11 (DS4_HC_GATES, default on, GPU): both gates in ONE launch (they were 8 tiny kernels - mul, add,
        // sigmoid, scale each - three hc_pre a decode layer: ~20 launches for 2 x 4 floats a token).  The op rounds
        // exactly as these ops do (no fused multiply-add): same bits.  The CPU keeps the ops (its gate stays as is).
        static const bool gates_on = [] { const char * e = std::getenv("DS4_HC_GATES"); return !e || std::atoi(e) != 0; }();
        if (!unfused() && gates_on && !ggml_backend_is_cpu(im->backend)) {
            ggml_tensor * gt = ggml_dsv4_hc_gates(c, mixes, sc, bs, (float) g->hc_eps);   // [2*hc, nt]: pre | post
            pre = v2(gt, hc, nt, gt->nb[1], 0);
            *post = v2(gt, hc, nt, gt->nb[1], (size_t) hc * gt->nb[0]);
        } else {
            pre = v2(mixes, hc, nt, mixes->nb[1], 0);
            pre = ggml_sigmoid(c, hc_affine(pre, scale_pre, base_pre));
            pre = ggml_scale_bias(c, pre, 1.0f, (float) g->hc_eps);

            *post = v2(mixes, hc, nt, mixes->nb[1], (size_t) hc * mixes->nb[0]);
            *post = ggml_sigmoid(c, hc_affine(*post, scale_post, base_post));
            *post = ggml_scale(c, *post, 2.0f);
        }

        ggml_tensor * scale_comb = v1(sc, 1, 2 * sizeof(float));
        ggml_tensor * base_comb  = v1(bs, hc * hc, (size_t) 2 * hc * 4);
        if (!unfused()) {
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_sinkhorn_iters);
        } else {
            *comb = v2(mixes, hc * hc, nt, mixes->nb[1], (size_t) 2 * hc * mixes->nb[0]);
            *comb = hc_affine(*comb, scale_comb, base_comb);
            *comb = ggml_reshape_3d(c, *comb, hc, hc, nt);
            *comb = sinkhorn(*comb);
        }
        return hc_mean(x, pre);
    }
    // deepseek4.cpp:409-444
    ggml_tensor * hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb) const {
        if (!unfused()) return ggml_dsv4_hc_post(c, x, residual, post, comb);
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[1];
        ggml_tensor * out = nullptr;
        for (int64_t dst = 0; dst < hc; ++dst) {
            ggml_tensor * post_dst = v2(post, 1, nt, post->nb[1], (size_t) dst * post->nb[0]);
            ggml_tensor * cur = ggml_mul(c, x, post_dst);
            for (int64_t src = 0; src < hc; ++src) {
                ggml_tensor * res_src = v2(residual, D, nt, residual->nb[2], (size_t) src * residual->nb[1]);
                ggml_tensor * cs = v2(comb, 1, nt, comb->nb[2],
                                      (size_t) dst * comb->nb[0] + (size_t) src * comb->nb[1]);
                cur = ggml_add(c, cur, ggml_mul(c, res_src, cs));
            }
            cur = ggml_reshape_3d(c, cur, D, 1, nt);
            out = out ? ggml_concat(c, out, cur, 1) : cur;
        }
        return out;
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wgt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wgt);
    }
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int n_ctx,
                       float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext(c, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) (g->d_head - n_dims));
    }
    ggml_tensor * rope_back(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int n_ctx,
                            float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext_back(c, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) (g->d_head - n_dims));
    }
    ggml_tensor * rope_at(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int64_t off, int n_ctx,
                          float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext(c, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) off);
    }
    // orthonormal Walsh-Hadamard rotation over contiguous `n`-blocks (llama-impl.h:57-75 / llama-kv-cache.cpp:24)
    ggml_tensor * hadamard(ggml_tensor * cur, ggml_tensor * rot_m) const {
        const int64_t n = rot_m->ne[0];
        ggml_tensor * res = ggml_is_contiguous(cur)
            ? ggml_reshape_2d(c, cur, n, ggml_nelements(cur) / n)
            : ggml_cont_2d(c, cur, n, ggml_nelements(cur) / n);
        res = ggml_mul_mat(c, rot_m, res);
        // as llama-impl.h: ggml-cuda then runs its fast Walsh-Hadamard kernel instead of the matmul (the CPU ignores it)
        ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
        return ggml_reshape_4d(c, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    }
    // the compress-RoPE / raw-RoPE parameter set ds4_ref derives per layer
    void rope_cfg(int64_t ratio, float & freq_base, float & freq_scale, float & ext_factor, float & attn_factor,
                  float & beta_fast, float & beta_slow, int & n_ctx) const {
        const bool comp = ratio != 0;
        freq_base   = comp ? (float) g->compress_rope_base : (float) g->rope_freq_base;
        freq_scale  = comp ? (float) (1.0 / g->rope_factor) : 1.0f;
        ext_factor  = comp ? 1.0f : 0.0f;
        attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
        beta_fast   = comp ? (float) g->yarn_beta_fast : 0.0f;
        beta_slow   = comp ? (float) g->yarn_beta_slow : 0.0f;
        n_ctx       = comp ? (int) g->rope_orig_ctx : 0;
    }
};

}  // namespace

// ================================================================== graph construction

// Attention + router for one compressed-row capacity `cap` (0 = nothing compressed yet) and `n` tokens per pass
// (positions pos0 .. pos0+n-1, one per column).  All tensors are created once; per-step values arrive through the
// layer's input leaves, so the graph is stable.  n = 1 is the single-token decode graph.
//
// The n tokens share one flash-attn: the raw keys are the union window (SWA + n - 1 keys, oldest -> newest) with a
// per-query window mask, the compressed keys are the cache after every token of the pass wrote its row, with a
// per-query visibility mask (a block completed by token t is visible to tokens > t, never the reverse).  The
// compressor rings take every token's state before any gather; each token gathers its own `ring` slots.
static bool sparse_on();   // DS4_SPARSE, defined with the other switches below
static bool sparse_n_off();   // mab776 DS4_SPARSE_N=0: multi-token decode passes back on the masked path
static bool vis_dev_on();  // DS4_VIS_DEV, ditto
static bool pf_argsort_on();   // mab776 M10 DS4_PF_ARGSORT=1: prompt chunks back on the full-row argsort for the top-k
static bool pf_lean_mask_on(); // mab776 M10 DS4_PF_LEAN_MASK=0: prompt chunks back on the dense visibility masks
static bool pf_hca1_on();      // mab776 M10 DS4_PF_HCA1=1: HCA chunk attention in one call (opt-in, see below)
static bool pf_hcab_on();      // mab776 M10c DS4_PF_HCAB: HCA chunk blocks as the sequences of one call (default on)
static bool pf_emit_on();      // mab776 M10d DS4_PF_EMIT: chunk compressors only for the rows that land (default on)
static Ds4Dense::Impl::PfRatio * pf_ratio_set(Ds4Dense::Impl & im, int64_t ratio);
static bool pf_lean_idx_on();  // ... DS4_PF_LEAN_IDX=0 / DS4_PF_LEAN_FA=0: one half of the lean masks off (diagnostics)
static bool pf_lean_fa_on();
static int64_t idx_block_n(int64_t n);   // DS4_IDX_BLOCK, ditto
static int64_t raw_block_n(int64_t n);   // DS4_RAW_BLOCK, ditto
static bool pf_sparse_on();   // mab776 DS4_PF_SPARSE, ditto
static bool fused_idx_on(bool cpu);     // DS4_FUSED_IDX, ditto
static bool ovl_gather_on();            // DS4_OVL_GATHER, ditto

static ggml_cgraph * build_attn(Ds4Dense::Impl & im, int il, int64_t cap, int64_t n, Ds4Dense::Impl::Layer & L,
                               bool emit = true) {
    Ds4Dense::Impl::Var var;
    var.cap = cap;
    var.n = n;
    var.emit = emit = emit || im.pf || n != 1;   // only one-token decode skips

    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 4096, false);
    B b { &im, gc, &im.g };

    const int64_t D = im.D, HC = im.HC, DH = im.DH, DHR = im.DHR, NH = im.NH, OGD = im.OGD, OL = im.OL, OG = im.OG;
    const int64_t SWA = im.SWA;
    const int64_t n_raw = SWA + n - 1;            // union of the n raw windows
    const int64_t ratio = L.ratio;
    const bool csa = L.csa;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int n_ctx;
    b.rope_cfg(ratio, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_ctx);
    const float crb = (float) im.g.compress_rope_base;

    auto first = [&](ggml_tensor * t, int64_t ne) { return ggml_view_1d(gc, t, ne, 0); };
    // a 2-D view of `t`: a free reshape when it is already contiguous (most matmul / rope / set_rows results, and a
    // permute that only moves a size-1 axis), a CONT copy otherwise.  Same bits either way; the copies were ~10 of the
    // ~27 CONT nodes of a decode CSA layer (op census, s35).
    auto flat2 = [&](ggml_tensor * t, int64_t ne0, int64_t ne1) {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(gc, t, ne0, ne1) : ggml_cont_2d(gc, t, ne0, ne1);
    };
    ggml_tensor * pos = first(im.i_pos, n);

    // ---- attention hyper-connection -------------------------------------------------
    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * attn_pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                      b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * xn = b.rms_w(attn_pre, b.BL(il, "attn_norm.weight"));

    // ---- q latent (kept for the indexer's query) --------------------------------------
    ggml_tensor * qr2 = ggml_mul_mat(gc, b.BL(il, "attn_q_a.weight"), xn);
    qr2 = b.rms_w(qr2, b.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_mul_mat(gc, b.BL(il, "attn_q_b.weight"), qr2);
    q = ggml_reshape_3d(gc, q, DH, NH, n);
    q = ggml_rms_norm(gc, q, (float) im.g.rms_eps);
    q = b.rope(q, pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- kv latent (K == V) -----------------------------------------------------------
    ggml_tensor * kv = ggml_mul_mat(gc, b.BL(il, "attn_kv.weight"), xn);
    kv = b.rms_w(kv, b.BL(il, "attn_kv_a_norm.weight"));
    kv = ggml_reshape_3d(gc, kv, DH, 1, n);
    kv = b.rope(kv, pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- raw sliding-window ring ------------------------------------------------------
    // chunk (im.pf): the window gathers from [ring | the chunk's keys]; the ring gets the chunk's last keys after
    // everything else has read it (`wb`, expanded last - ggml runs the nodes in this order)
    std::vector<ggml_tensor *> wb;
    auto tail_into = [&](ggml_tensor * ring, ggml_tensor * rows2, int64_t ring_sz, ggml_tensor * slots) {
        const int64_t k = std::min<int64_t>(n, ring_sz);
        ggml_tensor * last = ggml_view_2d(gc, rows2, rows2->ne[0], k, rows2->nb[1], (size_t) (n - k) * rows2->nb[1]);
        wb.push_back(ggml_set_rows(gc, ring, last, first(slots, k)));
    };
    ggml_tensor * raw_src = nullptr;
    if (!im.pf) {
        ggml_tensor * raw2 = ggml_set_rows(gc, L.raw, flat2(kv, DH, n), first(L.i_slot_raw, n));
        raw_src = ggml_get_rows(gc, raw2, first(L.i_idx_raw, n_raw));   // [DH, n_raw] oldest -> newest
    } else {
        ggml_tensor * kv2 = flat2(kv, DH, n);
        raw_src = ggml_get_rows(gc, ggml_concat(gc, L.raw, kv2, 1), first(L.i_idx_raw, n_raw));
        tail_into(L.raw, kv2, im.RAW, L.i_slot_raw);
    }
    ggml_tensor * k_raw = ggml_reshape_3d(gc, raw_src, DH, 1, n_raw);             // token axis on ne2 (ds4_ref layout)
    ggml_tensor * m_raw = ggml_reshape_2d(gc, first(L.i_mask_raw, n_raw * n), n_raw, n);
    ggml_tensor * k_cmp = nullptr, * m_cmp = nullptr;   // the compressed keys [DH, 1, rows] and their mask [rows, n]

    // ---- compressed keys -------------------------------------------------------------
    if (cap > 0) {
        ggml_tensor * comp_k = nullptr;
        ggml_tensor * cmask = nullptr;
        ggml_tensor * sel = nullptr;   // sparse CSA (one-token decode): the compressed rows attention gathers
        ggml_tensor * state_pos = first(L.i_state_pos, n);
        // mab776 M10d (DS4_PF_EMIT): a prompt chunk's compressors build only the rows that land in a slot - each block's
        // last token in the chunk (and the chunk's last token: its partial block's row) - not one candidate per token
        // that the slot `comp_max` swallowed (an HCA layer-chunk gathered [DH, 128, n] twice, ~512 MB each at n 2048,
        // to keep 16 rows).  The kept rows are computed exactly as before; the host sends their positions, slots and
        // state indices (prefill_begin).  Decode and verify passes keep one row per token.
        const int64_t ne = im.pf && pf_emit_on() ? pf_ratio_set(im, L.ratio)->n_emit : n;
        ggml_tensor * comp_pos  = first(L.i_comp_pos, ne);
        ggml_tensor * slot_st   = im.pf ? nullptr : first(L.i_slot_state, n);   // chunk: tail_into uses its own
        ggml_tensor * idx_st    = first(L.i_idx_state, L.ring * ne);  // [ring] per emitted row, flat (get_rows takes 1-D)
        ggml_tensor * vis_cnt   = nullptr;   // [1, n] F32 = i_nvis[t] (DS4_VIS_DEV): the lean masks' counts
        if (vis_dev_on()) {
            // DS4_VIS_DEV (default on): build the compressed-row visibility on the device from the per-layer count
            // `i_nvis` (the blocks each query may see) instead of a host mask.  The decode host mask carried
            // comp_max * kNtMax floats per compressed layer in the per-token input span, and the chunk mask
            // comp_max * N floats (at 256K: ~22 MiB per token, and 1 GiB per chunk per ratio, uploaded every
            // chunk), of which only `cap` rows were ever used (FINDINGS s22).  Row i is visible to query t iff
            // block i completed at or before pos_t, i.e. (i+1)*ratio <= pos_t+1 <=> i < i_nvis[t].  step() gives
            // exactly 1 (visible) / 0 (hidden) and log() exactly 0.0f / -INFINITY - the host's values; scaling by
            // NEG_INF instead would be 0 * inf = NaN.
            // nv - i as one broadcast add of the F32 count onto the persistent -i ramp (was cast + arange + 2 repeats
            // + sub: 7 kernels per compressed layer per token -> 3); small integers, so the same exact floats
            ggml_tensor * nvs = ggml_reshape_2d(gc, first(L.i_nvis, n), 1, n);                 // [1, n] = i_nvis[t]
            vis_cnt = nvs;
            ggml_tensor * ivN = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * sizeof(float), 0);   // -i
            if (n > 1) ivN = ggml_repeat(gc, ivN, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, n));
            var.vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ivN, nvs)));                      // i < i_nvis -> 0
        } else {
            var.vis = ggml_view_2d(gc, L.vis_full, cap, n, (size_t) cap * sizeof(float), 0);
        }

        if (csa) {
            // overlap compressor (ds4_ref attention():514-556): each token state is [ prev-half | cur-half ]
            auto overlap = [&](const std::string & pfx, int64_t head_dim, ggml_tensor * ring_kv, ggml_tensor * ring_sc)
                    -> ggml_tensor * {
                ggml_tensor * st_kv = ggml_mul_mat(gc, b.BL(il, pfx + "_kv.weight"), xn);
                ggml_tensor * st_sc = ggml_mul_mat(gc, b.BL(il, pfx + "_gate.weight"), xn);
                ggml_tensor * ape = ggml_get_rows(gc, b.BL(il, pfx + "_ape.weight"), state_pos);
                st_sc = ggml_add(gc, st_sc, ape);

                ggml_tensor * skv = flat2(st_kv, 2 * head_dim, n);
                ggml_tensor * ssc = flat2(st_sc, 2 * head_dim, n);
                ggml_tensor * rkv, * rsc;
                if (!im.pf) {
                    rkv = ggml_set_rows(gc, ring_kv, skv, slot_st);
                    rsc = ggml_set_rows(gc, ring_sc, ssc, slot_st);
                    if (!emit) {   // the states only: no block completes at this position
                        wb.push_back(rkv);
                        wb.push_back(rsc);
                        return nullptr;
                    }
                } else {   // [ring | the chunk's states]; the ring gets the chunk's last states afterwards
                    rkv = ggml_concat(gc, ring_kv, skv, 1);
                    rsc = ggml_concat(gc, ring_sc, ssc, 1);
                    tail_into(ring_kv, skv, L.ring_sz, L.i_slot_state);
                    tail_into(ring_sc, ssc, L.ring_sz, L.i_slot_state);
                }
                ggml_tensor * values, * scores;                               // [hd, 2*ratio, n]
                if (ovl_gather_on()) {
                    // DS4_OVL_GATHER (default on): the ring seen as [hd, 2 * slots] - state row s's prev half is
                    // half-row 2s, its cur half 2s+1 - and the host index list already names the half each of the
                    // 2*ratio gathered rows needs (2*slot for the first ratio, 2*slot+1 for the rest), so one
                    // get_rows lands the values in place: the same bits as gather + 4 strided conts + 2 concats
                    values = ggml_reshape_3d(gc, ggml_get_rows(gc, ggml_reshape_2d(gc, rkv, head_dim, 2 * rkv->ne[1]),
                                                               idx_st), head_dim, 2 * ratio, ne);
                    scores = ggml_reshape_3d(gc, ggml_get_rows(gc, ggml_reshape_2d(gc, rsc, head_dim, 2 * rsc->ne[1]),
                                                               idx_st), head_dim, 2 * ratio, ne);
                } else {
                    ggml_tensor * rows_kv = ggml_reshape_3d(gc, ggml_get_rows(gc, rkv, idx_st), 2 * head_dim, 2 * ratio, ne);
                    ggml_tensor * rows_sc = ggml_reshape_3d(gc, ggml_get_rows(gc, rsc, idx_st), 2 * head_dim, 2 * ratio, ne);
                    const size_t nb1 = rows_kv->nb[1], nb2 = rows_kv->nb[2];
                    const size_t cur_off = (size_t) ratio * nb1 + (size_t) head_dim * 4;
                    ggml_tensor * pkv = ggml_cont(gc, ggml_view_3d(gc, rows_kv, head_dim, ratio, ne, nb1, nb2, 0));
                    ggml_tensor * ckv = ggml_cont(gc, ggml_view_3d(gc, rows_kv, head_dim, ratio, ne, nb1, nb2, cur_off));
                    ggml_tensor * psc = ggml_cont(gc, ggml_view_3d(gc, rows_sc, head_dim, ratio, ne, nb1, nb2, 0));
                    ggml_tensor * csc = ggml_cont(gc, ggml_view_3d(gc, rows_sc, head_dim, ratio, ne, nb1, nb2, cur_off));
                    values = ggml_concat(gc, pkv, ckv, 1);
                    scores = ggml_concat(gc, psc, csc, 1);
                }
                values = ggml_cont(gc, ggml_permute(gc, values, 1, 0, 2, 3)); // [2*ratio, hd, n]
                scores = ggml_cont(gc, ggml_permute(gc, scores, 1, 0, 2, 3));
                ggml_tensor * wts = ggml_soft_max(gc, scores);
                ggml_tensor * cc = ggml_sum_rows(gc, ggml_mul(gc, values, wts));
                cc = ggml_reshape_3d(gc, cc, head_dim, 1, ne);                // [1, hd, ne] -> [hd, 1, ne]: same order
                cc = b.rms_w(cc, b.BL(il, pfx + "_norm.weight"));
                return b.rope_at(cc, comp_pos, DHR, head_dim - DHR, n_ctx, crb,
                                 freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            };

            comp_k = overlap("attn_compressor", DH, L.st_kv, L.st_sc);

            // lightning indexer: compressed keys (post-WHT) + query, then top-k over the visible blocks, per query
            ggml_tensor * icomp2 = L.icomp;   // !emit: the block keys as they are (the current block is invisible)
            if (ggml_tensor * lk = overlap("indexer_compressor", im.IDXK, L.ist_kv, L.ist_sc)) {
                ggml_tensor * lid_k = b.hadamard(ggml_is_contiguous(lk) ? lk : ggml_cont(gc, lk), im.rot);
                icomp2 = ggml_set_rows(gc, L.icomp, flat2(lid_k, im.IDXK, ne), first(L.i_slot_comp, ne));
            }
            // ds4_ref's lid_k is 3-D [idx_k, 1, n_blocks]; this 3-D view of the ring's first `cap`
            // columns has the same shape, so the permute below lands `cap` on ne[1] exactly like the
            // reference's kp = permute(lid_k, 0,2,1,3).  A 2-D view would put `cap` on ne[2] and the
            // mul_mat would then drop the block axis (the result would be [1,1,idx_h] instead of
            // [cap,1,idx_h]) - the decoded indexer scores would be wrong and the graph layout invalid.
            ggml_tensor * lid_src = ggml_view_3d(gc, icomp2, im.IDXK, 1, cap,
                                                 icomp2->nb[1], icomp2->nb[1], 0);

            ggml_tensor * iq = ggml_mul_mat(gc, b.BL(il, "indexer.attn_q_b.weight"), qr2);
            iq = ggml_reshape_3d(gc, iq, im.IDXK, im.IDXH, n);
            iq = b.rope_at(iq, pos, DHR, im.IDXK - DHR, n_ctx, crb,
                           freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            iq = b.hadamard(iq, im.rot);
            ggml_tensor * iw = ggml_mul_mat(gc, b.BL(il, "indexer.proj.weight"), xn);
            iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.IDXK * im.IDXH)));

            ggml_tensor * qp = ggml_permute(gc, iq, 0, 2, 1, 3);            // [idx_k, n, idx_h]
            ggml_tensor * kp = icomp2->type == GGML_TYPE_F32 ? ggml_permute(gc, lid_src, 0, 2, 1, 3)   // [idx_k, cap, 1]
                             : ggml_view_2d(gc, icomp2, im.IDXK, cap, icomp2->nb[1], 0);   // --icomp-q8: same rows
            // score[c,t] = sum_h relu(sum_k lid_k[k,c] * iq[k,h,t]) * iw[h,t].  The mul_mat's result is
            // [cap, n, idx_h] and the relu / head weight / sum need idx_h on ne[0], so the permute + cont
            // materialises a second tensor of that size: cap * n * 64 floats.  At a 64K prompt (cap 16384) that is
            // 16 GiB and at 256K (cap 65536) 64 GiB, which is why no long prompt can be read at all (FINDINGS s24).
            // DS4_IDX_BLOCK computes it in query blocks instead - a query's score depends only on its own iq/iw, so
            // the result is the same bits, and the peak becomes cap * block * 64 instead of cap * n * 64.  For
            // n <= block the loop runs once with `nb == n` and builds the identical graph, so decode, verify and
            // every chunk up to the block size are byte-identical by construction (DS4_IDX_BLOCK=0 = all at once).
            int64_t QB = idx_block_n(n);
            ggml_tensor * sc = nullptr;
            // DS4_FUSED_IDX (default off-CPU): ggml's fused lightning-indexer op does score, relu, head weights, sum
            // and mask in one kernel (tensor cores when icomp is F16/Q8_0: --icomp-q8) instead of the ~9-op chain
            // below with its two [cap, n, 64] permute+cont copies.  Same math, different summation order, so the
            // CPU keeps the chain (its gate is bit-exact against ds4_ref).
            // ... and only where the backend has a kernel for these shapes (ggml-cuda: idx_k 128, 32/64 heads - the
            // mini fixtures are smaller and take the chain); the probe node is never added to the graph
            const bool fidx = fused_idx_on(ggml_backend_is_cpu(im.backend)) &&
                ggml_backend_supports_op(im.backend, ggml_lightning_indexer(gc, iq, lid_src, iw,
                                                                            ggml_cast(gc, var.vis, GGML_TYPE_F16)));
            // mab776 M10 (DS4_PF_LEAN_MASK): a chunk hands the fused indexer the per-query counts instead of the mask
            const bool cidx = fidx && im.pf && vis_cnt && pf_lean_idx_on() &&
                ggml_backend_supports_op(im.backend, ggml_lightning_indexer(gc, iq, lid_src, iw, vis_cnt));
            ggml_tensor * vis16 = fidx ? (cidx ? vis_cnt : ggml_cast(gc, var.vis, GGML_TYPE_F16)) : nullptr;
            // mab776: the fused kernel's output is just [cap, n] (no [cap, n, 64] intermediate to bound), so it takes
            // every query in one call unless DS4_IDX_BLOCK is set: the blocks were joined by sc = concat(sc, scb),
            // which re-copied all scores so far per block - ~8 GB a CSA layer-chunk at 250k (cap 65536, 32 blocks)
            if (fidx && !std::getenv("DS4_IDX_BLOCK")) QB = n;
            for (int64_t t0 = 0; t0 < n; t0 += QB) {
                const int64_t nb = std::min<int64_t>(QB, n - t0);
                if (fidx) {
                    ggml_tensor * iqb = nb == n ? iq
                        : ggml_view_3d(gc, iq, im.IDXK, im.IDXH, nb, iq->nb[1], iq->nb[2], (size_t) t0 * iq->nb[2]);
                    ggml_tensor * iwb = nb == n ? iw
                        : ggml_view_2d(gc, iw, im.IDXH, nb, iw->nb[1], (size_t) t0 * iw->nb[1]);
                    ggml_tensor * mb = nb == n ? vis16
                        : ggml_view_2d(gc, vis16, vis16->ne[0], nb, vis16->nb[1], (size_t) t0 * vis16->nb[1]);
                    ggml_tensor * scb = ggml_lightning_indexer(gc, iqb, lid_src, iwb, mb);   // [cap, nb]
                    sc = sc ? ggml_concat(gc, sc, scb, 1) : scb;
                    continue;
                }
                ggml_tensor * qpb = nb == n ? qp
                    : ggml_view_3d(gc, qp, im.IDXK, nb, im.IDXH, qp->nb[1], qp->nb[2], (size_t) t0 * qp->nb[1]);
                ggml_tensor * iwb = nb == n ? iw
                    : ggml_view_2d(gc, iw, im.IDXH, nb, iw->nb[1], (size_t) t0 * iw->nb[1]);
                ggml_tensor * kq = ggml_mul_mat(gc, kp, qpb);               // [cap, nb, idx_h]
                kq = ggml_cont(gc, ggml_permute(gc, kq, 2, 1, 0, 3));       // [idx_h, nb, cap]
                ggml_tensor * scb = ggml_relu(gc, kq);
                scb = ggml_mul(gc, scb, ggml_reshape_4d(gc, iwb, im.IDXH, nb, 1, 1));
                scb = ggml_sum_rows(gc, scb);                               // [1, nb, cap]
                scb = ggml_cont(gc, ggml_permute(gc, scb, 2, 1, 0, 3));     // [cap, nb, 1]
                scb = ggml_reshape_2d(gc, scb, cap, nb);
                ggml_tensor * visb = nb == n ? var.vis
                    : ggml_view_2d(gc, var.vis, cap, nb, var.vis->nb[1], (size_t) t0 * var.vis->nb[1]);
                scb = ggml_add(gc, scb, visb);
                sc = sc ? ggml_concat(gc, sc, scb, 1) : scb;
            }
            var.dbg_isc = sc;
            ggml_set_output(sc);

            // mask = -inf everywhere, 0 at each query's top-k rows, then ANDed with visibility
            const int64_t ntk = std::min<int64_t>(cap, im.IDXTOPK);
            // chunk off-CPU: one batched argsort for every query row (ggml-cuda's top_k launched cub once per row: 3
            // kernels x n rows x 21 CSA layers, ~63k launches for a 3K-token chunk).  The same top-k set except among
            // exact ties (ReLU zeros), which cub breaks arbitrarily too; the CPU keeps top_k (no launch cost) so its
            // chunk gate stays bit-exact against the decode loop.  mab776 M10: with the radix-select patch ggml-cuda's
            // top_k is ONE launch for all rows (and needs no [cap, n] index buffer) - DS4_PF_ARGSORT=1 = the argsort
            const bool batched_tk = im.pf && !ggml_backend_is_cpu(im.backend) && pf_argsort_on();
            ggml_tensor * tk = ggml_cont(gc, batched_tk ? ggml_argsort_top_k(gc, sc, (int) ntk)
                                                        : ggml_top_k(gc, sc, (int) ntk));   // [ntk, n]
            if (n == 1 && !im.pf && sparse_on()) {
                // DS4_SPARSE (default): attend to the indexer's rows only - gather them instead of masking every block
                // (attention cost stops growing with the context).  The rows are the top-k set in ascending block
                // order - the order the masked flash-attention visits its unmasked keys, so the CPU result is the same
                // bits.  Sorting the ntk indices themselves (not an argsort of a [cap] mask, as before) keeps the
                // whole selection O(ntk): a selected row's mask is just its visibility, and the invisible rows the
                // top-k can only pick when fewer than ntk are visible all sit above every visible one (i >= i_nvis),
                // so they land in the tail, masked -inf, exactly as before.
                ggml_tensor * tk1 = ggml_reshape_2d(gc, tk, 1, ntk);                                     // [1, ntk]
                ggml_tensor * ord = ggml_argsort(gc, ggml_cast(gc, ggml_reshape_1d(gc, tk, ntk), GGML_TYPE_F32),
                                                 GGML_SORT_ORDER_ASC);
                sel = ggml_reshape_1d(gc, ggml_get_rows(gc, tk1, ord), ntk);                             // I32 [ntk]
                ggml_tensor * msel = ggml_get_rows(gc, ggml_reshape_2d(gc, var.vis, 1, cap), sel);      // [1, ntk]
                cmask = ggml_cast(gc, ggml_reshape_2d(gc, msel, ntk, 1), GGML_TYPE_F16);
            } else if (n > 1 && !im.pf && sparse_on() && !sparse_n_off()) {
                // mab776 M7b: the same gather for a multi-token decode pass (speculative verify, n <= kNtMax).  The
                // masked path below ran a dense flash-attn over all `cap` compressed rows per query (23k at 93k
                // tokens, 65k at 262k: a 2-token pass cost 2.3x a one-token step at 93k, so verify LOST to plain
                // decoding at depth).  Here every query gathers its own top-k rows, sorted ascending: K = [query 0's
                // ntk rows | query 1's | ...], each query's mask open on its own block only (and its visibility), -inf
                // on the others.  Each query attends exactly the keys it did, in the order it did (the reference
                // flash-attn skips -inf keys): bit-identical on the CPU.  Per-row argsort and batched get_rows keep it
                // to a handful of ops for any n.
                ggml_tensor * ordn = ggml_argsort(gc, ggml_cast(gc, tk, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);  // [ntk, n]
                ggml_tensor * seln = ggml_get_rows(gc, ggml_reshape_3d(gc, tk, 1, ntk, n), ordn);            // [1, ntk, n]
                sel = ggml_reshape_1d(gc, seln, ntk * n);                                                  // I32 [ntk * n]
                ggml_tensor * visn = ggml_reshape_3d(gc, ggml_cont(gc, var.vis), 1, cap, n);
                ggml_tensor * mseln = ggml_cont(gc, ggml_reshape_2d(gc, ggml_get_rows(gc, visn,
                                                    ggml_reshape_2d(gc, seln, ntk, n)), ntk, n));          // [ntk, n]
                ggml_tensor * mk = nullptr;
                for (int64_t j = 0; j < n; ++j) {
                    ggml_tensor * col = ggml_view_2d(gc, mseln, ntk, 1, mseln->nb[1], (size_t) j * mseln->nb[1]);
                    if (j > 0) col = ggml_concat(gc, b.fill_f32({ j * ntk, 1 }, NEG_INF), col, 0);
                    if (j < n - 1) col = ggml_concat(gc, col, b.fill_f32({ (n - 1 - j) * ntk, 1 }, NEG_INF), 0);
                    mk = mk ? ggml_concat(gc, mk, col, 1) : col;                                          // [ntk * n, j + 1]
                }
                cmask = ggml_cast(gc, mk, GGML_TYPE_F16);
            } else {
                // mab776 M10 (DS4_PF_LEAN_MASK): every query of the chunk sees >= ntk blocks -> the top-k picked
                // visible rows only, so -inf everywhere + 0 at the picks IS the mask: built in F16, no add, no cast
                const bool lean = im.pf && !ggml_backend_is_cpu(im.backend) && pf_lean_fa_on() &&
                                  ((int64_t) im.pf_pos0 + 1) / L.ratio >= ntk;
                ggml_tensor * a = lean ? ggml_fill_inplace(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F16, cap, n), NEG_INF)
                                       : ggml_fill(gc, var.vis, NEG_INF);
                a = ggml_view_4d(gc, a, 1, cap, n, 1, a->nb[0], a->nb[1], a->nb[2], 0);
                ggml_tensor * zv = b.fill_f32({ 1, ntk, n }, 0.0f);
                ggml_tensor * m = ggml_set_rows(gc, a, zv, tk);                     // query j's indices into slice j
                m = ggml_view_4d(gc, m, m->ne[1], m->ne[2], 1, 1, m->nb[2], m->nb[3], m->nb[3], 0);
                cmask = lean ? m : ggml_cast(gc, ggml_add(gc, m, var.vis), GGML_TYPE_F16);
            }
        } else {
            // HCA: one block == `ratio` consecutive tokens, no overlap (ds4_ref attention():603-622)
            ggml_tensor * st_kv = ggml_mul_mat(gc, b.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * st_sc = ggml_mul_mat(gc, b.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_tensor * ape = ggml_get_rows(gc, b.BL(il, "attn_compressor_ape.weight"), state_pos);
            st_sc = ggml_add(gc, st_sc, ape);

            ggml_tensor * skv = flat2(st_kv, DH, n);
            ggml_tensor * ssc = flat2(st_sc, DH, n);
            ggml_tensor * rkv, * rsc;
            if (!im.pf) {
                rkv = ggml_set_rows(gc, L.st_kv, skv, slot_st);
                rsc = ggml_set_rows(gc, L.st_sc, ssc, slot_st);
                if (!emit) { wb.push_back(rkv); wb.push_back(rsc); }
            } else {
                rkv = ggml_concat(gc, L.st_kv, skv, 1);
                rsc = ggml_concat(gc, L.st_sc, ssc, 1);
                tail_into(L.st_kv, skv, L.ring_sz, L.i_slot_state);
                tail_into(L.st_sc, ssc, L.ring_sz, L.i_slot_state);
            }
            if (emit) {
                ggml_tensor * rows_kv = ggml_reshape_3d(gc, ggml_get_rows(gc, rkv, idx_st), DH, ratio, ne);
                ggml_tensor * rows_sc = ggml_reshape_3d(gc, ggml_get_rows(gc, rsc, idx_st), DH, ratio, ne);
                ggml_tensor * values = ggml_cont(gc, ggml_permute(gc, rows_kv, 1, 0, 2, 3));   // [ratio, DH, n]
                ggml_tensor * scores = ggml_cont(gc, ggml_permute(gc, rows_sc, 1, 0, 2, 3));
                ggml_tensor * wts = ggml_soft_max(gc, scores);
                ggml_tensor * cc = ggml_sum_rows(gc, ggml_mul(gc, values, wts));  // [1, DH, n]
                cc = ggml_reshape_3d(gc, cc, DH, 1, ne);                          // [1, DH, ne] -> [DH, 1, ne]
                cc = b.rms_w(cc, b.BL(il, "attn_compressor_norm.weight"));
                comp_k = b.rope_at(cc, comp_pos, DHR, DH - DHR, n_ctx, crb,
                                   freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cmask = ggml_cast(gc, var.vis, GGML_TYPE_F16);
        }

        // comp_rot: rotate each row before it is rounded into the cache, and back after it is read (H = H^-1)
        ggml_tensor * comp2 = comp_k ? ggml_set_rows(gc, L.comp, im.comp_rot ? b.hadamard(flat2(comp_k, DH, ne), im.rot)
                                                                            : flat2(comp_k, DH, ne),
                                                     first(L.i_slot_comp, ne))
                                     : L.comp;   // !emit: no block completes here
        if (sel) {   // sparse: the gathered rows (ascending block order)
            ggml_tensor * rows = ggml_get_rows(gc, comp2, sel);                 // [DH, ntk]
            if (im.comp_rot) rows = b.hadamard(rows, im.rot);
            k_cmp = ggml_reshape_3d(gc, rows, DH, 1, rows->ne[1]);
        } else {
            // a raw view of the cache works only while it is F32: a quantized comp needs the dequantizing gather
            ggml_tensor * comp_src = L.i_idx_all ? ggml_get_rows(gc, comp2, first(L.i_idx_all, cap))
                                                 : ggml_view_2d(gc, comp2, DH, cap, comp2->nb[1], 0);
            if (im.comp_rot) comp_src = b.hadamard(comp_src, im.rot);
            k_cmp = ggml_reshape_3d(gc, comp_src, DH, 1, cap);
        }
        m_cmp = cmask;
    }

    // ---- attention --------------------------------------------------------------------
    // ggml-cuda has a flash-attn kernel for d_head 512 only when the key count is a multiple of FATTN_KQ_STRIDE
    // (256) - llama.cpp pads its KV cache to that.  Off-CPU, pad with zero keys masked -inf: they get exactly zero
    // weight.  The CPU path (the one verified bit-exact against ds4_ref) is left as it was.
    //
    // DS4_RAW_BLOCK (FINDINGS s31 addendum 3): a query's raw window is SWA keys, but one flash-attn over all n queries
    // makes each of them visit the whole union window (SWA + n - 1 keys, all but SWA masked) - 33x the raw work at a
    // 4096 chunk.  Blocks of nb queries attend [their union window (SWA + nb - 1) | the compressed rows] instead, and
    // the outputs are concatenated.  Query t sees union rows [t, t + SWA), so block [t0, t0 + nb) needs rows
    // [t0, t0 + SWA + nb - 1).  The dropped keys were all masked for every query of the block, and the reference
    // flash-attn (DS4_CPU_FA_REF=1) skips -inf keys, so the chunk gate stays bit-exact.  n <= block (decode, verify,
    // small chunks) builds the identical single-call graph.
    auto attend = [&](ggml_tensor * qb, ggml_tensor * kb, ggml_tensor * mb, int64_t nq) {
        if (!ggml_backend_is_cpu(im.backend)) {
            const int64_t n_kv = kb->ne[2], pad = (256 - n_kv % 256) % 256;
            if (pad) {
                // mab776: ggml-cuda's pad kernel puts ne1 / ne2*ne3 in gridDim.y / gridDim.z (max 65535), and here
                // ne2 is the key count: a 250k prompt has ~67.8k keys (cap 65536 + the raw window) -> "PAD failed:
                // invalid argument" mid-prefill.  Past that, pad the flat buffer instead: the same zeros, the launch
                // then spans gridDim.x.
                if (n_kv + pad > 65535) {
                    ggml_tensor * k1 = ggml_is_contiguous(kb) ? kb : ggml_cont(gc, kb);
                    k1 = ggml_reshape_1d(gc, k1, ggml_nelements(kb));
                    k1 = ggml_pad(gc, k1, (int) (pad * kb->ne[0] * kb->ne[1]), 0, 0, 0);
                    kb = ggml_reshape_3d(gc, k1, kb->ne[0], kb->ne[1], n_kv + pad);
                } else {
                    kb = ggml_pad(gc, kb, 0, 0, (int) pad, 0);
                }
                mb = ggml_concat(gc, mb, ggml_cast(gc, b.fill_f32({ pad, nq }, NEG_INF), GGML_TYPE_F16), 0);
            }
        }
        ggml_tensor * qp = ggml_permute(gc, qb, 0, 2, 1, 3);                 // [DH, nq, NH]
        ggml_tensor * kp = ggml_permute(gc, kb, 0, 2, 1, 3);
        ggml_tensor * kf = ggml_cast(gc, kp, GGML_TYPE_F16);   // K == V: one cast serves both (was two identical)
        ggml_tensor * o = ggml_flash_attn_ext(gc, qp, kf, kf, mb, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, b.BL(il, "attn_sinks.weight"));   // [DH, NH, nq]
        return o;
    };
    const int64_t RB = raw_block_n(n);
    ggml_tensor * out = nullptr;
    // mab776 (prefill at depth): a CSA chunk's queries each see SWA raw keys + their top-k compressed rows, yet the
    // blocked path below ran a DENSE masked flash-attn over every compressed row (cap: 32k at 127k, 64k at 250k) -
    // ~98 % of its work on -inf keys - and re-concatenated, re-cast and re-padded the whole compressed K for every
    // block of RB queries (nsys @85k: flash-attn 29 %, those copies ~17 % of the GPU).  Here: K = [compressed rows |
    // raw union window | zero keys to a multiple of 256] is cast to F16 ONCE; every block of nqb queries is one
    // sequence (ne[3]) of ONE flash-attn call over it (K broadcast, nb[3] = 0) with its rows of the [T, n] mask, and
    // n_kv_max = SWA + top-k (the finite entries of every mask row) lets ggml-cuda gather just those keys per query
    // (its sparse path: DKQ = DV = 512, one query x 8 heads a CTA).  nqb is the largest block for which that path
    // qualifies (keys >= max(4096, 2 x nqb x n_kv_max)); shallower chunks keep the blocked path.  Same keys, same
    // masked set: only the summation order of the softmax differs.  DS4_PF_SPARSE=0: the blocked path everywhere.
    int64_t nqb = 0;
    bool dense1 = false;   // a shallow CSA chunk: the same layout, one DENSE call (no n_kv_max)
    const int64_t nkv_q = SWA + (csa ? std::min<int64_t>(cap, im.IDXTOPK) : 0);
    if (im.pf && csa && k_cmp && m_cmp && RB < n && !ggml_backend_is_cpu(im.backend) && pf_sparse_on()) {   // (chunk mode never gathers: `sel` is one-token decode)
        for (int64_t c : { (int64_t) 64, (int64_t) 32, (int64_t) 16, (int64_t) 8 })
            if (cap >= std::max<int64_t>(4096, 2 * c * nkv_q)) { nqb = c; break; }
        // below the sparse threshold (cap < ~10k): all n queries in ONE dense call over [compressed | raw | pad]
        // instead of RB-blocks that each concat + cast + pad the compressed K (~32 of them a layer-chunk); with the
        // compressed rows first, ggml-cuda's tile skip (n >= 1024) stops each query tile at its window's end.
        // DS4_PF_DENSE1=0: the blocked path for these (A/B).
        static const bool d1 = [] { const char * e = std::getenv("DS4_PF_DENSE1"); return !e || std::atoi(e) != 0; }();
        if (nqb == 0 && d1) { nqb = n; dense1 = true; }
    }
    // mab776 M10 (DS4_PF_HCA1): an HCA chunk in the same one dense call (its compressed rows are few: <= 2,048 at 262k)
    if (nqb == 0 && im.pf && !csa && k_cmp && m_cmp && RB < n && !ggml_backend_is_cpu(im.backend) && pf_hca1_on()) {
        nqb = n;
        dense1 = true;
    }
    auto blocked = [&]() -> ggml_tensor * {
        std::vector<ggml_tensor *> outs;
        for (int64_t t0 = 0; t0 < n; t0 += RB) {
            const int64_t nb = std::min<int64_t>(RB, n - t0), nw = SWA + nb - 1;
            ggml_tensor * kb = ggml_cont(gc, ggml_view_3d(gc, k_raw, DH, 1, nw, k_raw->nb[1], k_raw->nb[2],
                                                          (size_t) t0 * k_raw->nb[2]));
            ggml_tensor * mb = ggml_cont(gc, ggml_view_2d(gc, m_raw, nw, nb, m_raw->nb[1],
                                                          (size_t) t0 * m_raw->nb[1] + (size_t) t0 * m_raw->nb[0]));
            if (k_cmp) {
                kb = ggml_concat(gc, kb, k_cmp, 2);
                mb = ggml_concat(gc, mb, ggml_cont(gc, ggml_view_2d(gc, m_cmp, m_cmp->ne[0], nb, m_cmp->nb[1],
                                                                    (size_t) t0 * m_cmp->nb[1])), 0);
            }
            ggml_tensor * qb = ggml_view_3d(gc, q, DH, NH, nb, q->nb[1], q->nb[2], (size_t) t0 * q->nb[2]);
            outs.push_back(attend(qb, kb, mb, nb));
        }
        // mab776: the blocks' outputs joined pairwise (a balanced tree, in order): `out = concat(out, ob)` per block
        // copied everything so far each time - 32 blocks of a 2048 chunk moved ~4 GB a layer (nsys: concat 9 %)
        while (outs.size() > 1) {
            std::vector<ggml_tensor *> nx;
            for (size_t i = 0; i + 1 < outs.size(); i += 2) nx.push_back(ggml_concat(gc, outs[i], outs[i + 1], 2));
            if (outs.size() % 2) nx.push_back(outs.back());
            outs.swap(nx);
        }
        return outs[0];
    };
    // mab776 M10c (DS4_PF_HCAB, default on): an HCA chunk's query blocks in ONE flash-attn call.  The blocked path
    // below gives every block of RB queries its own K = [its union window (SWA + RB - 1 raw rows) | the compressed rows
    // | zero pad] and mask, attends, and joins the outputs: ~160 small ops a layer-chunk (nsys @250k: the concats, casts
    // and pads ~6 % of the GPU, the 32 launches with their stream-k fixups ~7 %).  Here the same per-block K and mask are
    // built for every block at once - the windows as an overlapping strided view of k_raw, the compressed rows (+ pad)
    // repeated per block - and each block is one sequence (ne[3]) of one call: the same keys, masks and key order as
    // the blocked path, only the kernel's work split can differ.  DS4_PF_HCAB=0: the blocked path.
    const bool hcab = nqb == 0 && im.pf && !csa && k_cmp && m_cmp && RB < n && n / RB >= 2 &&
                      !ggml_backend_is_cpu(im.backend) && pf_hcab_on();
    if (hcab) {
        const int64_t nseq = n / RB, rem = n - nseq * RB, nw = SWA + RB - 1, nkb = nw + cap;
        const int64_t T = (nkb + 255) / 256 * 256, pad = T - nkb;
        // K = [window | compressed | pad] per block, F16: [DH, T, nseq]
        ggml_tensor * kc = ggml_reshape_3d(gc, ggml_cast(gc, k_cmp, GGML_TYPE_F16), DH, cap, 1);
        if (pad) kc = ggml_concat(gc, kc, ggml_fill_inplace(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F16, DH, pad, 1), 0.0f), 1);
        kc = ggml_repeat_4d(gc, kc, DH, cap + pad, nseq, 1);
        // the windows overlap (block s = rows [s RB, s RB + nw)): ggml sizes a view as if contiguous, so it is made one
        // window deep and given its block count and stride after (the last row read is n_raw - 1 at most)
        ggml_tensor * kw = ggml_view_3d(gc, k_raw, DH, nw, 1, k_raw->nb[2], (size_t) nw * k_raw->nb[2], 0);
        kw->ne[2] = nseq;
        kw->nb[2] = (size_t) RB * k_raw->nb[2];
        kw->nb[3] = kw->nb[2] * nseq;
        ggml_tensor * k4 = ggml_reshape_4d(gc, ggml_concat(gc, ggml_cast(gc, kw, GGML_TYPE_F16), kc, 1), DH, T, 1, nseq);
        // mask = [window rows | compressed rows | -inf] per block: [T, RB, nseq]
        ggml_tensor * mc = m_cmp;
        if (pad) mc = ggml_concat(gc, mc, ggml_fill_inplace(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F16, pad, n), NEG_INF), 0);
        mc = ggml_view_3d(gc, mc, cap + pad, RB, nseq, mc->nb[1], (size_t) RB * mc->nb[1], 0);
        ggml_tensor * mw = ggml_cont(gc, ggml_view_3d(gc, m_raw, nw, RB, nseq, m_raw->nb[1],
                                                      (size_t) RB * (m_raw->nb[1] + m_raw->nb[0]), 0));
        ggml_tensor * m4 = ggml_reshape_4d(gc, ggml_concat(gc, mw, mc, 0), T, RB, 1, nseq);
        ggml_tensor * q4 = ggml_view_4d(gc, q, DH, NH, RB, nseq, q->nb[1], q->nb[2], (size_t) RB * q->nb[2], 0);
        ggml_tensor * o = ggml_flash_attn_ext(gc, ggml_permute(gc, q4, 0, 2, 1, 3), k4, k4, m4,
                                              1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, b.BL(il, "attn_sinks.weight"));
        out = ggml_reshape_3d(gc, o, DH, NH, nseq * RB);                            // [DH, NH, nseq * RB]
        if (rem) {   // the last partial block: the blocked path's own graph
            const int64_t t0 = nseq * RB, nwr = SWA + rem - 1;
            ggml_tensor * kb = ggml_cont(gc, ggml_view_3d(gc, k_raw, DH, 1, nwr, k_raw->nb[1], k_raw->nb[2],
                                                          (size_t) t0 * k_raw->nb[2]));
            ggml_tensor * mb = ggml_cont(gc, ggml_view_2d(gc, m_raw, nwr, rem, m_raw->nb[1],
                                                          (size_t) t0 * m_raw->nb[1] + (size_t) t0 * m_raw->nb[0]));
            kb = ggml_concat(gc, kb, k_cmp, 2);
            mb = ggml_concat(gc, mb, ggml_cont(gc, ggml_view_2d(gc, m_cmp, m_cmp->ne[0], rem, m_cmp->nb[1],
                                                                (size_t) t0 * m_cmp->nb[1])), 0);
            ggml_tensor * qb = ggml_view_3d(gc, q, DH, NH, rem, q->nb[1], q->nb[2], (size_t) t0 * q->nb[2]);
            out = ggml_concat(gc, out, attend(qb, kb, mb, rem), 2);
        }
        // DS4_CHECK_HCAB=1: also run the blocked path, keep ITS output, and report per query row how far apart they are
        static const bool chk = [] { const char * e = std::getenv("DS4_CHECK_HCAB"); return e && std::atoi(e) != 0; }();
        if (chk) {
            ggml_tensor * ob = blocked();
            ggml_tensor * a2 = ggml_reshape_2d(gc, ggml_cont(gc, out), DH * NH, n);
            ggml_tensor * b2 = ggml_reshape_2d(gc, ggml_cont(gc, ob), DH * NH, n);
            im.dbg_hcab_d2 = ggml_sum_rows(gc, ggml_sqr(gc, ggml_sub(gc, a2, b2)));   // [1, n]
            im.dbg_hcab_r2 = ggml_sum_rows(gc, ggml_sqr(gc, b2));
            ggml_set_output(im.dbg_hcab_d2);
            ggml_set_output(im.dbg_hcab_r2);
            wb.push_back(im.dbg_hcab_d2);   // not ancestors of the layer's outputs: expanded into the graph with the write-backs
            wb.push_back(im.dbg_hcab_r2);
            im.dbg_hcab_il = il;
            out = ob;
        }
    } else if (nqb > 0) {
        const int64_t nk = cap + n_raw, T = (nk + 255) / 256 * 256;
        ggml_tensor * kall = ggml_concat(gc, k_cmp, k_raw, 2);                      // [DH, 1, nk]
        if (T > nk) {
            if (T > 65535) {   // ggml-cuda's pad kernel puts ne2 in gridDim.z (<= 65535): pad the flat buffer
                ggml_tensor * k1 = ggml_reshape_1d(gc, kall, ggml_nelements(kall));
                k1 = ggml_pad(gc, k1, (int) ((T - nk) * DH), 0, 0, 0);
                kall = ggml_reshape_3d(gc, k1, DH, 1, T);
            } else {
                kall = ggml_pad(gc, kall, 0, 0, (int) (T - nk), 0);
            }
        }
        ggml_tensor * kf = ggml_cast(gc, ggml_permute(gc, kall, 0, 2, 1, 3), GGML_TYPE_F16);   // [DH, T, 1]
        ggml_tensor * mr = m_raw;                                                   // [n_raw, n] F16
        if (T > nk) mr = ggml_concat(gc, m_raw, ggml_cast(gc, b.fill_f32({ T - nk, n }, NEG_INF), GGML_TYPE_F16), 0);
        ggml_tensor * mall = ggml_concat(gc, m_cmp, mr, 0);                         // [T, n] F16, rows = queries
        const float scale = 1.0f / std::sqrt((float) DH);
        ggml_tensor * sinks = b.BL(il, "attn_sinks.weight");
        auto batch = [&](int64_t t0, int64_t nq, int64_t nblk) {   // queries [t0, t0 + nq * nblk) as nblk sequences
            ggml_tensor * q4 = ggml_view_4d(gc, q, DH, NH, nq, nblk, q->nb[1], q->nb[2], (size_t) nq * q->nb[2],
                                            (size_t) t0 * q->nb[2]);
            ggml_tensor * qp = ggml_permute(gc, q4, 0, 2, 1, 3);                   // [DH, nq, NH, nblk]
            // one K for all sequences: ggml_view sizes a view as if contiguous (nblk copies would not fit in kf), so
            // the broadcast stride is set after - ggml-cuda reads an F16 K through its strides, nb[3] = 0 included
            ggml_tensor * k4 = ggml_view_4d(gc, kf, DH, T, 1, 1, kf->nb[1], kf->nb[2], kf->nb[2], 0);
            k4->ne[3] = nblk;
            k4->nb[3] = 0;
            const size_t row = mall->nb[1];
            ggml_tensor * m4 = ggml_view_4d(gc, mall, T, nq, 1, nblk, row, (size_t) nq * row, (size_t) nq * row,
                                            (size_t) t0 * row);                     // contiguous rows of the mask
            ggml_tensor * o = ggml_flash_attn_ext(gc, qp, k4, k4, m4, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(o, sinks);
            if (!dense1) ggml_flash_attn_ext_set_n_kv_max(o, (int32_t) nkv_q);
            return ggml_reshape_3d(gc, o, DH, NH, nq * nblk);                       // [DH, NH, nq * nblk]
        };
        const int64_t nblk = n / nqb, rem = n - nblk * nqb;
        out = batch(0, nqb, nblk);
        if (rem) out = ggml_concat(gc, out, batch(nblk * nqb, rem, 1), 2);
    } else if (RB >= n) {
        ggml_tensor * k_all = k_cmp ? ggml_concat(gc, k_raw, k_cmp, 2) : k_raw;
        ggml_tensor * mask  = m_cmp ? ggml_concat(gc, m_raw, m_cmp, 0) : m_raw;   // [n_raw + cap, n]
        out = attend(q, k_all, mask, n);
    } else {
        out = blocked();
    }

    // the gate's attn_raw tap: ds4_ref dumps the raw flash-attn output *before* the de-RoPE
    // (its attn_csa_lid / attn_hca probe is named at the flash-attn result, not after rope_back)
    // (decode with gate_taps only: it and the t_attn copy were ~3 nodes per layer every token for nobody)
    const bool taps = !im.pf && !L.host_attn_raw.empty();
    ggml_tensor * attn_raw = taps ? ggml_cont_2d(gc, out, NH * DH, n) : nullptr;

    // de-RoPE then grouped output LoRA
    out = ggml_reshape_3d(gc, out, DH, NH, n);
    out = b.rope_back(out, pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    out = ggml_reshape_3d(gc, out, OGD, OG, n);
    out = ggml_permute(gc, out, 0, 2, 1, 3);                                 // [OGD, n, OG]
    ggml_tensor * wo_a = b.BL(il, "attn_output_a.weight");
    if (ggml_n_dims(wo_a) == 2) wo_a = ggml_reshape_3d(gc, wo_a, OGD, OL, OG);
    ggml_tensor * oa = ggml_mul_mat(gc, wo_a, out);                          // [OL, n, OG]
    oa = ggml_permute(gc, oa, 0, 2, 1, 3);
    oa = flat2(oa, OL * OG, n);   // n == 1: the permute only moved a size-1 axis
    ggml_tensor * attn_out = ggml_mul_mat(gc, b.BL(il, "attn_output_b.weight"), oa);   // [D, n]

    // ---- feed-forward hyper-connection -------------------------------------------------
    ggml_tensor * hap = b.hc_post(attn_out, xin, post_a, comb_a);
    ggml_tensor * post_f = nullptr, * comb_f = nullptr;
    ggml_tensor * ffn_pre = b.hc_pre(hap, b.BL(il, "hc_ffn_fn.weight"), b.BL(il, "hc_ffn_scale.weight"),
                                     b.BL(il, "hc_ffn_base.weight"), &post_f, &comb_f);
    ggml_tensor * fn = b.rms_w(ffn_pre, b.BL(il, "ffn_norm.weight"));

    // ---- MoE router (ds4_ref main():904-928) -------------------------------------------
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(gc, ggml_softplus(gc, rlog));             // [NEXP, n]

    ggml_tensor * selected = nullptr;
    if (il < (int) im.g.hash_layer_count) {
        selected = ggml_get_rows(gc, b.BL(il, "ffn_gate_tid2eid.weight"), first(im.i_tid, n));
    } else {
        ggml_tensor * sel = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
        // cache-aware routing (set_route_bias): a selection-only bias like exp_probs_b; the mixing weights below still
        // come from the unbiased probs.  All zeros (the default) = the model's own routing, bit for bit.
        sel = ggml_add(gc, sel, ggml_reshape_2d(gc, L.rbias, im.NEXP, 1));
        selected = ggml_argsort_top_k(gc, sel, (int) im.NUSED);
    }
    ggml_set_output(selected);                                               // [NUSED, n]
    ggml_tensor * probs3 = ggml_reshape_3d(gc, probs, 1, im.NEXP, n);
    ggml_tensor * wts = ggml_get_rows(gc, probs3, selected);              // [1, topk, n]
    wts = ggml_reshape_2d(gc, wts, im.NUSED, n);
    ggml_tensor * wsum = ggml_add(gc, ggml_sum_rows(gc, wts), b.fill_f32({ 1, 1 }, 0.0f));
    wsum = ggml_clamp(gc, wsum, 6.103515625e-5f, INFINITY);
    wts = ggml_div(gc, wts, wsum);
    wts = ggml_reshape_3d(gc, wts, 1, im.NUSED, n);
    wts = ggml_scale(gc, wts, (float) im.g.weights_scale);
    ggml_set_output(wts);

    // ---- shared expert (ds4_ref main():946-956) ----------------------------------------
    ggml_tensor * up_s = ggml_mul_mat(gc, b.BL(il, "ffn_up_shexp.weight"), fn);
    ggml_tensor * gate_s = ggml_mul_mat(gc, b.BL(il, "ffn_gate_shexp.weight"), fn);
    const float lim = (float) im.g.swiglu_clamp_shexp[(size_t) il];
    up_s = ggml_clamp(gc, up_s, -lim, lim);
    gate_s = ggml_clamp(gc, gate_s, NEG_INF, lim);
    ggml_tensor * z = ggml_swiglu_split(gc, gate_s, up_s);
    ggml_tensor * shexp = ggml_mul_mat(gc, b.BL(il, "ffn_down_shexp.weight"), z);

    // ---- persisted hand-offs (finish_layer + the gate's taps), first n columns ----------
    auto cols2 = [&](ggml_tensor * t, int64_t n0) { return ggml_view_2d(gc, t, n0, n, t->nb[1], 0); };
    auto cols3 = [&](ggml_tensor * t, int64_t n0, int64_t n1) {
        return ggml_view_3d(gc, t, n0, n1, n, t->nb[1], t->nb[2], 0);
    };
    const size_t ob = (size_t) im.o_blk;
    ggml_tensor * c_fn  = ggml_cpy(gc, ggml_reshape_2d(gc, fn, D, n), ggml_view_2d(gc, L.o_f, D, n, ob, 0));
    ggml_tensor * c_hap = ggml_cpy(gc, ggml_reshape_3d(gc, hap, D, HC, n), cols3(L.hap, D, HC));
    ggml_tensor * c_pf  = ggml_cpy(gc, post_f, cols2(L.post_f, HC));   // [HC, n], a strided view of mixes
    ggml_tensor * c_cf  = ggml_cpy(gc, ggml_reshape_3d(gc, comb_f, HC, HC, n), cols3(L.comb_f, HC, HC));
    ggml_tensor * c_sh  = ggml_cpy(gc, ggml_reshape_2d(gc, shexp, D, n), cols2(L.shexp, D));
    ggml_tensor * c_ar  = taps ? ggml_cpy(gc, attn_raw, cols2(L.t_attn_raw, NH * DH)) : nullptr;
    ggml_tensor * c_at  = taps ? ggml_cpy(gc, ggml_reshape_2d(gc, attn_out, D, n), cols2(L.t_attn, D)) : nullptr;

    for (ggml_tensor * t : { c_fn, c_hap, c_pf, c_cf, c_sh }) ggml_build_forward_expand(gf, t);
    if (taps) for (ggml_tensor * t : { c_ar, c_at }) ggml_build_forward_expand(gf, t);   // gate taps (decode only)
    ggml_build_forward_expand(gf, selected);
    ggml_build_forward_expand(gf, wts);
    ggml_build_forward_expand(gf, ggml_cpy(gc, selected,   // [NUSED, n], a top-k view: strided for n > 1
                                           ggml_view_2d(gc, L.o_i, im.NUSED, n, ob, (size_t) im.o_ids_off)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_2d(gc, wts, im.NUSED, n),
                                           ggml_view_2d(gc, L.o_f, im.NUSED, n, ob, (size_t) im.o_wts_off)));

    for (ggml_tensor * t : wb) ggml_build_forward_expand(gf, t);   // chunk ring write-backs, after every read
    if (im.pf) return gf;   // a chunk graph: run once by the caller, not registered
    var.gf = gf;
    var.allo = im.make_allo();   // one arena per graph variant; see Impl::make_allo
    L.vars.push_back(var);
    return gf;
}

// finish_layer: ffn_out = routed_sum + shexp ; l_last = hc_post(ffn_out, hap, post_f, comb_f), per token
static ggml_cgraph * build_finish(Ds4Dense::Impl & im, int64_t n, Ds4Dense::Impl::Layer & L) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 512, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * shex2 = ggml_view_2d(gc, L.shexp, D, n, L.shexp->nb[1], 0);
    ggml_tensor * rs    = ggml_view_2d(gc, im.routed_sum, D, n, im.routed_sum->nb[1], 0);
    ggml_tensor * ffn_out = ggml_add(gc, shex2, rs);

    ggml_tensor * hap3 = ggml_view_3d(gc, L.hap, D, HC, n, L.hap->nb[1], L.hap->nb[2], 0);
    ggml_tensor * pf2  = ggml_view_2d(gc, L.post_f, HC, n, L.post_f->nb[1], 0);
    ggml_tensor * cf3  = ggml_view_3d(gc, L.comb_f, HC, HC, n, L.comb_f->nb[1], L.comb_f->nb[2], 0);
    ggml_tensor * llast = b.hc_post(ffn_out, hap3, pf2, cf3);
    ggml_tensor * ll3 = ggml_reshape_3d(gc, llast, D, HC, n);

    ggml_build_forward_expand(gf, ggml_cpy(gc, ll3, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1],
                                                                 im.x_state->nb[2], 0)));
    if (im.pf) return gf;   // a chunk graph: no l_last tap, not registered
    ggml_build_forward_expand(gf, ggml_cpy(gc, ll3, ggml_view_3d(gc, L.t_llast, D, HC, n, L.t_llast->nb[1],
                                                                 L.t_llast->nb[2], 0)));
    L.gf_finish[n] = gf;
    return gf;
}

// predict: the layer's router on rms_norm(hc_attn_pre) * ffn_norm.weight  (prefetch hint "A"), token 0 of the pass
static ggml_cgraph * build_predict(Ds4Dense::Impl & im, int il, Ds4Dense::Impl::Layer & L) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 512, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, 1, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * attn_pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                      b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * fnp = b.rms_w(attn_pre, b.BL(il, "ffn_norm.weight"));
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fnp);
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(gc, ggml_softplus(gc, rlog));

    ggml_tensor * out = nullptr;
    if (il < (int) im.g.hash_layer_count) {
        out = ggml_cast(gc, ggml_get_rows(gc, b.BL(il, "ffn_gate_tid2eid.weight"), ggml_view_1d(gc, im.i_tid, 1, 0)),
                        GGML_TYPE_F32);
    } else {
        out = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    L.predict_out = out;
    return gf;
}

// begin_tokens: hc_init = repeat(embd[tid], hc), per token
static ggml_cgraph * build_init(Ds4Dense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_view_3d(gc, im.i_emb, D, 1, n, im.i_emb->nb[1], im.i_emb->nb[1], 0);   // embd rows
    x = ggml_repeat_4d(gc, x, D, HC, n, 1);
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1],
                                                               im.x_state->nb[2], 0)));
    return gf;
}

// logits: hc_head -> output_norm -> output, per token.  `mtp`: the MTP head - its own hc_head (`mtp.output_hc_*`, from
// the MTP file: llama.cpp loads that file as its own model, so graph_mtp's hc_head_* are the file's) and
// shared_head_norm, the trunk's output (the file's is a BF16 copy of the original V4's; ours is 0731's).
// DS4_MTP_TRUNK_HC=1: the trunk's hc_head instead (the pre-2026-10-07 behaviour, for A/B and mutation checks)
// mab776 M7c: `arg_col` >= 0 = the MTP head on x_state column arg_col alone (n = 1), reduced to its argmax (I32 [1] in
// mtp_arg_t[arg_col]): a draft needs no [V] row on the host
static ggml_cgraph * build_head(Ds4Dense::Impl & im, int64_t n, bool mtp = false, int64_t arg_col = -1) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    const size_t x_off = im.pf ? (size_t) im.pf_row0 * im.x_state->nb[2]                 // chunk: rows row0..
                       : arg_col >= 0 ? (size_t) arg_col * im.x_state->nb[2] : 0;
    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], x_off);
    ggml_tensor * flat = ggml_reshape_2d(gc, xin, HC * D, n);
    ggml_tensor * flat_norm = ggml_rms_norm(gc, flat, (float) im.g.rms_eps);
    static const bool trunk_hc = [] { const char * e = std::getenv("DS4_MTP_TRUNK_HC"); return e && *e && *e != '0'; }();
    const std::string hp = mtp && !trunk_hc ? "mtp." : "";
    ggml_tensor * mixes = ggml_mul_mat(gc, im.w.get(hp + "output_hc_fn.weight"), flat_norm);
    ggml_tensor * scale = im.w.get(hp + "output_hc_scale.weight");
    ggml_tensor * base  = im.w.get(hp + "output_hc_base.weight");
    ggml_tensor * pre = ggml_sigmoid(gc, b.hc_affine(mixes, b.v1(scale, 1, 0), b.v1(base, HC, 0)));
    pre = ggml_scale_bias(gc, pre, 1.0f, (float) im.g.hc_eps);
    ggml_tensor * hc_head = b.hc_mean(xin, pre);
    ggml_tensor * rn = b.rms_w(hc_head, mtp ? b.BL((int) im.il_mtp, "nextn.shared_head_norm.weight")
                                            : im.w.get("output_norm.weight"));
    ggml_tensor * lg = ggml_mul_mat(gc, im.w.get("output.weight"), rn);
    if (arg_col >= 0) {
        ggml_tensor * am = ggml_argmax(gc, lg);
        ggml_set_output(am);
        ggml_build_forward_expand(gf, am);
        im.mtp_arg_t[arg_col] = am;
        return gf;
    }
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);
    if (im.pf) im.pf_logits = lg;
    else (mtp ? im.mtp_logits_t : im.logits_t)[n] = lg;
    return gf;
}

// MTP input: x_state <- eh_proj(concat(enorm(embd[next]) repeated over the hc streams, hnorm(x_state))), per token.
// x_state holds the trunk's final hc streams (after the last finish_layer); i_emb the next tokens' embeddings.
// mab776 M7c: `src` > 0 (n = 1, a chained draft) reads h from column src - the MTP block's own output for the
// previous draft - and writes column 0.
static ggml_cgraph * build_mtp_in(Ds4Dense::Impl & im, int64_t n, int64_t src = 0) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;
    const int il = (int) im.il_mtp;
    ggml_tensor * xs = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * xh = src == 0 ? xs : ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2],
                                                    (size_t) src * im.x_state->nb[2]);
    ggml_tensor * emb = ggml_view_2d(gc, im.i_emb, D, n, im.i_emb->nb[1], 0);
    ggml_tensor * en = b.rms_w(emb, b.BL(il, "nextn.enorm.weight"));
    en = ggml_repeat_4d(gc, ggml_reshape_3d(gc, en, D, 1, n), D, HC, n, 1);
    ggml_tensor * hn = b.rms_w(xh, b.BL(il, "nextn.hnorm.weight"));
    ggml_tensor * x = ggml_mul_mat(gc, b.BL(il, "nextn.eh_proj.weight"), ggml_concat(gc, en, hn, 0));   // [D, HC, n]
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, xs));
    return gf;
}

// ================================================================== public API

Ds4Dense::Ds4Dense() : p_(new Impl()) {}
Ds4Dense::~Ds4Dense() = default;

bool Ds4Dense::init(const std::string & model_path, const Ds4DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.path = model_path;
    im.n_threads = cfg.n_threads;
    im.pf_cap = std::max<int64_t>(0, cfg.prefill_chunk);

    im.backend = cfg.backend;
    if (!im.backend) {
        im.backend = ggml_backend_cpu_init();
        im.own_backend = true;
        if (!im.backend) { err = "cannot create a CPU backend"; return false; }
    }
    if (ggml_backend_is_cpu(im.backend)) ggml_backend_cpu_set_n_threads(im.backend, im.n_threads);
    // DS4_CPU_FA_REF=1: ggml-cpu's reference flash-attn for any query count (its tiled kernel takes >= 64 queries and
    // rounds differently) - the chunk gate uses it to show 64+-token chunks are the decode math exactly
    if (ggml_backend_is_cpu(im.backend) && std::getenv("DS4_CPU_FA_REF")) ggml_backend_cpu_set_use_ref(im.backend, true);

    im.model = std::make_unique<GgufModel>(GgufModel::open(model_path));
    if (im.model->size() == 0) { err = "cannot open " + model_path; return false; }
    if (!(err = ds4_read_geometry(*im.model, im.g)).empty()) return false;
    if (!(err = check_ds4_model(*im.model, im.g)).empty()) return false;
    std::map<std::string, uint32_t> want_type;
    for (const auto & sp : ds4_expected_tensors(im.g)) want_type[sp.name] = sp.type;
    if (!load_weights(*im.model, im.backend, im.w, cfg.skip_routed_experts,
                      ggml_backend_is_cpu(im.backend) ? -1 : cfg.requant_type, err, "", {}, cfg.f16_q8,
                      &want_type)) return false;
    im.n_trunk = im.g.n_layer();
    if (!cfg.mtp_path.empty()) {   // the MTP head: its own GGUF, block blk.<n_trunk>
        im.mtp_model = std::make_unique<GgufModel>(GgufModel::open(cfg.mtp_path));
        if (im.mtp_model->size() == 0) { err = "cannot open " + cfg.mtp_path; return false; }
        if (!(err = ds4_attach_mtp(*im.mtp_model, im.g, im.il_mtp)).empty()) return false;
        if (!load_weights(*im.mtp_model, im.backend, im.w, cfg.skip_routed_experts,
                          ggml_backend_is_cpu(im.backend) ? -1 : cfg.requant_type, err,
                          "blk." + std::to_string(im.il_mtp) + ".", ds4_mtp_head_alias())) return false;
    }
    {
        size_t sh = 0;
        const TensorInfo * te = im.model->find("token_embd.weight", &sh);
        if (!te) { err = "no token_embd.weight"; return false; }
        im.embd_data = im.model->shard(sh).tensor_data(*te);
        im.embd_type = (int) te->type;
        im.embd_row  = ggml_row_size((ggml_type) te->type, (int64_t) te->shape.at(0));
        if (!ggml_get_type_traits((ggml_type) te->type)->to_float && te->type != GGML_TYPE_F32) {
            err = "token_embd type has no to_float"; return false;
        }
    }

    const Ds4Geometry & g = im.g;
    im.D = g.n_embd; im.HC = g.hc; im.DH = g.d_head; im.DHR = g.d_rope; im.NH = g.n_head;
    im.OG = g.o_groups; im.OL = g.o_lora; im.OGD = g.o_group_dim(); im.NHPG = g.n_head / g.o_groups;
    im.RQ = g.r_q; im.SWA = g.n_swa; im.RAW = g.n_swa + kNtMax - 1;
    im.IDXH = g.indexer_n_head; im.IDXK = g.indexer_key_dim; im.IDXTOPK = g.indexer_top_k;
    im.NEXP = g.n_expert; im.NUSED = g.n_expert_used; im.NFF = g.n_ff; im.NSHEXP = g.n_expert_shared;

    const int64_t nl = im.n_trunk + (im.il_mtp >= 0 ? 1 : 0);   // + the MTP block
    if (nl <= 0 || im.D <= 0 || im.HC <= 0 || im.SWA <= 0) { err = "degenerate geometry"; return false; }
    if (im.NUSED <= 0) { err = "expert_used_count is zero"; return false; }

    bool any_csa = false;
    // ---- persistent decode state -------------------------------------------------------
    {
        ggml_init_params ip = { /*mem_size*/ 128ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.sctx = ggml_init(ip);
        if (!im.sctx) { err = "ggml_init(state) failed"; return false; }
        ggml_set_no_alloc(im.sctx, true);

        auto nt3 = [&](ggml_type ty, int64_t n0, int64_t n1, int64_t n2) { return ggml_new_tensor_3d(im.sctx, ty, n0, n1, n2); };
        auto nt2 = [&](ggml_type ty, int64_t n0, int64_t n1) { return ggml_new_tensor_2d(im.sctx, ty, n0, n1); };
        auto nt1 = [&](ggml_type ty, int64_t n0) { return ggml_new_tensor_1d(im.sctx, ty, n0); };
        ggml_init_params ipi = { /*mem_size*/ 16ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.ictx = ggml_init(ipi);
        if (!im.ictx) { err = "ggml_init(inputs) failed"; return false; }
        std::vector<ggml_tensor *> in_list;                    // placed back to back in ibuf
        auto in1 = [&](ggml_type ty, int64_t n0) {
            ggml_tensor * t = ggml_new_tensor_1d(im.ictx, ty, n0);
            in_list.push_back(t);
            return t;
        };
        auto on1 = [&](ggml_type ty, int64_t n0) { return ggml_new_tensor_1d(im.ictx, ty, n0); };
        const int64_t NT = kNtMax;
        // the compressed K cache: type, rotation, and (--comp-host) a separate context placed in pinned host RAM
        const ggml_type comp_ty = cfg.comp_type != 0 ? (ggml_type) cfg.comp_type
                                : cfg.comp_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        const bool comp_quant = comp_ty != GGML_TYPE_F32;
        im.comp_rot = cfg.comp_rot >= 0 ? cfg.comp_rot != 0
                    : comp_ty == GGML_TYPE_Q4_0 || comp_ty == GGML_TYPE_IQ4_NL || comp_ty == GGML_TYPE_Q5_0;
        ggml_backend_buffer_type_t comp_hbt = nullptr;
        if (cfg.comp_host && !ggml_backend_is_cpu(im.backend))
            comp_hbt = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(im.backend));
        if (comp_hbt) {
            ggml_init_params iph = { /*mem_size*/ 1ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
            im.hctx = ggml_init(iph);
            if (!im.hctx) { err = "ggml_init(comp host) failed"; return false; }
        }
        ggml_context * comp_ctx = im.hctx ? im.hctx : im.sctx;

        im.x_state    = nt3(GGML_TYPE_F32, im.D, im.HC, NT);
        im.routed_sum = nt2(GGML_TYPE_F32, im.D, NT);
        im.i_tid      = nt1(GGML_TYPE_I32, NT);
        im.i_emb      = nt2(GGML_TYPE_F32, im.D, NT);
        im.i_pos      = in1(GGML_TYPE_I32, NT);

        im.ly.resize((size_t) nl);
        for (int64_t il = 0; il < nl; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            L.ratio = (size_t) il < g.compress_ratios.size() ? g.compress_ratios[(size_t) il] : 0;
            L.csa = (L.ratio == 4);
            if (L.ratio != 0 && L.ratio != 4 && L.ratio != 128) { err = "unsupported compress ratio"; return false; }
            any_csa = any_csa || L.csa;
            L.ring = L.csa ? 2 * L.ratio : L.ratio;
            L.ring_sz = L.ring + kNtMax - 1;
            L.sdim = L.csa ? 2 * im.DH : im.DH;
            const int64_t by_ctx = L.ratio != 0 ? std::max<int64_t>(1, g.context_length / L.ratio) : 0;
            L.comp_max = cfg.comp_cap_max > 0 ? std::min<int64_t>(cfg.comp_cap_max, by_ctx) : by_ctx;

            L.raw     = nt2(GGML_TYPE_F32, im.DH, im.RAW);
            L.hap     = nt3(GGML_TYPE_F32, im.D, im.HC, NT);
            L.post_f  = nt2(GGML_TYPE_F32, im.HC, NT);
            L.comb_f  = nt3(GGML_TYPE_F32, im.HC, im.HC, NT);
            L.shexp   = nt2(GGML_TYPE_F32, im.D, NT);
            L.fn      = on1(GGML_TYPE_F32, im.D);
            L.o_ids   = on1(GGML_TYPE_I32, im.NUSED);
            L.o_wts   = on1(GGML_TYPE_F32, im.NUSED);
            L.t_attn_raw = nt2(GGML_TYPE_F32, im.NH * im.DH, NT);
            L.t_attn  = nt2(GGML_TYPE_F32, im.D, NT);
            L.t_llast = nt3(GGML_TYPE_F32, im.D, im.HC, NT);

            L.i_slot_raw  = in1(GGML_TYPE_I32, NT);
            L.i_idx_raw   = in1(GGML_TYPE_I32, im.SWA + NT - 1);
            L.i_mask_raw  = in1(GGML_TYPE_F16, (im.SWA + NT - 1) * NT);
            L.i_slot_comp = in1(GGML_TYPE_I32, NT);
            L.i_slot_state = in1(GGML_TYPE_I32, NT);
            L.i_comp_pos  = in1(GGML_TYPE_I32, NT);
            L.i_state_pos = in1(GGML_TYPE_I32, NT);
            L.i_idx_state = in1(GGML_TYPE_I32, std::max<int64_t>(1, L.ring) * NT);
            if (L.ratio != 0) {
                // DS4_VIS_DEV: a 16-byte per-layer count replaces the whole-capacity host mask in the span
                if (vis_dev_on()) L.i_nvis   = in1(GGML_TYPE_F32, NT);
                else              L.vis_full = in1(GGML_TYPE_F32, L.comp_max * NT);
            }
            L.rbias = in1(GGML_TYPE_F32, im.NEXP);

            if (L.ratio != 0) {
                // comp_q8 (s22 (b)): the compressed-row K cache as Q8_0 - ~0.65 GB per 32K tokens back for the
                // expert cache at 256K.  Written by ggml's quantized set_rows, read back dequantized (get_rows),
                // so it is lossy by construction; the default F32 path is untouched.
                // comp_type (s33/s34): Q8_0 / 4-bit / --comp-host - see Ds4DenseConfig
                L.comp  = ggml_new_tensor_2d(comp_ctx, comp_ty, im.DH, L.comp_max + 1);
                if (comp_quant) L.i_idx_all = nt1(GGML_TYPE_I32, L.comp_max + 1);
                L.st_kv = nt2(GGML_TYPE_F32, L.sdim, L.ring_sz);
                L.st_sc = nt2(GGML_TYPE_F32, L.sdim, L.ring_sz);
                if (L.csa) {
                    L.icomp  = nt2(cfg.icomp_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, im.IDXK, L.comp_max + 1);
                    L.ist_kv = nt2(GGML_TYPE_F32, 2 * im.IDXK, L.ring_sz);   // CSA: ring == 2 * ratio
                    L.ist_sc = nt2(GGML_TYPE_F32, 2 * im.IDXK, L.ring_sz);
                }
            }
            // every layer (ratio-0 sliding-window layers too): ffn_norm is the expert tier's input
            L.host_fn.resize((size_t) (im.D * NT));
            if (cfg.gate_taps) {   // the gate's taps: ~10 MB of device->host copies per token, off by default
                L.host_attn_raw.resize((size_t) (im.NH * im.DH * NT));
                L.host_attn.resize((size_t) (im.D * NT));
                L.host_llast.resize((size_t) (im.D * im.HC * NT));
            }
        }
        if (any_csa) im.rot = nt2(GGML_TYPE_F32, im.IDXK, im.IDXK);
        {
            int64_t cm = 0;
            for (const Impl::Layer & L : im.ly) cm = std::max<int64_t>(cm, L.ratio ? L.comp_max : 0);
            if (cm > 0) im.neg_ramp = ggml_new_tensor_1d(im.sctx, GGML_TYPE_F32, cm);
        }
        if (im.comp_rot && (!im.rot || im.DH % im.IDXK != 0)) { err = "--comp-rot needs the indexer rotation"; return false; }
        if (im.hctx) {
            im.hbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.hctx, comp_hbt);
            if (!im.hbuf) { err = "cannot allocate the compressed K caches in pinned host RAM"; return false; }
            ggml_backend_buffer_clear(im.hbuf, 0);
        }

        im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!im.sbuf) { err = "cannot allocate the persistent decode state"; return false; }
        ggml_backend_buffer_clear(im.sbuf, 0);
        for (Impl::Layer & L : im.ly) {   // comp_q8: the ascending row indices the masked path gathers with (once)
            if (!L.i_idx_all) continue;
            std::vector<int32_t> ix((size_t) (L.comp_max + 1));
            for (int64_t i = 0; i <= L.comp_max; ++i) ix[(size_t) i] = (int32_t) i;
            ggml_backend_tensor_set(L.i_idx_all, ix.data(), 0, ix.size() * 4);
        }

        // ---- the input span and the per-layer output blocks: tensors placed at fixed offsets ----
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(im.backend);
        const size_t al = std::max<size_t>(64, ggml_backend_buft_get_alignment(buft));
        auto up = [&](size_t n) { return (n + al - 1) / al * al; };
        size_t in_bytes = 0;
        for (ggml_tensor * t : in_list) in_bytes += up(ggml_nbytes(t));
        im.ibuf = ggml_backend_buft_alloc_buffer(buft, in_bytes + al);
        if (!im.ibuf) { err = "cannot allocate the input span"; return false; }
        im.in_base = (uint8_t *) ggml_backend_buffer_get_base(im.ibuf);
        {
            size_t off = 0;
            for (ggml_tensor * t : in_list) {
                if (ggml_backend_tensor_alloc(im.ibuf, t, im.in_base + off) != GGML_STATUS_SUCCESS) {
                    err = "cannot place an input tensor"; return false;
                }
                off += up(ggml_nbytes(t));
            }
            im.i_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) in_bytes);
            if (ggml_backend_tensor_alloc(im.ibuf, im.i_span, im.in_base) != GGML_STATUS_SUCCESS) {
                err = "cannot place the input span"; return false;
            }
            im.in_host.assign(in_bytes, 0);
        }
        // router outputs: per layer kNtMax token blocks [fn | ids | wts] of o_blk bytes; a pass of n tokens reads back
        // the first n blocks of its layer in one tensor_get
        const size_t o1 = up(ggml_row_size(GGML_TYPE_F32, im.D)), o2 = o1 + up(4 * im.NUSED);
        const size_t blk = o2 + up(4 * im.NUSED);
        im.o_blk = (int64_t) blk; im.o_ids_off = (int64_t) o1; im.o_wts_off = (int64_t) o2;
        const size_t lblk = blk * (size_t) NT;
        im.obuf = ggml_backend_buft_alloc_buffer(buft, lblk * (size_t) nl + al);
        if (!im.obuf) { err = "cannot allocate the router output blocks"; return false; }
        {
            uint8_t * ob = (uint8_t *) ggml_backend_buffer_get_base(im.obuf);
            for (int64_t il = 0; il < nl; ++il) {
                Impl::Layer & L = im.ly[(size_t) il];
                uint8_t * b0 = ob + (size_t) il * lblk;
                L.o_f    = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, (int64_t) (lblk / 4));
                L.o_i    = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, (int64_t) (lblk / 4));
                L.o_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) lblk);
                for (auto [t, o] : { std::pair<ggml_tensor *, size_t>{ L.fn, 0 }, { L.o_ids, o1 }, { L.o_wts, o2 },
                                     { L.o_f, 0 }, { L.o_i, 0 }, { L.o_span, 0 } })
                    if (ggml_backend_tensor_alloc(im.obuf, t, b0 + o) != GGML_STATUS_SUCCESS) {
                        err = "cannot place a router output block"; return false;
                    }
                L.host_out.assign(lblk, 0);
            }
        }

        // compressor score rings start at SENTINEL: a token that was never written gets weight 0, exactly
        // like ds4_ref's zero/-inf row appended for the first block's missing previous half
        std::vector<float> sent;
        for (Impl::Layer & L : im.ly) {
            if (L.ratio == 0) continue;
            sent.assign((size_t) (L.sdim * L.ring_sz), SENTINEL);
            ggml_backend_tensor_set(L.st_sc, sent.data(), 0, sent.size() * 4);
            if (L.csa) {
                sent.assign((size_t) (2 * im.IDXK * L.ring_sz), SENTINEL);
                ggml_backend_tensor_set(L.ist_sc, sent.data(), 0, sent.size() * 4);
            }
        }
        if (im.neg_ramp) {
            std::vector<float> r((size_t) im.neg_ramp->ne[0]);
            for (size_t i = 0; i < r.size(); ++i) r[i] = -(float) i;
            ggml_backend_tensor_set(im.neg_ramp, r.data(), 0, r.size() * 4);
        }
        if (im.rot) {
            const int n = (int) im.IDXK;
            std::vector<float> rd((size_t) n * n, 0.0f);
            rd[0] = 1.0f / std::sqrt((float) n);
            for (int s = 1; s < n; s *= 2)
                for (int i = 0; i < s; ++i)
                    for (int j = 0; j < s; ++j) {
                        const float v = rd[(size_t) i * n + j];
                        rd[(size_t) (i + s) * n + j] = v;
                        rd[(size_t) i * n + (j + s)] = v;
                        rd[(size_t) (i + s) * n + (j + s)] = -v;
                    }
            ggml_backend_tensor_set(im.rot, rd.data(), 0, rd.size() * 4);
        }
        // mab776 (M5): the ring snapshots, here with the state rather than at the first snapshot(): a caller sizing its
        // expert cache from the free VRAM after init() has then paid for them (strata-glm allocated its snapshot lazily
        // and it failed under VRAM pressure once the cache had the card, FINDINGS s18)
        if (cfg.snapshot_slots > 0 && !im.snap_alloc(cfg.snapshot_slots)) { err = im.err; return false; }
    }

    // ---- graph scaffolding -------------------------------------------------------------
    {
        // node structs of every graph variant (no tensor data): 256 MiB overflowed once --verify's 2-token variants
        // joined the compressed-row-skip ones (2026-10-08); host pages are only touched as graphs are built
        ggml_init_params ip = { /*mem_size*/ 1024ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.gctx = ggml_init(ip);
        if (!im.gctx) { err = "ggml_init(graph) failed"; return false; }
        ggml_set_no_alloc(im.gctx, true);

        if (!im.ensure_n(1)) { err = im.err; return false; }
        for (int64_t il = 0; il < nl; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            L.gf_predict = build_predict(im, (int) il, L);
            L.allo_predict = im.make_allo();
            if (!L.allo_predict) { err = "ggml_gallocr_new failed"; return false; }
            build_attn(im, (int) il, 0, 1, L);   // cap 0; larger capacities (and n > 1) are built on demand
        }
    }
    im.host_logits.resize((size_t) (g.vocab_size * kNtMax));
    return true;
}

// init / head / finish graphs for passes of n tokens, built on first use
bool Ds4Dense::Impl::ensure_n(int64_t n) {
    if (n < 1 || n > kNtMax) { err = "pass size out of range (1.." + std::to_string(kNtMax) + ")"; return false; }
    if (!gf_init[n]) {
        gf_init[n] = build_init(*this, n);
        allo_init[n] = make_allo();
        gf_head[n] = build_head(*this, n);
        allo_head[n] = make_allo();
        if (!allo_init[n] || !allo_head[n]) { err = "ggml_gallocr_new failed"; return false; }
    }
    if (il_mtp >= 0 && !gf_mtp_in[n]) {
        gf_mtp_in[n] = build_mtp_in(*this, n);
        allo_mtp_in[n] = make_allo();
        gf_mtp_head[n] = build_head(*this, n, true);
        allo_mtp_head[n] = make_allo();
        if (!allo_mtp_in[n] || !allo_mtp_head[n]) { err = "ggml_gallocr_new failed"; return false; }
    }
    for (Layer & L : ly) {
        if (L.gf_finish[n]) continue;
        build_finish(*this, n, L);
        L.allo_finish[n] = make_allo();
        if (!L.allo_finish[n]) { err = "ggml_gallocr_new failed"; return false; }
    }
    return true;
}

// the attention variant for (cap, n), built on first use
static Ds4Dense::Impl::Var * attn_var(Ds4Dense::Impl & im, int il, int64_t cap, int64_t n, bool emit = true) {
    Ds4Dense::Impl::Layer & L = im.ly[(size_t) il];
    emit = emit || n != 1 || cap == 0;
    for (Ds4Dense::Impl::Var & v : L.vars)
        if (v.cap == cap && v.n == n && v.emit == emit) return &v;
    build_attn(im, il, cap, n, L, emit);
    return &L.vars.back();
}

// DS4_SPARSE (default on): one-token CSA attention gathers the indexer's top rows instead of masking all blocks
// mab776 M7b, DS4_SPARSE_N (default on): a multi-token decode pass (speculative verify) gathers each query's top-k
// compressed rows as one-token decode does, instead of a dense masked flash-attn over every row (build_attn).
static bool sparse_n_off() {
    static const bool off = [] { const char * e = std::getenv("DS4_SPARSE_N"); return e && std::atoi(e) == 0; }();
    return off;
}

static bool sparse_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_SPARSE"); return !e || std::atoi(e) != 0; }();
    return on;
}

// mab776 M10, DS4_PF_ARGSORT (default off): a prompt chunk's indexer top-k is ggml_top_k - ggml-cuda's TOP_K is one
// radix-select launch for every row with tools/ds4/patches/ggml-topk-radix-select.patch.  =1: the full-row argsort it
// replaced (one batched cub sort of [cap, n] plus that whole index buffer), for an A/B.
static bool pf_argsort_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_ARGSORT"); return e && std::atoi(e) != 0; }();
    return on;
}

// mab776 M10, DS4_PF_LEAN_MASK (default on): a prompt chunk's CSA layers stop materialising the [cap, n] visibility
// mask (4 full F32 passes - repeat, add, step, log - plus an F16 cast for the indexer and an add + cast for the
// attention mask: ~6.6 GB of traffic per layer-chunk at 250K).  The fused indexer takes the per-query counts
// (ggml-cuda li_mma: key kv visible iff kv < i_nvis[t], the same 0 / -inf), and once every query of the chunk sees
// >= top-k blocks the top-k only picks visible rows, so the attention mask is the F16 -inf fill with zeros at the
// selected rows - no visibility add, no cast.  Same values, so the same bits.  =0: the dense masks (A/B).
static bool pf_lean_mask_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_LEAN_MASK"); return !e || std::atoi(e) != 0; }();
    return on;
}
// its two halves apart (diagnostics): DS4_PF_LEAN_IDX=0 keeps the dense indexer mask, DS4_PF_LEAN_FA=0 the attention one
static bool pf_lean_idx_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_LEAN_IDX"); return !e || std::atoi(e) != 0; }();
    return on && pf_lean_mask_on();
}
static bool pf_lean_fa_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_LEAN_FA"); return !e || std::atoi(e) != 0; }();
    return on && pf_lean_mask_on();
}

// mab776 M10, DS4_PF_HCA1 (default OFF): an HCA layer's prompt chunk takes the CSA chunks' one-call layout (K =
// [compressed rows | raw union window | pad] cast to F16 once, ONE flash-attn over every query, ggml-cuda's tile skip
// ending each query tile at its window) instead of 32 query blocks that each re-concatenate, cast and pad the
// compressed K and the mask (nsys @250k: ~60 concats + 32 casts + 32 pads + 32 small flash-attns per layer-chunk).  Same
// keys and mask, only the softmax's summation order differs - yet ppl768 (real text, one chunk, deterministic) moves
// 9.9034 -> 9.9512 (+0.48 %) and the 8k filler prompt's NLL +0.0005 consistently, so it stays opt-in (=1) until the
// cause is known (a precision difference of the long masked flash-attn?).  Prefill +~3 % at 250k with it.
static bool pf_hca1_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_HCA1"); return e && std::atoi(e) != 0; }();
    return on;
}

// mab776 M10d, DS4_PF_EMIT (default ON): prompt-chunk compressors compute only the rows that land in a slot (build_attn,
// prefill_begin).  =0: one candidate row per token, the non-final ones written to the dummy slot comp_max (as before).
static bool pf_emit_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_EMIT"); return !e || std::atoi(e) != 0; }();
    return on;
}

// mab776 M10c, DS4_PF_HCAB (default ON): an HCA layer's prompt chunk runs its blocked layout (blocks of RB queries,
// each over its own window + the compressed rows) as the sequences of ONE flash-attn call (build_attn).
static bool pf_hcab_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_HCAB"); return !e || std::atoi(e) != 0; }();
    return on;
}

// DS4_COMP_SKIP (default on): a one-token pass that completes no compressed block skips the pooled row (FINDINGS s21)
static bool comp_skip_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_COMP_SKIP"); return !e || std::atoi(e) != 0; }();
    return on;
}

// DS4_VIS_DEV (default on): the compressed-row visibility mask is built in the graph from the per-layer count
// `i_nvis` instead of a host mask that rides the per-token input span (FINDINGS s22).  DS4_VIS_DEV=0 restores the
// host path (and its span cost) for an A/B - both must produce the same bits on the CPU gate.
static bool vis_dev_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_VIS_DEV"); return !e || std::atoi(e) != 0; }();
    return on;
}

// DS4_IDX_BLOCK (default 64, 0 = all queries at once): how many queries the lightning indexer's [cap, query, 64]
// score tensor is computed for at a time - the tensor that otherwise blocks every long prompt (FINDINGS s24).
// Queries are independent here, so the value is the same; only n > block changes the graph at all.
// DS4_FUSED_IDX (default 1 off-CPU, 0 on the CPU; set it to force either way): ggml_lightning_indexer
static bool fused_idx_on(bool cpu) {
    static const int v = [] { const char * e = std::getenv("DS4_FUSED_IDX"); return e ? std::atoi(e) : -1; }();
    return v < 0 ? !cpu : v != 0;
}

// DS4_OVL_GATHER (default on): the CSA overlap compressor gathers its prev/cur state halves with one get_rows over
// half-rows (the host writes 2*slot / 2*slot+1 into i_idx_state for CSA layers) instead of whole rows + 4 strided
// conts + 2 concats.  DS4_OVL_GATHER=0 restores the old graph and host indices for an A/B (same bits).
static bool ovl_gather_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_OVL_GATHER"); return !e || std::atoi(e) != 0; }();
    return on;
}

// DS4_RAW_BLOCK (default 64, 0 = all queries in one flash-attn): how many queries attend together over their union raw
// window (build_attn's attention section).  n <= block builds the single-call graph.
static int64_t raw_block_n(int64_t n) {
    static const int64_t blk = [] { const char * e = std::getenv("DS4_RAW_BLOCK"); return e ? (int64_t) std::atoi(e) : (int64_t) 64; }();
    return blk > 0 ? blk : n;
}

// mab776 DS4_PF_SPARSE (default on): prefill CSA attention at depth as one batched sparse flash-attn (build_attn)
static bool pf_sparse_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_SPARSE"); return !e || std::atoi(e) != 0; }();
    return on;
}

static int64_t idx_block_n(int64_t n) {
    static const int64_t blk = [] { const char * e = std::getenv("DS4_IDX_BLOCK"); return e ? (int64_t) std::atoi(e) : (int64_t) 64; }();
    return blk > 0 && blk < n ? blk : n;
}

// compressed-row capacity a pass whose last token sits at `pos_last` needs (0 for a ratio-0 layer)
static int64_t attn_cap(const Ds4Dense::Impl::Layer & L, int64_t pos_last) {
    if (L.ratio == 0) return 0;
    return std::min<int64_t>(L.comp_max, next_pow2(pos_last / L.ratio + 1));
}

const strata::Ds4Geometry & Ds4Dense::geom()    const { return p_->g; }
int64_t                     Ds4Dense::n_layer() const { return p_->n_trunk; }
int                         Ds4Dense::mtp_layer() const { return (int) p_->il_mtp; }
int                         Ds4Dense::max_tokens() const { return (int) kNtMax; }
ggml_backend_t              Ds4Dense::backend() const { return p_->backend; }
const std::string &         Ds4Dense::last_error() const { return p_->err; }
ggml_tensor *               Ds4Dense::weight(const std::string & name) const { return p_->w.get(name); }

bool Ds4Dense::begin_token(int tid) { return begin_tokens(&tid, 1); }

bool Ds4Dense::begin_tokens(const int * tids, int n) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_init[n], im.gf_init[n])) { im.err = "gallocr(init) failed"; return false; }
    std::vector<int32_t> t32((size_t) n);
    // embd[tid] on the host, exactly what get_rows + cast computes (F16 -> F32 is exact; quantized rows dequantize
    // with the same to_float)
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        const int tid = tids[t];
        if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
        t32[(size_t) t] = tid;
        const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(im.i_tid, t32.data(), 0, t32.size() * 4);
    im.cur_tid = n == 1 ? tids[0] : -1;
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (ggml_backend_graph_compute(im.backend, im.gf_init[n]) != GGML_STATUS_SUCCESS) {
        im.err = "init graph compute failed";
        return false;
    }
    return true;
}

static bool pf_hash_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_PF_HASH"); return !e || std::atoi(e) != 0; }();
    return on;
}

// predict's host half: the hash layers' ids, else the top n_top of the router scores (ties keep index order as
// std::partial_sort leaves them - the same pick whichever way the readback came)
static void predict_pick(Ds4Dense::Impl & im, int il, const void * raw, int * top_ids, int n_top) {
    ggml_tensor * out = im.ly[(size_t) il].predict_out;
    const size_t ne = (size_t) ggml_nelements(out);
    if (il < (int) im.g.hash_layer_count) {
        // build_predict casts tid2eid's I32 rows to F32.  mab776 M12a (DS4_PF_HASH, default on): read them as the
        // floats they are - token 0's exact experts, so the hash layers get a hint that is always right.
        // DS4_PF_HASH=0: the old int32 read (float bit patterns, which prefetch() dropped as out of range: no hint).
        const int n = std::min<int>(n_top, (int) ne);
        if (pf_hash_on() && out->type == GGML_TYPE_F32) {
            std::vector<float> f(ne);
            std::memcpy(f.data(), raw, ne * 4);
            for (int i = 0; i < n; ++i) top_ids[i] = (int) f[(size_t) i];
        } else {
            std::vector<int32_t> ids(ne);
            std::memcpy(ids.data(), raw, ne * 4);
            for (int i = 0; i < n; ++i) top_ids[i] = ids[(size_t) i];
        }
        return;
    }
    std::vector<float> sel(ne);
    std::memcpy(sel.data(), raw, ne * 4);
    std::vector<int> idx(ne);
    for (size_t i = 0; i < ne; ++i) idx[i] = (int) i;
    const int n = std::min<int>(n_top, (int) ne);
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    for (int i = 0; i < n; ++i) top_ids[i] = idx[(size_t) i];
}

bool Ds4Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || n_top <= 0) { im.err = "predict: bad layer/args"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    for (int i = 0; i < n_top; ++i) top_ids[i] = -1;

    if (!alloc_graph(L.allo_predict, L.gf_predict, im.backend)) { im.err = "gallocr(predict) failed"; return false; }
    if (!compute_dec(im.backend, L.gf_predict)) {
        im.err = "predict graph compute failed";
        return false;
    }
    ggml_tensor * out = L.predict_out;
    std::vector<uint8_t> raw(ggml_nbytes(out));
    get_dec(im.backend, out, raw.data(), raw.size());
    predict_pick(im, il, raw.data(), top_ids, n_top);
    return true;
}

bool Ds4Dense::predict_launch(int il) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ovl_setup()) return false;
    Impl::Layer & L = im.ly[(size_t) il];
    if (!L.predict_out || ggml_nbytes(L.predict_out) > Impl::kPredHost) return false;
    if (!alloc_graph(L.allo_predict, L.gf_predict, im.backend)) { im.err = "gallocr(predict) failed"; return false; }
    if (!compute_dec(im.backend, L.gf_predict)) {
        im.err = "predict graph compute failed";
        return false;
    }
    ggml_backend_tensor_get_async(im.backend, L.predict_out, im.pred_host, 0, ggml_nbytes(L.predict_out));
    ggml_backend_event_record(im.ev_pred, im.backend);
    im.pred_il = il;
    return true;
}

bool Ds4Dense::predict_collect(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (il != im.pred_il || n_top <= 0) return false;
    im.pred_il = -1;
    for (int i = 0; i < n_top; ++i) top_ids[i] = -1;
    ggml_backend_event_synchronize(im.ev_pred);   // the readback alone: the attention queued behind it keeps running
    predict_pick(im, il, im.pred_host, top_ids, n_top);
    return true;
}

bool Ds4Dense::attn_router(int il, int pos, int tid, int * routed_ids, float * routed_w,
                           const float ** ffn_norm_host, ggml_tensor ** ffn_norm_dev) {
    Impl & im = *p_;
    if (tid >= 0 && tid != im.cur_tid) {
        const int32_t t32 = tid;
        ggml_backend_tensor_set(im.i_tid, &t32, 0, sizeof t32);
        im.cur_tid = tid;
    }
    if (!attn_router_n(il, pos, 1, routed_ids, routed_w, ffn_norm_host)) return false;
    if (ffn_norm_dev) *ffn_norm_dev = im.ly[(size_t) il].fn;
    return true;
}

bool Ds4Dense::attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w,
                             const float ** ffn_norm_host, const std::function<void()> * queued) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "attn_router: bad layer"; return false; }
    if (n < 1 || n > kNtMax) { im.err = "attn_router: pass size out of range"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    if (pos0 < 0) { im.err = "attn_router: negative position"; return false; }
    const int64_t pos_last = (int64_t) pos0 + n - 1;

    // pick the graph variant for the compressed-row capacity the pass needs, and allocate its arena
    if (L.ratio != 0 && pos_last / L.ratio >= L.comp_max) {
        im.err = "attn_router: position beyond the compressed cache capacity";
        return false;
    }
    const bool emit = !(comp_skip_on() && n == 1 && L.ratio != 0 && (pos_last + 1) % L.ratio != 0);
    Impl::Var * var = attn_var(im, il, attn_cap(L, pos_last), n, emit);
    if (!alloc_graph(var->allo, var->gf, im.backend)) { im.err = "gallocr(attn) failed"; return false; }

    // ---- per-step inputs: every layer's, for this pass, in ONE upload per pass ------------------
    if (pos0 != im.in_pos || n != im.in_n) {
        auto put = [&](ggml_tensor * t, const void * src, size_t nbytes) {
            std::memcpy(im.in_host.data() + ((const uint8_t *) t->data - im.in_base), src, nbytes);
        };
        std::vector<int32_t> p32((size_t) n), s32((size_t) n);
        for (int t = 0; t < n; ++t) {
            p32[(size_t) t] = pos0 + t;
            s32[(size_t) t] = (int32_t) ((pos0 + t) % im.RAW);
        }
        put(im.i_pos, p32.data(), p32.size() * 4);
        // raw keys: the union of the n windows, oldest -> newest; query t sees [pos_t - SWA + 1, pos_t] and tok >= 0
        const int64_t n_raw = im.SWA + n - 1;
        std::vector<int32_t> idx((size_t) n_raw);
        std::vector<ggml_fp16_t> msk((size_t) (n_raw * n));
        const ggml_fp16_t h0 = ggml_fp32_to_fp16(0.0f), hinf = ggml_fp32_to_fp16(NEG_INF);
        for (int64_t i = 0; i < n_raw; ++i) {
            const int64_t tok = (int64_t) pos0 - im.SWA + 1 + i;
            idx[(size_t) i] = (int32_t) (((tok % im.RAW) + im.RAW) % im.RAW);
            for (int t = 0; t < n; ++t) {
                const int64_t pt = (int64_t) pos0 + t;
                msk[(size_t) (t * n_raw + i)] = tok >= 0 && tok <= pt && tok > pt - im.SWA ? h0 : hinf;
            }
        }
        const bool vdev = vis_dev_on();
        std::vector<int32_t> sp(n), cpp(n), sc(n), ss(n), sidx;
        std::vector<float> nvis;
        std::vector<float> vis;
        for (Impl::Layer & Ly : im.ly) {
            put(Ly.i_slot_raw, s32.data(), s32.size() * 4);
            put(Ly.i_idx_raw, idx.data(), idx.size() * 4);
            put(Ly.i_mask_raw, msk.data(), msk.size() * 2);
            if (Ly.ratio == 0) continue;
            if (pos_last / Ly.ratio >= Ly.comp_max) continue;   // attn_router refuses this layer above
            const int64_t cp_ = attn_cap(Ly, pos_last);
            sidx.resize((size_t) (Ly.ring * n));
            if (vdev) nvis.resize((size_t) n);
            else      vis.assign((size_t) (cp_ * n), NEG_INF);
            for (int t = 0; t < n; ++t) {
                const int64_t pt = (int64_t) pos0 + t, bl = pt / Ly.ratio;
                sp[(size_t) t]  = (int32_t) (pt % Ly.ratio);      // compressor `ape` row
                cpp[(size_t) t] = (int32_t) (Ly.ratio * bl);      // block start, for the compressed-key RoPE
                // block slot in the compressed caches; a later token of the pass in the same block owns the row,
                // this one writes its partial row to the spare slot
                const bool later_same = t + 1 < n && (pt + 1) / Ly.ratio == bl;
                sc[(size_t) t]  = (int32_t) (later_same ? Ly.comp_max : bl);
                ss[(size_t) t]  = (int32_t) (pt % Ly.ring_sz);    // token slot in the compressor state ring
                const bool half = Ly.csa && ovl_gather_on();   // DS4_OVL_GATHER: half-row indices
                for (int64_t i = 0; i < Ly.ring; ++i) {
                    const int64_t tok = pt - Ly.ring + 1 + i;
                    const int32_t slot = (int32_t) (((tok % Ly.ring_sz) + Ly.ring_sz) % Ly.ring_sz);
                    sidx[(size_t) (t * Ly.ring + i)] = half ? 2 * slot + (i >= Ly.ratio ? 1 : 0) : slot;
                }
                const int64_t nv = (pt + 1) / Ly.ratio;           // complete blocks query t may see
                if (vdev) nvis[(size_t) t] = (float) nv;
                else for (int64_t i = 0; i < nv && i < cp_; ++i) vis[(size_t) (t * cp_ + i)] = 0.0f;
            }
            put(Ly.i_state_pos, sp.data(), (size_t) n * 4);
            put(Ly.i_comp_pos, cpp.data(), (size_t) n * 4);
            put(Ly.i_slot_comp, sc.data(), (size_t) n * 4);
            put(Ly.i_slot_state, ss.data(), (size_t) n * 4);
            put(Ly.i_idx_state, sidx.data(), sidx.size() * 4);
            if (vdev) put(Ly.i_nvis, nvis.data(), (size_t) n * 4);
            else      put(Ly.vis_full, vis.data(), vis.size() * 4);
        }
        ggml_backend_tensor_set(im.i_span, im.in_host.data(), 0, im.in_host.size());
        im.in_pos = pos0;
        im.in_n = n;
    }

    // DS4_DBG_GRAPH=1: each attention graph variant's node count and op histogram, once (A/B of graph shapes)
    static const bool dbg_graph = std::getenv("DS4_DBG_GRAPH") != nullptr;
    if (dbg_graph) {
        static std::map<const void *, int> seen;
        if (!seen.count(var->gf)) {
            seen[var->gf] = 1;
            std::map<std::string, int> hist;
            const int nn = ggml_graph_n_nodes(var->gf);
            for (int i = 0; i < nn; ++i) {
                ggml_tensor * t = ggml_graph_node(var->gf, i);
                std::string k = ggml_op_desc(t);
                hist[k]++;
            }
            std::fprintf(stderr, "[graph] attn layer %d pos %lld: %d nodes |", il, (long long) pos0, nn);
            for (auto & kv : hist) std::fprintf(stderr, " %s:%d", kv.first.c_str(), kv.second);
            std::fprintf(stderr, "\n");
        }
    }
    if (!compute_dec(im.backend, var->gf)) {
        im.err = "attn graph compute failed";
        return false;
    }
    if (queued && *queued) (*queued)();   // mab776 M11b: e.g. predict_collect + the prefetch, under the attention

    static const bool dbg_idx = std::getenv("DS4_DBG_INDEXER") != nullptr;
    if (async_on() && (dbg_idx || !L.host_attn_raw.empty() || !L.host_attn.empty()))
        ggml_backend_synchronize(im.backend);   // the diagnostic readbacks below are plain (another stream)
    if (dbg_idx && var->dbg_isc) {
        std::vector<float> v((size_t) (var->cap * n));
        ggml_backend_tensor_get(var->dbg_isc, v.data(), 0, v.size() * 4);
        for (int t = 0; t < n; ++t) {
            std::fprintf(stderr, "[idx] L%d pos %d:", il, pos0 + t);
            for (int64_t i = 0; i < var->cap; ++i)
                if (std::isfinite(v[(size_t) (t * var->cap + i)])) std::fprintf(stderr, " %.6g", v[(size_t) (t * var->cap + i)]);
            std::fprintf(stderr, "\n");
        }
    }
    // ---- router outputs + hand-offs: n token blocks [fn | ids | wts], one readback ----------------
    {
        const size_t ob = (size_t) im.o_blk;
        get_dec(im.backend, L.o_span, L.host_out.data(), ob * (size_t) n);
        for (int t = 0; t < n; ++t) {
            const uint8_t * b0 = L.host_out.data() + ob * (size_t) t;
            const int32_t * ids = (const int32_t *) (b0 + im.o_ids_off);
            const float * wv = (const float *) (b0 + im.o_wts_off);
            for (int64_t i = 0; i < im.NUSED; ++i) {
                routed_ids[t * im.NUSED + i] = ids[i];
                routed_w[t * im.NUSED + i] = wv[i];
            }
            std::memcpy(L.host_fn.data() + (size_t) t * im.D, b0, (size_t) im.D * 4);
        }
    }
    if (!L.host_attn_raw.empty())
        ggml_backend_tensor_get(L.t_attn_raw, L.host_attn_raw.data(), 0, (size_t) (im.NH * im.DH * n) * 4);
    if (!L.host_attn.empty()) ggml_backend_tensor_get(L.t_attn, L.host_attn.data(), 0, (size_t) (im.D * n) * 4);

    if (ffn_norm_host) *ffn_norm_host = L.host_fn.data();
    L.tap_n = n;
    L.have_attn = true;
    return true;
}

bool Ds4Dense::reserve_graphs(int n_max) {
    Impl & im = *p_;
    n_max = std::max(1, std::min<int>(n_max, (int) kNtMax));
    for (int n = 1; n <= n_max; ++n)
        if (!im.ensure_n(n)) return false;
    for (int il = 0; il < (int) im.ly.size(); ++il) {
        Impl::Layer & L = im.ly[(size_t) il];
        std::vector<int64_t> caps;
        if (L.ratio == 0) caps.push_back(0);
        else {
            for (int64_t c = 1; c < L.comp_max; c *= 2) caps.push_back(c);
            caps.push_back(L.comp_max);
        }
        for (int n = 1; n <= n_max; ++n) {
            // mab776 M7c: passes of 3+ tokens (chained drafts) reserve their largest capacity only - it sizes the shared
            // arena; the smaller ones are built when the context first reaches them (~3x the graph nodes otherwise)
            for (int64_t cap : caps) {
                if (n > 2 && Impl::shared_on() && cap != caps.back()) continue;
                Impl::Var * var = attn_var(im, il, cap, n);
                if (!ggml_gallocr_reserve(var->allo, var->gf)) { im.err = "reserve: attention graph"; return false; }
                if (n == 1 && cap > 0 && comp_skip_on()) {   // the no-block-completes variant too
                    var = attn_var(im, il, cap, n, false);
                    if (!ggml_gallocr_reserve(var->allo, var->gf)) { im.err = "reserve: attention graph"; return false; }
                }
            }
            if (!ggml_gallocr_reserve(L.allo_finish[n], L.gf_finish[n])) { im.err = "reserve: finish"; return false; }
        }
        if (L.gf_predict && !ggml_gallocr_reserve(L.allo_predict, L.gf_predict)) { im.err = "reserve: predict"; return false; }
    }
    for (int n = 1; n <= n_max; ++n) {
        if (!ggml_gallocr_reserve(im.allo_init[n], im.gf_init[n])) { im.err = "reserve: init"; return false; }
        if (!ggml_gallocr_reserve(im.allo_head[n], im.gf_head[n])) { im.err = "reserve: head"; return false; }
        if (im.il_mtp >= 0 && (!ggml_gallocr_reserve(im.allo_mtp_in[n], im.gf_mtp_in[n]) ||
                               !ggml_gallocr_reserve(im.allo_mtp_head[n], im.gf_mtp_head[n]))) {
            im.err = "reserve: mtp"; return false;
        }
    }
    if (im.il_mtp >= 0 && n_max > 1) {   // mab776 M7c: the chained drafts' input and argmax graphs
        for (int c = 0; c < n_max; ++c) {
            if (!im.gf_mtp_arg[c]) {
                im.gf_mtp_arg[c] = build_head(im, 1, true, c);
                im.allo_mtp_arg[c] = im.make_allo();
            }
            if (!im.allo_mtp_arg[c] || !ggml_gallocr_reserve(im.allo_mtp_arg[c], im.gf_mtp_arg[c])) {
                im.err = "reserve: mtp argmax"; return false;
            }
            if (c > 0 && !im.gf_mtp_chain[c]) {
                im.gf_mtp_chain[c] = build_mtp_in(im, 1, c);
                im.allo_mtp_chain[c] = im.make_allo();
                if (!im.allo_mtp_chain[c] || !ggml_gallocr_reserve(im.allo_mtp_chain[c], im.gf_mtp_chain[c])) {
                    im.err = "reserve: mtp chain"; return false;
                }
            }
        }
    }
    if (Impl::shared_on() && im.shared_allo) {
        std::fprintf(stderr, "dense compute buffers: one shared arena %.1f MiB for every decode graph\n",
                     ggml_gallocr_get_buffer_size(im.shared_allo, 0) / 1048576.0);
    } else {   // where the dense half's compute-buffer VRAM goes: one gallocr arena per graph
        auto sz = [](ggml_gallocr_t a) { return a ? (double) ggml_gallocr_get_buffer_size(a, 0) : 0.0; };
        double attn = 0, attn_max = 0, fin = 0, pred = 0, other = 0;
        int64_t n_attn = 0;
        for (Impl::Layer & L : im.ly) {
            for (Impl::Var & v : L.vars) { const double b = sz(v.allo); attn += b; attn_max = std::max(attn_max, b); ++n_attn; }
            for (ggml_gallocr_t a : L.allo_finish) fin += sz(a);
            pred += sz(L.allo_predict);
        }
        for (int n = 0; n <= (int) kNtMax; ++n) other += sz(im.allo_init[n]) + sz(im.allo_head[n]);
        for (int n = 0; n <= (int) kNtMax; ++n) other += sz(im.allo_mtp_in[n]) + sz(im.allo_mtp_head[n]);
        for (int c = 0; c < (int) kNtMax; ++c) other += sz(im.allo_mtp_chain[c]) + sz(im.allo_mtp_arg[c]);
        const double M = 1048576.0;
        std::fprintf(stderr, "dense compute buffers: attention %.0f MiB (%lld graphs, largest %.1f MiB), finish %.0f, "
                             "predict %.0f, init/head/mtp %.0f MiB\n", attn / M, (long long) n_attn, attn_max / M,
                     fin / M, pred / M, other / M);
    }
    return true;
}

void Ds4Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    Impl::Layer & L = im.ly[(size_t) il];
    std::memcpy(im.in_host.data() + ((const uint8_t *) L.rbias->data - im.in_base), bias, (size_t) im.NEXP * 4);
    im.in_pos = -1;   // re-upload the span before the next attention graph
}

bool Ds4Dense::finish_layer(int il, const float * routed_sum) { return finish_layer_n(il, 1, routed_sum); }

bool Ds4Dense::finish_layer_n(int il, int n, const float * routed_sum, bool wait_sum_event) {
    Impl & im = *p_;
    // mab776 M11b: the tier's mix into routed_sum ran on its own stream and recorded ev_sum - the finish waits there.
    // Queued first, so even if a step below fails, the next dense sync (reset, alloc) also drains the tier's work.
    if (!routed_sum && wait_sum_event && im.ev_sum) ggml_backend_event_wait(im.backend, im.ev_sum);
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "finish_layer: bad layer"; return false; }
    if (!im.ensure_n(n)) return false;
    Impl::Layer & L = im.ly[(size_t) il];
    if (!alloc_graph(L.allo_finish[n], L.gf_finish[n], im.backend)) { im.err = "gallocr(finish) failed"; return false; }
    const size_t rbytes = (size_t) (im.D * n) * 4;
    if (!routed_sum) {}   // mab776 M11 (DS4_DEV_SUM): the expert tier mixed straight into im.routed_sum on the device
    else if (async_on() && !im.dec_routed) {   // once: D x kNtMax floats, pinned when the backend has a host buffer type
        if (ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(im.backend)))
            if ((im.dec_hbuf = ggml_backend_buft_alloc_buffer(hbt, (size_t) (im.D * kNtMax) * 4)))
                im.dec_routed = (float *) ggml_backend_buffer_get_base(im.dec_hbuf);
    }
    if (!routed_sum) {}
    else if (async_on() && im.dec_routed) {   // the previous upload from it finished: a readback waited since
        std::memcpy(im.dec_routed, routed_sum, rbytes);
        ggml_backend_tensor_set_async(im.backend, im.routed_sum, im.dec_routed, 0, rbytes);
    } else {
        ggml_backend_tensor_set(im.routed_sum, routed_sum, 0, rbytes);
    }
    if (!compute_dec(im.backend, L.gf_finish[n])) {
        im.err = "finish graph compute failed";
        return false;
    }
    // the async window stays INSIDE a pass: its last trunk layer and the MTP block wait, so whatever follows a pass
    // (plain copies of the state - prefill_state_to_decode writes x_state, which the MTP block's finish writes too -
    // checkpoints, snapshots) finds the stream idle
    if (async_on() && (il + 1 >= im.n_trunk || il == im.il_mtp)) ggml_backend_synchronize(im.backend);
    if (!L.host_llast.empty()) {
        if (async_on()) ggml_backend_synchronize(im.backend);
        ggml_backend_tensor_get(L.t_llast, L.host_llast.data(), 0, (size_t) (im.D * im.HC * n) * 4);
    }
    L.have_finish = true;
    return true;
}

bool Ds4Dense::logits(const float ** out, int * n_vocab) { return logits_n(1, out, n_vocab); }

bool Ds4Dense::logits_n(int n, const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_head[n], im.gf_head[n])) { im.err = "gallocr(head) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_head[n]) != GGML_STATUS_SUCCESS) {
        im.err = "head graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(im.logits_t[n], im.host_logits.data(), 0, (size_t) (im.g.vocab_size * n) * 4);
    if (out) *out = im.host_logits.data();
    if (n_vocab) *n_vocab = (int) im.g.vocab_size;
    return true;
}

// embd[tids] for n tokens into i_emb's first n columns (looked up on the host, from the GGUF mmap)
static bool put_emb(Ds4Dense::Impl & im, const int * tids, int n) {
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        const int tid = tids[t];
        if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
        const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    return true;
}

bool Ds4Dense::mtp_chain_begin(int next_tid, int src) {
    Impl & im = *p_;
    if (im.il_mtp < 0) { im.err = "mtp_chain_begin: no MTP head loaded"; return false; }
    if (src < 0 || src >= (int) kNtMax) { im.err = "mtp_chain_begin: bad source column"; return false; }
    if (!im.ensure_n(1)) return false;
    if (!im.gf_mtp_chain[src]) {
        im.gf_mtp_chain[src] = src == 0 ? im.gf_mtp_in[1] : build_mtp_in(im, 1, src);
        im.allo_mtp_chain[src] = src == 0 ? nullptr : im.make_allo();
        if (src != 0 && !im.allo_mtp_chain[src]) { im.err = "ggml_gallocr_new failed"; return false; }
    }
    ggml_gallocr_t a = src == 0 ? im.allo_mtp_in[1] : im.allo_mtp_chain[src];
    if (!alloc_graph(a, im.gf_mtp_chain[src])) { im.err = "gallocr(mtp chain) failed"; return false; }
    if (!put_emb(im, &next_tid, 1)) return false;
    if (ggml_backend_graph_compute(im.backend, im.gf_mtp_chain[src]) != GGML_STATUS_SUCCESS) {
        im.err = "mtp chain input graph compute failed";
        return false;
    }
    return true;
}

bool Ds4Dense::mtp_argmax(int col, int * tok) {
    Impl & im = *p_;
    if (im.il_mtp < 0) { im.err = "mtp_argmax: no MTP head loaded"; return false; }
    if (col < 0 || col >= (int) kNtMax) { im.err = "mtp_argmax: bad column"; return false; }
    if (!im.gf_mtp_arg[col]) {
        im.gf_mtp_arg[col] = build_head(im, 1, true, col);
        im.allo_mtp_arg[col] = im.make_allo();
        if (!im.allo_mtp_arg[col]) { im.err = "ggml_gallocr_new failed"; return false; }
    }
    if (!alloc_graph(im.allo_mtp_arg[col], im.gf_mtp_arg[col])) { im.err = "gallocr(mtp argmax) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_mtp_arg[col]) != GGML_STATUS_SUCCESS) {
        im.err = "mtp argmax graph compute failed";
        return false;
    }
    int32_t v = -1;
    ggml_backend_tensor_get(im.mtp_arg_t[col], &v, 0, 4);
    if (v < 0 || v >= im.g.vocab_size) { im.err = "mtp argmax out of range"; return false; }
    *tok = (int) v;
    return true;
}

bool Ds4Dense::mtp_begin(const int * next_tids, int n) {
    Impl & im = *p_;
    if (im.il_mtp < 0) { im.err = "mtp_begin: no MTP head loaded"; return false; }
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_mtp_in[n], im.gf_mtp_in[n])) { im.err = "gallocr(mtp_in) failed"; return false; }
    if (!put_emb(im, next_tids, n)) return false;
    if (const char * dbg = std::getenv("DS4_DBG_MTP_H")) {    // append the trunk state the MTP reads (D*HC*n f32)
        std::vector<float> v((size_t) (im.D * im.HC * n));
        ggml_backend_tensor_get(im.x_state, v.data(), 0, v.size() * 4);
        if (std::FILE * f = std::fopen(dbg, "ab")) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
    }
    if (ggml_backend_graph_compute(im.backend, im.gf_mtp_in[n]) != GGML_STATUS_SUCCESS) {
        im.err = "mtp input graph compute failed";
        return false;
    }
    if (const char * dbg = std::getenv("DS4_DBG_MTP_IN")) {   // append the MTP input state (D*HC*n f32) to a file
        std::vector<float> v((size_t) (im.D * im.HC * n));
        ggml_backend_tensor_get(im.x_state, v.data(), 0, v.size() * 4);
        if (std::FILE * f = std::fopen(dbg, "ab")) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
    }
    return true;
}

bool Ds4Dense::mtp_logits_n(int n, const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (im.il_mtp < 0) { im.err = "mtp_logits_n: no MTP head loaded"; return false; }
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_mtp_head[n], im.gf_mtp_head[n])) { im.err = "gallocr(mtp head) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_mtp_head[n]) != GGML_STATUS_SUCCESS) {
        im.err = "mtp head graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(im.mtp_logits_t[n], im.host_logits.data(), 0, (size_t) (im.g.vocab_size * n) * 4);
    if (out) *out = im.host_logits.data();
    if (n_vocab) *n_vocab = (int) im.g.vocab_size;
    return true;
}

// ================================================================== prompt chunks (prefill_*)

static const Ds4Dense::Impl::Layer * pf_layer_of_ratio(const Ds4Dense::Impl & im, int64_t ratio) {
    for (const auto & L : im.ly) if (L.ratio == ratio) return &L;
    return nullptr;
}

static Ds4Dense::Impl::PfRatio * pf_ratio_set(Ds4Dense::Impl & im, int64_t ratio) {
    return ratio == 4 ? &im.P.r4 : ratio == 128 ? &im.P.r128 : nullptr;
}

// the chunk-sized hand-off and input tensors (P) + the shared graph allocator, at the first prefill_begin
static bool pf_alloc(Ds4Dense::Impl & im) {
    const int64_t NP = im.pf_cap, D = im.D, HC = im.HC;
    ggml_init_params ip = { /*mem_size*/ 64ull * ggml_tensor_overhead(), /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    im.pf_ctx = ggml_init(ip);
    if (!im.pf_ctx) { im.err = "prefill: ggml_init"; return false; }
    ggml_context * c = im.pf_ctx;
    auto& P = im.P;
    P.x_state    = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NP);
    P.routed_sum = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NP);
    P.i_tid      = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
    P.i_emb      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NP);
    P.i_pos      = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
    P.hap        = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NP);
    P.post_f     = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NP);
    P.comb_f     = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NP);
    P.shexp      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NP);
    P.i_slot_raw = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.RAW);
    P.i_idx_raw  = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA - 1 + NP);
    P.i_mask_raw = ggml_new_tensor_1d(c, GGML_TYPE_F16, (im.SWA - 1 + NP) * NP);
    for (int64_t r : { (int64_t) 4, (int64_t) 128 }) {
        const Ds4Dense::Impl::Layer * L = pf_layer_of_ratio(im, r);
        if (!L) continue;
        Ds4Dense::Impl::PfRatio & R = *pf_ratio_set(im, r);
        R.i_slot_comp  = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
        R.i_slot_state = ggml_new_tensor_1d(c, GGML_TYPE_I32, L->ring_sz);
        R.i_comp_pos   = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
        R.i_state_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
        R.i_idx_state  = ggml_new_tensor_1d(c, GGML_TYPE_I32, L->ring * NP);
        if (vis_dev_on()) R.i_nvis   = ggml_new_tensor_1d(c, GGML_TYPE_F32, NP);   // DS4_VIS_DEV (FINDINGS s23)
        else              R.vis_full = ggml_new_tensor_1d(c, GGML_TYPE_F32, L->comp_max * NP);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(im.backend);
    im.pf_buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!im.pf_buf) { im.err = "prefill: cannot allocate the chunk tensors"; return false; }
    // router output blocks: the decode layout (o_blk bytes per token), o_f and o_i alias the same bytes
    const size_t ob = (size_t) im.o_blk * (size_t) NP;
    im.pf_obuf = ggml_backend_buft_alloc_buffer(buft, ob + 256);
    if (!im.pf_obuf) { im.err = "prefill: cannot allocate the router output blocks"; return false; }
    P.o_f = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t) (ob / 4));
    P.o_i = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) (ob / 4));
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(im.pf_obuf);
    if (ggml_backend_tensor_alloc(im.pf_obuf, P.o_f, base) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(im.pf_obuf, P.o_i, base) != GGML_STATUS_SUCCESS) {
        im.err = "prefill: cannot place the router output blocks"; return false;
    }
    {
        const size_t rows = (size_t) (D * NP) * sizeof(float), al = 256;
        const size_t o_sz = (ob + al - 1) / al * al, r_sz = (rows + al - 1) / al * al, tot = o_sz + 2 * r_sz;
        uint8_t * base = nullptr;
        ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(im.backend));
        if (hbt && (im.pf_hbuf = ggml_backend_buft_alloc_buffer(hbt, tot)))
            base = (uint8_t *) ggml_backend_buffer_get_base(im.pf_hbuf);
        else {
            im.pf_hvec.assign(tot, 0);
            base = im.pf_hvec.data();
        }
        im.pf_out_host = base;
        im.pf_fn = (float *) (base + o_sz);
        im.pf_routed = (float *) (base + o_sz + r_sz);
    }
    im.pf_allo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));   // its own (freed by pf_free)
    if (!im.pf_allo) { im.err = "prefill: ggml_gallocr_new"; return false; }
    im.pf_meta.assign(64ull * 1024 * 1024, 0);
    return true;
}

// point the decode tensors the builders use at the chunk's (and back: swapping twice restores)
static void pf_swap(Ds4Dense::Impl & im, Ds4Dense::Impl::Layer * L) {
    auto& P = im.P;
    std::swap(im.x_state, P.x_state); std::swap(im.routed_sum, P.routed_sum);
    std::swap(im.i_tid, P.i_tid); std::swap(im.i_emb, P.i_emb); std::swap(im.i_pos, P.i_pos);
    if (!L) return;
    std::swap(L->hap, P.hap); std::swap(L->post_f, P.post_f); std::swap(L->comb_f, P.comb_f);
    std::swap(L->shexp, P.shexp); std::swap(L->o_f, P.o_f); std::swap(L->o_i, P.o_i);
    std::swap(L->i_slot_raw, P.i_slot_raw); std::swap(L->i_idx_raw, P.i_idx_raw); std::swap(L->i_mask_raw, P.i_mask_raw);
    if (Ds4Dense::Impl::PfRatio * R = pf_ratio_set(im, L->ratio)) {
        std::swap(L->i_slot_comp, R->i_slot_comp); std::swap(L->i_slot_state, R->i_slot_state);
        std::swap(L->i_comp_pos, R->i_comp_pos); std::swap(L->i_state_pos, R->i_state_pos);
        std::swap(L->i_idx_state, R->i_idx_state); std::swap(L->vis_full, R->vis_full);
        std::swap(L->i_nvis, R->i_nvis);
    }
}

// build one chunk graph (pf mode), run it once on the shared allocator, `after` reads its outputs, drop it.  Built
// fresh each time (uid 0): ggml-cuda runs a graph whose nodes changed directly, without capturing it.
template <class Build, class After>
static bool pf_run(Ds4Dense::Impl & im, Ds4Dense::Impl::Layer * L, Build build, After after, const char * what) {
    ggml_init_params ip = { im.pf_meta.size(), im.pf_meta.data(), /*no_alloc*/ true };
    ggml_context * tc = ggml_init(ip);
    if (!tc) { im.err = std::string(what) + ": ggml_init"; return false; }
    ggml_context * keep = im.gctx;
    im.gctx = tc;
    im.pf = true;
    pf_swap(im, L);
    ggml_cgraph * gf = build();
    pf_swap(im, L);
    im.pf = false;
    im.gctx = keep;
    bool ok = gf && ggml_gallocr_alloc_graph(im.pf_allo, gf) &&
              ggml_backend_graph_compute(im.backend, gf) == GGML_STATUS_SUCCESS;
    if (!ok) im.err = std::string(what) + ": chunk graph failed";
    else ok = after();
    ggml_free(tc);
    return ok;
}

bool Ds4Dense::prefill_begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (im.pf_cap <= 0) { im.err = "prefill: Ds4DenseConfig::prefill_chunk is 0"; return false; }
    if (n < 1 || n > im.pf_cap || pos0 < 0) { im.err = "prefill: chunk size/position out of range"; return false; }
    if (!im.pf_ctx && !pf_alloc(im)) return false;
    auto& P = im.P;
    const int64_t pos_last = (int64_t) pos0 + n - 1;
    for (const Impl::Layer & L : im.ly)
        if (L.ratio != 0 && pos_last / L.ratio >= L.comp_max) { im.err = "prefill: beyond the compressed cache capacity"; return false; }

    // the decode input span on the device (the chunk graphs read the route bias from it); decode re-uploads after
    ggml_backend_tensor_set(im.i_span, im.in_host.data(), 0, im.in_host.size());
    im.in_pos = -1;

    std::vector<int32_t> t32((size_t) n), p32((size_t) n);
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        const int tid = tids[t];
        if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
        t32[(size_t) t] = tid;
        p32[(size_t) t] = pos0 + t;
        const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(P.i_tid, t32.data(), 0, t32.size() * 4);
    ggml_backend_tensor_set(P.i_pos, p32.data(), 0, p32.size() * 4);
    ggml_backend_tensor_set(P.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);

    // a ring row of position tok: before the chunk -> its slot in the ring, inside it -> ring_sz + (tok - pos0)
    auto ext = [&](int64_t tok, int64_t ring_sz) -> int32_t {
        return (int32_t) (tok < pos0 ? ((tok % ring_sz) + ring_sz) % ring_sz : ring_sz + (tok - pos0));
    };
    auto tail = [&](int64_t ring_sz) {   // ring slots of the chunk's last min(n, ring_sz) positions
        const int64_t k = std::min<int64_t>(n, ring_sz);
        std::vector<int32_t> v((size_t) k);
        for (int64_t j = 0; j < k; ++j) v[(size_t) j] = (int32_t) ((pos0 + n - k + j) % ring_sz);
        return v;
    };
    {   // raw window (the same for every layer)
        const int64_t n_raw = im.SWA + n - 1;
        std::vector<int32_t> idx((size_t) n_raw);
        std::vector<ggml_fp16_t> msk((size_t) (n_raw * n));
        const ggml_fp16_t h0 = ggml_fp32_to_fp16(0.0f), hinf = ggml_fp32_to_fp16(NEG_INF);
        for (int64_t i = 0; i < n_raw; ++i) {
            const int64_t tok = (int64_t) pos0 - im.SWA + 1 + i;
            idx[(size_t) i] = ext(tok, im.RAW);
            for (int t = 0; t < n; ++t) {
                const int64_t pt = (int64_t) pos0 + t;
                msk[(size_t) (t * n_raw + i)] = tok >= 0 && tok <= pt && tok > pt - im.SWA ? h0 : hinf;
            }
        }
        const std::vector<int32_t> sl = tail(im.RAW);
        ggml_backend_tensor_set(P.i_idx_raw, idx.data(), 0, idx.size() * 4);
        ggml_backend_tensor_set(P.i_mask_raw, msk.data(), 0, msk.size() * 2);
        ggml_backend_tensor_set(P.i_slot_raw, sl.data(), 0, sl.size() * 4);
    }
    for (int64_t r : { (int64_t) 4, (int64_t) 128 }) {   // compressed layers: the same per ratio
        const Impl::Layer * L = pf_layer_of_ratio(im, r);
        if (!L) continue;
        Impl::PfRatio & R = *pf_ratio_set(im, r);
        const int64_t cap = attn_cap(*L, pos_last);
        std::vector<int32_t> sp((size_t) n), cpp, sc, sidx;   // cpp / sc / sidx: one entry (ring) per emitted row
        cpp.reserve((size_t) n); sc.reserve((size_t) n); sidx.reserve((size_t) (L->ring * n));
        const bool emit_only = pf_emit_on();
        std::vector<float> nvis;
        std::vector<float> vis;
        if (vis_dev_on()) nvis.resize((size_t) n);
        else              vis.assign((size_t) (cap * n), NEG_INF);
        for (int t = 0; t < n; ++t) {
            const int64_t pt = (int64_t) pos0 + t, bl = pt / r;
            sp[(size_t) t]  = (int32_t) (pt % r);
            const bool later_same = t + 1 < n && (pt + 1) / r == bl;   // a later token of the chunk owns the row
            if (!emit_only || !later_same) {   // DS4_PF_EMIT: the rows that land in a slot only
                cpp.push_back((int32_t) (r * bl));
                sc.push_back((int32_t) (later_same ? L->comp_max : bl));
                const bool half = L->csa && ovl_gather_on();   // DS4_OVL_GATHER: half-row indices
                for (int64_t i = 0; i < L->ring; ++i) {
                    const int32_t slot = ext(pt - L->ring + 1 + i, L->ring_sz);
                    sidx.push_back(half ? 2 * slot + (i >= r ? 1 : 0) : slot);
                }
            }
            const int64_t nv = (pt + 1) / r;
            if (vis_dev_on()) nvis[(size_t) t] = (float) nv;
            else for (int64_t i = 0; i < nv && i < cap; ++i) vis[(size_t) (t * cap + i)] = 0.0f;
        }
        const std::vector<int32_t> ss = tail(L->ring_sz);
        R.n_emit = (int64_t) sc.size();
        ggml_backend_tensor_set(R.i_state_pos, sp.data(), 0, sp.size() * 4);
        ggml_backend_tensor_set(R.i_comp_pos, cpp.data(), 0, cpp.size() * 4);
        ggml_backend_tensor_set(R.i_slot_comp, sc.data(), 0, sc.size() * 4);
        ggml_backend_tensor_set(R.i_idx_state, sidx.data(), 0, sidx.size() * 4);
        ggml_backend_tensor_set(R.i_slot_state, ss.data(), 0, ss.size() * 4);
        if (vis_dev_on()) ggml_backend_tensor_set(R.i_nvis, nvis.data(), 0, nvis.size() * 4);
        else              ggml_backend_tensor_set(R.vis_full, vis.data(), 0, vis.size() * 4);
    }
    im.pf_pos0 = pos0;
    im.pf_n = n;
    return pf_run(im, nullptr, [&] { return build_init(im, n); }, [] { return true; }, "prefill_begin");
}

bool Ds4Dense::prefill_attn(int il, int * routed_ids, float * routed_w, const float ** ffn_norm_host) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || im.pf_n <= 0) { im.err = "prefill_attn: bad layer / no chunk"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    const int n = im.pf_n;
    const int64_t cap = attn_cap(L, (int64_t) im.pf_pos0 + n - 1);
    return pf_run(im, &L, [&] { return build_attn(im, il, cap, n, L); }, [&] {
        const size_t ob = (size_t) im.o_blk;
        ggml_backend_tensor_get(im.P.o_f, im.pf_out_host, 0, ob * (size_t) n);
        for (int t = 0; t < n; ++t) {
            const uint8_t * b0 = im.pf_out_host + ob * (size_t) t;
            const int32_t * ids = (const int32_t *) (b0 + im.o_ids_off);
            const float * wv = (const float *) (b0 + im.o_wts_off);
            for (int64_t i = 0; i < im.NUSED; ++i) {
                routed_ids[t * im.NUSED + i] = ids[i];
                routed_w[t * im.NUSED + i] = wv[i];
            }
            std::memcpy(im.pf_fn + (size_t) t * im.D, b0, (size_t) im.D * 4);
        }
        if (ffn_norm_host) *ffn_norm_host = im.pf_fn;
        if (im.dbg_hcab_d2) {   // DS4_CHECK_HCAB=1
            std::vector<float> d2((size_t) n), r2((size_t) n);
            ggml_backend_tensor_get(im.dbg_hcab_d2, d2.data(), 0, d2.size() * 4);
            ggml_backend_tensor_get(im.dbg_hcab_r2, r2.data(), 0, r2.size() * 4);
            const int64_t rb = raw_block_n(n);
            double mx = 0.0, sum = 0.0;
            int imx = 0;
            std::vector<double> blk((size_t) ((n + rb - 1) / rb), 0.0);
            for (int t = 0; t < n; ++t) {
                const double e = std::sqrt((double) d2[(size_t) t] / std::max((double) r2[(size_t) t], 1e-30));
                sum += e;
                blk[(size_t) (t / rb)] = std::max(blk[(size_t) (t / rb)], e);
                if (e > mx) { mx = e; imx = t; }
            }
            std::string bs;
            for (double e : blk) { char c[16]; std::snprintf(c, sizeof c, " %.1e", e); bs += c; }
            std::fprintf(stderr, "check-hcab: layer %d pos0 %d n %d: rel err max %.3g (row %d, block %d) mean %.3g | per block max:%s\n",
                         im.dbg_hcab_il, im.pf_pos0, n, mx, imx, (int) (imx / rb), sum / n, bs.c_str());
            im.dbg_hcab_d2 = im.dbg_hcab_r2 = nullptr;
        }
        return true;
    }, "prefill_attn");
}

bool Ds4Dense::prefill_finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || im.pf_n <= 0) { im.err = "prefill_finish: bad layer / no chunk"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    const int n = im.pf_n;
    if (routed_sum) ggml_backend_tensor_set(im.P.routed_sum, routed_sum, 0, (size_t) (im.D * n) * 4);
    // (nullptr: the sums are already in P.routed_sum - see prefill_routed_device)
    return pf_run(im, &L, [&] { return build_finish(im, n, L); }, [] { return true; }, "prefill_finish");
}

bool Ds4Dense::prefill_logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (im.pf_n <= 0 || row0 < 0 || nrows < 1 || row0 + nrows > im.pf_n) { im.err = "prefill_logits: rows out of range"; return false; }
    im.pf_row0 = row0;
    return pf_run(im, nullptr, [&] { return build_head(im, nrows); }, [&] {
        ggml_backend_tensor_get(im.pf_logits, out, 0, (size_t) (im.g.vocab_size * nrows) * 4);
        return true;
    }, "prefill_logits");
}

bool Ds4Dense::prefill_state_to_decode(int row0, int n) {
    Impl & im = *p_;
    if (!im.pf_ctx || !im.P.x_state || n < 1 || n > kNtMax || row0 < 0 || row0 + n > (int) im.P.x_state->ne[2]) {
        im.err = "prefill_state_to_decode: rows out of range";
        return false;
    }
    if (!im.ensure_n(n)) return false;
    const size_t row = (size_t) (im.D * im.HC) * sizeof(float);   // one position's hc streams
    std::vector<uint8_t> buf(row * (size_t) n);
    ggml_backend_tensor_get(im.P.x_state, buf.data(), row * (size_t) row0, buf.size());
    ggml_backend_tensor_set(im.x_state, buf.data(), 0, buf.size());
    return true;
}

bool Ds4Dense::prefill_end() {
    Impl & im = *p_;
    im.pf_n = 0;
    im.in_pos = -1;      // the decode inputs are uploaded again by the next attn_router
    im.cur_tid = -1;
    return true;
}

void Ds4Dense::prefill_release() { p_->pf_free(); }

const void * Ds4Dense::router_block_device(int il, int64_t * blk, int64_t * ids_off) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || ggml_backend_is_cpu(im.backend) || !im.ly[(size_t) il].o_span) return nullptr;
    if (blk) *blk = im.o_blk;
    if (ids_off) *ids_off = im.o_ids_off;
    return im.ly[(size_t) il].o_span->data;
}

void * Ds4Dense::record_attn_event() {
    Impl & im = *p_;
    if (!im.ovl_setup() || !im.ev_attn) return nullptr;
    ggml_backend_event_record(im.ev_attn, im.backend);   // behind whatever is queued: the attention graph just launched
    return im.ev_attn->context;
}

void * Ds4Dense::decode_sum_event() {
    Impl & im = *p_;
    return im.ovl_setup() && im.ev_sum ? im.ev_sum->context : nullptr;   // ggml-cuda's event context: the cudaEvent_t
}

void * Ds4Dense::decode_routed_device() {
    Impl & im = *p_;
    if (ggml_backend_is_cpu(im.backend) || !im.routed_sum) return nullptr;
    return im.routed_sum->data;
}

void * Ds4Dense::prefill_routed_device() {
    Impl & im = *p_;
    if (ggml_backend_is_cpu(im.backend)) return nullptr;
    if (!im.pf_ctx && im.pf_cap > 0 && !pf_alloc(im)) return nullptr;
    return im.P.routed_sum ? im.P.routed_sum->data : nullptr;
}

float * Ds4Dense::prefill_routed_buffer() {
    Impl & im = *p_;
    if (!im.pf_ctx && im.pf_cap > 0 && !pf_alloc(im)) return nullptr;
    return im.pf_routed;
}

// taps: the LAST token of the most recent pass (for n = 1, the token)
const float * Ds4Dense::tap_attn_out(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_attn ? L.host_attn.data() + (size_t) (L.tap_n - 1) * im.D : nullptr;
}

const float * Ds4Dense::tap_attn_raw(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_attn ? L.host_attn_raw.data() + (size_t) (L.tap_n - 1) * im.NH * im.DH : nullptr;
}

const float * Ds4Dense::tap_l_last(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_finish ? L.host_llast.data() + (size_t) (L.tap_n - 1) * im.D * im.HC : nullptr;
}

// ================================================================== sequence state (mab776 M5: tools/ds4/ds4_serve.cpp)

bool Ds4Dense::Impl::snap_alloc(int slots) {
    const std::vector<ggml_tensor *> src = rings();
    ggml_init_params ip = { /*mem_size*/ ((size_t) slots * src.size() + 8) * ggml_tensor_overhead(), /*mem_buffer*/ nullptr,
                            /*no_alloc*/ true };
    snapctx = ggml_init(ip);
    if (!snapctx) { err = "ggml_init(snapshots) failed"; return false; }
    snap_bytes = 0;
    for (ggml_tensor * t : src) snap_bytes += ggml_nbytes(t);
    snaps.assign((size_t) slots, {});
    for (int s = 0; s < slots; ++s)
        for (ggml_tensor * t : src) snaps[(size_t) s].push_back(ggml_dup_tensor(snapctx, t));   // same layout
    snapbuf = ggml_backend_alloc_ctx_tensors_from_buft(snapctx, ggml_backend_get_default_buffer_type(backend));
    if (!snapbuf) {
        err = "cannot allocate " + std::to_string(slots) + " ring snapshots (" +
              std::to_string(snap_bytes >> 10) + " KiB each)";
        return false;
    }
    ggml_backend_buffer_clear(snapbuf, 0);
    snap_pos.assign((size_t) slots, -1);
    return true;
}

void Ds4Dense::reset() {
    Impl & im = *p_;
    ggml_backend_synchronize(im.backend);
    // exactly what init leaves: the state buffer cleared (raw window, value rings), then SENTINEL into the score rings.
    // NOT ggml_backend_buffer_clear(sbuf): rot (the indexer's Hadamard matrix), neg_ramp and i_idx_all are constants in
    // the same buffer.  Without the SENTINEL, a new sequence's first CSA block would pool the previous sequence's
    // states as its missing previous half (the compressor softmax has no position mask over its 2 * ratio rows).
    std::vector<float> sent;
    for (Impl::Layer & L : im.ly) {
        for (ggml_tensor * t : { L.raw, L.st_kv, L.ist_kv })
            if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.st_sc, L.ist_sc }) {
            if (!t) continue;
            sent.assign((size_t) ggml_nelements(t), SENTINEL);
            ggml_backend_tensor_set(t, sent.data(), 0, sent.size() * 4);
        }
    }
    ggml_backend_synchronize(im.backend);
    im.in_pos = -1;
    im.cur_tid = -1;
    std::fill(im.snap_pos.begin(), im.snap_pos.end(), -1);
}

bool Ds4Dense::snapshot(int pos, int slot) {
    Impl & im = *p_;
    if (slot < 0 || slot >= (int) im.snaps.size()) {
        im.err = "snapshot: no slot " + std::to_string(slot) + " (Ds4DenseConfig::snapshot_slots = " +
                 std::to_string(im.snaps.size()) + ")";
        return false;
    }
    if (pos < 0) { im.err = "snapshot: negative position"; return false; }
    ggml_backend_synchronize(im.backend);
    const std::vector<ggml_tensor *> src = im.rings();
    for (size_t i = 0; i < src.size(); ++i) ggml_backend_tensor_copy(src[i], im.snaps[(size_t) slot][i]);
    ggml_backend_synchronize(im.backend);
    im.snap_pos[(size_t) slot] = pos;
    return true;
}

bool Ds4Dense::restore(int slot) {
    Impl & im = *p_;
    if (slot < 0 || slot >= (int) im.snaps.size() || im.snap_pos[(size_t) slot] < 0) {
        im.err = "restore: slot " + std::to_string(slot) + " holds no snapshot";
        return false;
    }
    ggml_backend_synchronize(im.backend);
    const std::vector<ggml_tensor *> dst = im.rings();
    for (size_t i = 0; i < dst.size(); ++i) ggml_backend_tensor_copy(im.snaps[(size_t) slot][i], dst[i]);
    ggml_backend_synchronize(im.backend);
    im.in_pos = -1;      // the decode inputs are uploaded again by the next attn_router
    im.cur_tid = -1;
    return true;
}

int Ds4Dense::snapshot_pos(int slot) const {
    const Impl & im = *p_;
    return slot >= 0 && slot < (int) im.snap_pos.size() ? im.snap_pos[(size_t) slot] : -1;
}

size_t Ds4Dense::snapshot_bytes() const { return p_->snap_bytes; }
int    Ds4Dense::snapshot_slots() const { return (int) p_->snaps.size(); }

// ---- conversation parking (mab776 M6)
namespace {
struct ParkHdr {
    uint32_t magic = 0x4B524150;   // "PARK"
    uint32_t version = 1;
    int32_t  max_pos = 0, n_slots = 0;
    uint64_t ring_bytes = 0, total = 0;
};
// the compressed rows a park keeps for one layer: the blocks below max_pos and the partial block's row
int64_t park_rows(const Ds4Dense::Impl::Layer & L, int max_pos) {
    if (!L.ratio || !L.comp) return 0;
    return std::min<int64_t>(L.comp_max + 1, (int64_t) max_pos / L.ratio + 1);
}
}  // namespace

size_t Ds4Dense::park_bytes(int max_pos) const {
    const Impl & im = *p_;
    size_t ring_b = 0;
    for (ggml_tensor * t : im.rings()) ring_b += ggml_nbytes(t);
    size_t b = sizeof(ParkHdr) + im.snaps.size() * sizeof(int32_t) + ring_b * (1 + im.snaps.size());
    for (const Impl::Layer & L : im.ly) {
        const int64_t r = park_rows(L, max_pos);
        if (L.comp)  b += (size_t) r * L.comp->nb[1];
        if (L.icomp) b += (size_t) r * L.icomp->nb[1];
    }
    return b;
}

bool Ds4Dense::park(int max_pos, std::vector<uint8_t> & out) {
    Impl & im = *p_;
    if (max_pos < 0) { im.err = "park: negative position"; return false; }
    for (int sp : im.snap_pos)
        if (sp > max_pos) { im.err = "park: a snapshot holds positions past max_pos"; return false; }
    const std::vector<ggml_tensor *> rs = im.rings();
    ParkHdr h;
    h.max_pos = max_pos;
    h.n_slots = (int32_t) im.snaps.size();
    for (ggml_tensor * t : rs) h.ring_bytes += ggml_nbytes(t);
    h.total = park_bytes(max_pos);
    out.resize((size_t) h.total);
    uint8_t * p = out.data();
    std::memcpy(p, &h, sizeof h);
    p += sizeof h;
    for (int sp : im.snap_pos) { const int32_t v = sp; std::memcpy(p, &v, 4); p += 4; }
    ggml_backend_synchronize(im.backend);
    auto get = [&](ggml_tensor * t, size_t n) { ggml_backend_tensor_get(t, p, 0, n); p += n; };
    for (ggml_tensor * t : rs) get(t, ggml_nbytes(t));
    for (const auto & slot : im.snaps)
        for (ggml_tensor * t : slot) get(t, ggml_nbytes(t));
    for (const Impl::Layer & L : im.ly) {
        const int64_t r = park_rows(L, max_pos);
        if (L.comp)  get(L.comp,  (size_t) r * L.comp->nb[1]);
        if (L.icomp) get(L.icomp, (size_t) r * L.icomp->nb[1]);
    }
    if ((size_t) (p - out.data()) != out.size()) { im.err = "park: size mismatch"; return false; }
    return true;
}

bool Ds4Dense::unpark(const std::vector<uint8_t> & in) {
    Impl & im = *p_;
    ParkHdr h;
    if (in.size() < sizeof h) { im.err = "unpark: truncated"; return false; }
    std::memcpy(&h, in.data(), sizeof h);
    const std::vector<ggml_tensor *> rs = im.rings();
    size_t ring_b = 0;
    for (ggml_tensor * t : rs) ring_b += ggml_nbytes(t);
    if (h.magic != ParkHdr().magic || h.version != 1 || h.n_slots != (int32_t) im.snaps.size() || h.ring_bytes != ring_b ||
        h.total != in.size() || h.total != park_bytes(h.max_pos)) {
        im.err = "unpark: the parked state does not fit this engine's layout";
        return false;
    }
    const uint8_t * p = in.data() + sizeof h;
    std::vector<int> sp((size_t) h.n_slots);
    for (int & v : sp) { int32_t x = 0; std::memcpy(&x, p, 4); v = x; p += 4; }
    ggml_backend_synchronize(im.backend);
    auto set = [&](ggml_tensor * t, size_t n) { ggml_backend_tensor_set(t, p, 0, n); p += n; };
    for (ggml_tensor * t : rs) set(t, ggml_nbytes(t));
    for (auto & slot : im.snaps)
        for (ggml_tensor * t : slot) set(t, ggml_nbytes(t));
    for (Impl::Layer & L : im.ly) {
        const int64_t r = park_rows(L, h.max_pos);
        if (L.comp)  set(L.comp,  (size_t) r * L.comp->nb[1]);
        if (L.icomp) set(L.icomp, (size_t) r * L.icomp->nb[1]);
    }
    ggml_backend_synchronize(im.backend);
    im.snap_pos = sp;
    im.in_pos = -1;      // the decode inputs are uploaded again by the next attn_router
    im.cur_tid = -1;
    return true;
}
