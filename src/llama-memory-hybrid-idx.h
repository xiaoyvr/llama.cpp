#pragma once

#include "llama-memory-hybrid.h"
#include "qsa-prefix-state.h"

#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both
//
// Two indexer-pooling designs coexist here, selected by architecture:
//   - kpool (glm5-next): persistent pools of get_kpool() cells, pooled keys kept in the idx cache
//   - qsa   (qwen4exp):  per-ubatch block map over the cells, consumed by the selected-attention graph

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // ---- kpool (glm5-next) ----

    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

    // ---- qsa (qwen4exp) ----

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio) const;

    void qsa_apply(const llama_ubatch & ubatch, const llama_kv_cache::slot_info & slots);
    void qsa_invalidate();
    bool qsa_prefix_matches(const llama_ubatch & ubatch) const;
    bool qsa_fast(int il, const llama_ubatch & ubatch) const;
    ggml_tensor * qsa_cache(ggml_context * ctx, int il, int64_t blocks) const;
    void qsa_fill_updates(ggml_tensor * members, ggml_tensor * positions, ggml_tensor * rows) const;
    void qsa_commit(int il) const;

private:
    void state_drop(llama_seq_id seq_id);

    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    std::unique_ptr<kpool_layout> kpool_lay;

    bool kpool_layout_shared() const;

    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();

    // ---- qsa state ----
    bool incremental_qsa = false;
    bool qsa_recover_pending = false;
    bool qsa_recover(llama_seq_id seq);
    qsa_prefix_state qsa_prefix;
    mutable std::vector<int64_t> qsa_ready;
    std::vector<ggml_tensor *> qsa_keys;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> qsa_buffers;
    bool qsa_metadata(ggml_tensor * cells, ggml_tensor * positions, ggml_tensor * bias,
                      ggml_tensor * tails, const llama_ubatch & ubatch, uint32_t ratio) const;
    void set_input_qsa_impl(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs,
                            const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context();

    bool next()  override;
    bool apply() override;

    const llama_kv_cache_context * get_idx() const;

    uint32_t get_n_stream() const;

    // ---- kpool (glm5-next) ----
    uint32_t get_n_kpool    () const;
    uint32_t get_n_kpool_new() const;
    bool get_kpool_cache_safe() const;
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    ggml_tensor * gather_mla_rows(ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const;
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

    // ---- qsa (qwen4exp) ----
    bool qsa_prefix_matches(const llama_ubatch & u) const { return mem && mem->qsa_prefix_matches(u); }
    bool qsa_fast(int il, const llama_ubatch & u) const { return mem && mem->qsa_fast(il, u); }
    uint32_t qsa_n_kv_window() const;
    ggml_tensor * qsa_cache(ggml_context * ctx, int il) const {
        return mem ? mem->qsa_cache(ctx, il, (qsa_n_kv_window()+3)/4) : nullptr;
    }
    void qsa_fill_updates(ggml_tensor * c, ggml_tensor * p, ggml_tensor * r) const { mem->qsa_fill_updates(c,p,r); }
    void qsa_commit(int il) const { mem->qsa_commit(il); }

    bool qsa_scalar_visibility(const llama_ubatch & ubatch) const;
    bool qsa_position_prefix(const llama_ubatch & ubatch) const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    const std::vector<uint32_t> ns_ubatch;

    // feeds both the kpool and qsa paths
    const slot_info_vec_t sinfos_kpool;

    const llama_memory_context_ptr ctx_idx;

    size_t i_cur = 0;

    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    std::unique_ptr<kpool_state> kpool_st;

    size_t i_kpool = SIZE_MAX;

    bool kpool_track() const;

    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
