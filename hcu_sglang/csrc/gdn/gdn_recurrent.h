/*
 * GDN recurrent HIP kernel: exported C entry points and their ABI.
 *
 * This file is the single source of truth for the argument order, types and
 * tensor meanings of the C symbols exported by gdn_recurrent.hip.  The Python
 * wrapper (hcu_sglang/layers/attention/linear/gdn_recurrent.py) feeds this
 * file verbatim to ffi.cdef() and then calls the functions by name through
 * cffi, so it must stay a plain C declaration list: no preprocessor
 * directives, no include guard and no extern "C" wrapper (cffi rejects all
 * three).  It is not included by the .hip translation unit; the signatures
 * below are copied from the extern "C" definitions and must be kept in sync
 * with them by hand when an entry changes.
 *
 * Common notation:
 *   N        sequences in the batch.
 *   T        tokens per sequence (rows per sequence in the padded layout).
 *   H        number of q/k heads.
 *   HV       number of v/a/b heads; HV == H * hratio.
 *   K, V     q/k and v/o head dimensions.
 *   tokens   N * T.
 *   D        cache_steps: snapshot rows per state slot.
 *   slots    number of state slots (rows of h0_source).
 *
 * Stride arguments are in elements, not bytes.  q/k are (tokens, H, K) bf16,
 * v/o are (tokens, HV, V) bf16 and a/b are (tokens, HV) bf16; their token
 * stride is the element distance between two consecutive tokens (head and
 * dim are contiguous).  a and b carry independent strides on purpose: b is
 * often a slice of a wider buffer, so reusing a's stride reads another
 * token's row.  h0_source, o, snap_buf and the norm-pass buffers are dense:
 * the kernels index those without a stride argument.
 *
 * Pointers marked const are read-only for the kernel; the others are
 * written.  Null pointers are allowed only where a comment says so; the
 * kernels test them before use.
 */

/*
 * Full recurrent update, every step writes its own snapshot row (BPLUS off,
 * FMA_OUTER on).  The Python wrapper does not use this entry today; it is
 * kept exported for A/B work.
 */
void gdn_recurrent_update(
    const void *q,              /* bf16 (tokens, H, K), token stride stride_q */
    const void *k,              /* bf16 (tokens, H, K), token stride stride_k */
    const void *v,              /* bf16 (tokens, HV, V), token stride stride_v */
    const void *a,              /* bf16 (tokens, HV), token stride stride_a */
    const void *b,              /* bf16 (tokens, HV), token stride stride_b */
    const void *A_log,          /* fp32 (HV) */
    const void *dt_bias,        /* fp32 (HV); bf16 weights must be pre-cast */
    const void *cu_seqlens,     /* int32 (N + 1) offsets, or NULL if varlen=0 */
    void *h0_source,            /* fp32 (slots, HV, K, V), initial state in */
    const void *h0_indices,     /* int32 (N) slot id per sequence */
    void *o,                    /* bf16 (tokens, HV, V), dense output */
    void *snap_buf,             /* fp32 (slots * D, HV, K, V), or NULL */
    const void *snap_idx,       /* int32 (N) snapshot slot id per sequence */
    int cache_steps,            /* D, snapshot rows per slot */
    int N,                      /* sequences per launch */
    int T,                      /* tokens per sequence */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int V,                      /* v/o head dim */
    int hratio,                 /* HV / H */
    int stride_q,               /* q token stride, in elements */
    int stride_k,               /* k token stride, in elements */
    int stride_v,               /* v token stride, in elements */
    int stride_a,               /* a token stride, in elements */
    int stride_b,               /* b token stride, in elements */
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    int use_initial_state,      /* 1: read h0_source[h0_indices[n]] */
    int varlen,                 /* 1: cu_seqlens gives the token ranges */
    int disable_state_update,   /* 1: do not write the final state back */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * B+ update: identical math, but a slot writes only the one snapshot row
 * selected by store_step[n] (store_step[n] < 0 writes nothing) and slots
 * with h0_indices[n] < 0 exit immediately.  This is the production verify
 * entry.
 */
void gdn_recurrent_update_bplus(
    const void *q,              /* bf16 (tokens, H, K), token stride stride_q */
    const void *k,              /* bf16 (tokens, H, K), token stride stride_k */
    const void *v,              /* bf16 (tokens, HV, V), token stride stride_v */
    const void *a,              /* bf16 (tokens, HV), token stride stride_a */
    const void *b,              /* bf16 (tokens, HV), token stride stride_b */
    const void *A_log,          /* fp32 (HV) */
    const void *dt_bias,        /* fp32 (HV); bf16 weights must be pre-cast */
    const void *cu_seqlens,     /* int32 (N + 1) offsets, or NULL if varlen=0 */
    void *h0_source,            /* fp32 (slots, HV, K, V), initial state in */
    const void *h0_indices,     /* int32 (N) slot id per sequence; < 0 skips */
    void *o,                    /* bf16 (tokens, HV, V), dense output */
    void *snap_buf,             /* fp32 (slots * D, HV, K, V), or NULL */
    const void *snap_idx,       /* int32 (N) snapshot slot id per sequence */
    int cache_steps,            /* D, snapshot rows per slot */
    const void *store_step,     /* int32 (N) device array: step to write, or
                                 * NULL; a negative entry writes nothing */
    const void *store_row,      /* int32 (N) device array: snapshot row to
                                 * write, or NULL to write row store_step[n] */
    int N,                      /* sequences per launch */
    int T,                      /* tokens per sequence */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int V,                      /* v/o head dim */
    int hratio,                 /* HV / H */
    int stride_q,               /* q token stride, in elements */
    int stride_k,               /* k token stride, in elements */
    int stride_v,               /* v token stride, in elements */
    int stride_a,               /* a token stride, in elements */
    int stride_b,               /* b token stride, in elements */
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    int use_initial_state,      /* 1: read h0_source[h0_indices[n]] */
    int varlen,                 /* 1: cu_seqlens gives the token ranges */
    int disable_state_update,   /* 1: do not write the final state back */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * B+ state-only update (compensate replay): same arguments and store_step /
 * store_row semantics as gdn_recurrent_update_bplus, but skips the q l2norm
 * and the o = h . q output.  o is accepted and never written.
 */
void gdn_recurrent_update_bplus_state(
    const void *q,              /* bf16 (tokens, H, K), token stride stride_q */
    const void *k,              /* bf16 (tokens, H, K), token stride stride_k */
    const void *v,              /* bf16 (tokens, HV, V), token stride stride_v */
    const void *a,              /* bf16 (tokens, HV), token stride stride_a */
    const void *b,              /* bf16 (tokens, HV), token stride stride_b */
    const void *A_log,          /* fp32 (HV) */
    const void *dt_bias,        /* fp32 (HV); bf16 weights must be pre-cast */
    const void *cu_seqlens,     /* int32 (N + 1) offsets, or NULL if varlen=0 */
    void *h0_source,            /* fp32 (slots, HV, K, V), initial state in */
    const void *h0_indices,     /* int32 (N) slot id per sequence; < 0 skips */
    void *o,                    /* bf16 (tokens, HV, V), never written here */
    void *snap_buf,             /* fp32 (slots * D, HV, K, V), or NULL */
    const void *snap_idx,       /* int32 (N) snapshot slot id per sequence */
    int cache_steps,            /* D, snapshot rows per slot */
    const void *store_step,     /* int32 (N) device array: step to write, or
                                 * NULL; a negative entry writes nothing */
    const void *store_row,      /* int32 (N) device array: snapshot row to
                                 * write, or NULL to write row store_step[n] */
    int N,                      /* sequences per launch */
    int T,                      /* tokens per sequence */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int V,                      /* v/o head dim */
    int hratio,                 /* HV / H */
    int stride_q,               /* q token stride, in elements */
    int stride_k,               /* k token stride, in elements */
    int stride_v,               /* v token stride, in elements */
    int stride_a,               /* a token stride, in elements */
    int stride_b,               /* b token stride, in elements */
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    int use_initial_state,      /* 1: read h0_source[h0_indices[n]] */
    int varlen,                 /* 1: cu_seqlens gives the token ranges */
    int disable_state_update,   /* 1: do not write the final state back */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * Full-snapshot reference entry: every step writes its own snapshot row
 * (BPLUS off) with the o-accumulation FMA factoring disabled.  Used as the
 * bitwise A/B reference for the B+ row.
 */
void gdn_recurrent_update_nofma(
    const void *q,              /* bf16 (tokens, H, K), token stride stride_q */
    const void *k,              /* bf16 (tokens, H, K), token stride stride_k */
    const void *v,              /* bf16 (tokens, HV, V), token stride stride_v */
    const void *a,              /* bf16 (tokens, HV), token stride stride_a */
    const void *b,              /* bf16 (tokens, HV), token stride stride_b */
    const void *A_log,          /* fp32 (HV) */
    const void *dt_bias,        /* fp32 (HV); bf16 weights must be pre-cast */
    const void *cu_seqlens,     /* int32 (N + 1) offsets, or NULL if varlen=0 */
    void *h0_source,            /* fp32 (slots, HV, K, V), initial state in */
    const void *h0_indices,     /* int32 (N) slot id per sequence */
    void *o,                    /* bf16 (tokens, HV, V), dense output */
    void *snap_buf,             /* fp32 (slots * D, HV, K, V), or NULL */
    const void *snap_idx,       /* int32 (N) snapshot slot id per sequence */
    int cache_steps,            /* D, snapshot rows per slot */
    int N,                      /* sequences per launch */
    int T,                      /* tokens per sequence */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int V,                      /* v/o head dim */
    int hratio,                 /* HV / H */
    int stride_q,               /* q token stride, in elements */
    int stride_k,               /* k token stride, in elements */
    int stride_v,               /* v token stride, in elements */
    int stride_a,               /* a token stride, in elements */
    int stride_b,               /* b token stride, in elements */
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    int use_initial_state,      /* 1: read h0_source[h0_indices[n]] */
    int varlen,                 /* 1: cu_seqlens gives the token ranges */
    int disable_state_update,   /* 1: do not write the final state back */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * Multi-layer B+ state-only replay: one launch covers all GDN layers,
 * blockIdx.z selects a layer.  layer_ptrs is an int64 table of L rows with
 * 15 columns each, all values stored as int64:
 *   [0..9]  q, k, v, a, b, A_log, dt_bias, h0_source, o, snap_buf
 *   [10..14] token strides of q, k, v, a, b
 * The remaining arguments have the same meaning as the single-layer entry.
 * store_step may be NULL; store_row does not exist here (the kernel always
 * writes row store_step[n]).
 */
void gdn_recurrent_update_bplus_state_multi(
    const void *layer_ptrs,     /* int64 (L, 15) layer meta table (device) */
    const void *h0_indices,     /* int32 (N) slot id per sequence; < 0 skips */
    const void *snap_idx,       /* int32 (N) snapshot slot id per sequence */
    int cache_steps,            /* D, snapshot rows per slot */
    const void *store_step,     /* int32 (N) device array, or NULL */
    int L,                      /* number of layers in layer_ptrs */
    int N,                      /* sequences per launch */
    int T,                      /* tokens per sequence */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int V,                      /* v/o head dim */
    int hratio,                 /* HV / H */
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    int use_initial_state,      /* 1: read h0_source[h0_indices[n]] */
    int varlen,                 /* 1: cu_seqlens gives the token ranges */
    int disable_state_update,   /* 1: do not write the final state back */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * Norm pass: computes the l2normalized q/k and the softplus/sigmoid gating
 * for one layer, once per (token, head) instead of once per verify CTA.
 * n_units is tokens * H; unit u maps to token u / H and head u % H.
 * qn/kn are fp32 (tokens, H, K), dense.  eg_out/beta_out are fp32
 * (tokens, HV), dense.
 */
void gdn_verify_norm_pass(
    const void *q,              /* bf16 (tokens, H, K), token stride stride_q */
    const void *k,              /* bf16 (tokens, H, K), token stride stride_k */
    const void *a,              /* bf16 (tokens, HV), token stride stride_a */
    const void *b,              /* bf16 (tokens, HV), token stride stride_b */
    const void *A_log,          /* fp32 (HV) */
    const void *dt_bias,        /* fp32 (HV); bf16 weights must be pre-cast */
    void *qn,                   /* fp32 (tokens, H, K), dense */
    void *kn,                   /* fp32 (tokens, H, K), dense */
    void *eg_out,               /* fp32 (tokens, HV), dense */
    void *beta_out,             /* fp32 (tokens, HV), dense */
    int n_units,                /* tokens * H, one unit per (token, head) */
    int H,                      /* q/k heads */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int hratio,                 /* HV / H */
    int stride_q,               /* q token stride, in elements */
    int stride_k,               /* k token stride, in elements */
    int stride_a,               /* a token stride, in elements */
    int stride_b,               /* b token stride, in elements */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * Prenormalized verify: B+ update that reads q/k (and, when gate=1, eg/beta)
 * from the norm-pass buffers instead of normalizing inside every CTA.
 * qn/kn are fp32 (tokens, H, K) with row stride stride_qn; eg_buf/beta_buf
 * are fp32 (tokens, HV) with row stride stride_ab.  With gate=0 the gating
 * is recomputed from a/b and eg_buf/beta_buf are not read.
 */
void gdn_recurrent_update_bplus_prenorm(
    const void *q,              /* bf16 (tokens, H, K), token stride stride_q */
    const void *k,              /* bf16 (tokens, H, K), token stride stride_k */
    const void *v,              /* bf16 (tokens, HV, V), token stride stride_v */
    const void *a,              /* bf16 (tokens, HV), token stride stride_a */
    const void *b,              /* bf16 (tokens, HV), token stride stride_b */
    const void *A_log,          /* fp32 (HV) */
    const void *dt_bias,        /* fp32 (HV); bf16 weights must be pre-cast */
    const void *cu_seqlens,     /* int32 (N + 1) offsets, or NULL if varlen=0 */
    void *h0_source,            /* fp32 (slots, HV, K, V), initial state in */
    const void *h0_indices,     /* int32 (N) slot id per sequence; < 0 skips */
    void *o,                    /* bf16 (tokens, HV, V), dense output */
    void *snap_buf,             /* fp32 (slots * D, HV, K, V), or NULL */
    const void *snap_idx,       /* int32 (N) snapshot slot id per sequence */
    int cache_steps,            /* D, snapshot rows per slot */
    const void *store_step,     /* int32 (N) device array: step to write, or
                                 * NULL; a negative entry writes nothing */
    const void *store_row,      /* int32 (N) device array: snapshot row to
                                 * write, or NULL to write row store_step[n] */
    int N,                      /* sequences per launch */
    int T,                      /* tokens per sequence */
    int HV,                     /* v/a/b heads */
    int K,                      /* q/k head dim */
    int V,                      /* v/o head dim */
    int hratio,                 /* HV / H */
    int stride_q,               /* q token stride, in elements */
    int stride_k,               /* k token stride, in elements */
    int stride_v,               /* v token stride, in elements */
    int stride_a,               /* a token stride, in elements */
    int stride_b,               /* b token stride, in elements */
    float scale,                /* 1 / sqrt(K), applied to normalized q */
    float softplus_beta,        /* softplus(beta * x + threshold) */
    float softplus_threshold,
    int use_initial_state,      /* 1: read h0_source[h0_indices[n]] */
    int varlen,                 /* 1: cu_seqlens gives the token ranges */
    int disable_state_update,   /* 1: do not write the final state back */
    const void *qn,             /* fp32 (tokens, H, K), row stride stride_qn */
    const void *kn,             /* fp32 (tokens, H, K), row stride stride_qn */
    const void *eg_buf,         /* fp32 (tokens, HV), row stride stride_ab */
    const void *beta_buf,       /* fp32 (tokens, HV), row stride stride_ab */
    int stride_qn,              /* qn/kn token stride, in elements */
    int stride_ab,              /* eg_buf/beta_buf token stride, in elements */
    int gate,                   /* 1: load eg_buf/beta_buf, 0: recompute */
    void *stream);              /* hipStream_t as an opaque handle */

/*
 * causal_conv1d update for one GDN layer, bitwise against the Triton
 * `_causal_conv1d_update_kernel` for KERNEL_WIDTH=4 / bf16.
 *
 *   x           bf16 (batch, dim, seqlen), strides (s_x_seq, s_x_dim, s_x_tok)
 *   w           bf16 (dim, 4), strides (s_w_dim, s_w_width)
 *   bias        bf16 (dim), or NULL when has_bias=0
 *   conv_state  bf16 (num_cache_lines, dim, state_len), strides
 *               (s_cs_seq, s_cs_dim, s_cs_tok); updated in place
 *   conv_idx    int32 (batch), stride s_ci; cache line per sequence
 *   window      bf16 (rows, seqlen, dim, 4), strides
 *               (s_wseq, s_wstep, s_wdim, s_win), or NULL when save_inter=0
 *   inter_idx   int32 (batch), stride s_ii; window row per sequence
 *   nxt, sib    int32 or int64 (batch, seqlen), strides (s_nseq, s_ntok) and
 *               (s_sseq, s_stok); the type is chosen by idx64
 *   par_out     same dtype as nxt, (batch, seqlen), strides (s_pseq, s_ptok);
 *               written only in tree mode, may be NULL
 *   o           bf16 (batch, dim, seqlen), strides (s_o_seq, s_o_dim, s_o_tok)
 *
 * Flag semantics: has_bias, silu and save_inter select the corresponding
 * pieces; cont selects indexed (conv_idx) versus direct sequence addressing;
 * tree enables the parent-map path; use_pad enables the pad_slot_id skip;
 * idx64 selects int64 tree indices (0 = int32).  pad_slot_id is ignored when
 * use_pad=0.
 */
void gdn_causal_conv1d_update(
    const void *x,              /* bf16 (batch, dim, seqlen) */
    const void *w,              /* bf16 (dim, 4) conv weights */
    const void *bias,           /* bf16 (dim), or NULL when has_bias=0 */
    void *conv_state,           /* bf16 (num_cache_lines, dim, state_len) */
    const void *conv_idx,       /* int32 (batch), stride s_ci, or NULL */
    void *window,               /* bf16 (rows, seqlen, dim, 4), or NULL */
    const void *inter_idx,      /* int32 (batch), stride s_ii, or NULL */
    const void *nxt,            /* tree next-token map, or NULL */
    const void *sib,            /* tree sibling map, or NULL */
    void *par_out,              /* tree parent map, or NULL */
    void *o,                    /* bf16 (batch, dim, seqlen), dense output */
    int batch,                  /* sequences per launch */
    int dim,                    /* feature (channel) count */
    int seqlen,                 /* tokens per sequence, 1..GDN_CONV_MAX_T */
    int state_len,              /* conv_state columns, >= 3 */
    int num_cache_lines,        /* valid rows of conv_state */
    int s_x_seq,                /* x stride over batch */
    int s_x_dim,                /* x stride over features */
    int s_x_tok,                /* x stride over tokens */
    int s_w_dim,                /* w stride over features */
    int s_w_width,              /* w stride over the 4 taps */
    int s_cs_seq,               /* conv_state stride over cache lines */
    int s_cs_dim,               /* conv_state stride over features */
    int s_cs_tok,               /* conv_state stride over columns */
    int s_ci,                   /* conv_idx stride over batch */
    int s_wseq,                 /* window stride over rows */
    int s_wstep,                /* window stride over tokens */
    int s_wdim,                 /* window stride over features */
    int s_win,                  /* window stride over the 4 taps */
    int s_ii,                   /* inter_idx stride over batch */
    int s_nseq,                 /* nxt stride over batch */
    int s_ntok,                 /* nxt stride over tokens */
    int s_sseq,                 /* sib stride over batch */
    int s_stok,                 /* sib stride over tokens */
    int s_pseq,                 /* par_out stride over batch */
    int s_ptok,                 /* par_out stride over tokens */
    int s_o_seq,                /* o stride over batch */
    int s_o_dim,                /* o stride over features */
    int s_o_tok,                /* o stride over tokens */
    int pad_slot_id,            /* slot id to skip, meaningful when use_pad=1 */
    int has_bias,               /* 1: add bias */
    int silu,                   /* 1: apply silu to the conv result */
    int cont,                   /* 1: address conv_state via conv_idx */
    int save_inter,             /* 1: write the conv window */
    int tree,                   /* 1: use the tree parent-map path */
    int use_pad,                /* 1: skip sequences whose slot is pad_slot_id */
    int idx64,                  /* 1: tree index arrays are int64 */
    void *stream);              /* hipStream_t as an opaque handle */
