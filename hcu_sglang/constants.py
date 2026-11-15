"""Numeric contracts shared by the HCU integrations."""

# Largest head dim the fa3 fp8 path builds a descale tile for; the upstream
# prepare_paged_mha_query applies the same bound.
FA3_MAX_DESCALE_HEAD_DIM = 256

# Spare rows on slot-indexed buffers, so an out-of-range slot parks without a
# bounds check on the hot path.
SLOT_BUFFER_SLACK = 3

# Row floor for the GDN seed scratch: a one-slot chunk must not force the next
# chunk to reallocate.
SEED_SCRATCH_MIN_ROWS = 128

# M values a decode CUDA graph can issue (verify runs bs * draft tokens).
DECODE_MAX_M = 128

# Bytes in a GiB, for the size lines these modules log.
BYTES_PER_GIB = 1 << 30

# Cap on how many skipped layers one log line lists.
SKIPPED_LOG_LIMIT = 8

# Row block the MTP draft head is quantized in.
MTP_LMHEAD_QUANT_BLOCK = 4096

# element alignment of the fp8 chunk size, and the amax floor used when scaling.
FP8_ALIGN_ELEMS = 16
AMAX_EPS = 1e-8

# lm_head quant methods SGLang runs as plain bf16, so the packed-B path may serve
# them too; mirrors logits_processor._UNQUANTIZED_LM_HEAD_METHODS.
UNQUANTIZED_LM_HEAD_METHODS = frozenset(
    {"UnquantizedEmbeddingMethod", "UnquantizedLinearMethod", "PackWeightMethod"}
)
