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
