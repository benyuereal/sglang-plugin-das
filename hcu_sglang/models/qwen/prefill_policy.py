def pure_prefill_modes(modes, *, extend, idle) -> bool:
    return bool(modes) and extend in modes and all(m in (extend, idle) for m in modes)


def compact_prefill_counts(counts, *, pure_prefill, eligible):
    if not eligible or not pure_prefill or len(counts) <= 1:
        return None
    if any(n < 0 for n in counts):
        raise ValueError("negative DP token count")
    if not any(counts):
        return None
    return [max(int(n), 1) for n in counts]
