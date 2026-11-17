import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from hcu_sglang.utils.detection import get_gcn_arch


@triton.jit
def _flat_scatter(S, D, DI, SI, RI, E: tl.constexpr, BLOCKS: tl.constexpr,
                  SL: tl.constexpr, SR: tl.constexpr, SS: tl.constexpr,
                  DL: tl.constexpr, DR: tl.constexpr, RS: tl.constexpr,
                  ST: tl.constexpr, DS: tl.constexpr, B: tl.constexpr):
    request = tl.program_id(0) // BLOCKS
    block = tl.program_id(0) % BLOCKS
    layer = tl.program_id(1).to(tl.int64)
    step = tl.load(SI + request).to(tl.int64)
    if step < 0:
        return
    dst = tl.load(DI + request).to(tl.int64)
    src = tl.load(RI + request).to(tl.int64)
    if (dst < 0) | (dst >= DS) | (src < 0) | (src >= RS) | (step >= ST):
        return
    offsets = block * B + tl.arange(0, B)
    value = tl.load(S + layer * SL + src * SR + step * SS + offsets, offsets < E)
    tl.store(D + layer * DL + dst * DR + offsets, value, offsets < E)


def flat_scatter(dst, src, dst_ids, steps, src_ids=None):
    dst_ids = dst_ids.to(torch.int32).contiguous()
    steps = steps.to(torch.int32).contiguous()
    if src_ids is None:
        src_ids = torch.arange(steps.numel(), device=steps.device, dtype=torch.int32)
    else:
        src_ids = src_ids.to(torch.int32).contiguous()
    elems = dst[0, 0].numel()
    blocks = triton.cdiv(elems, 1024)
    _flat_scatter[(steps.numel() * blocks, dst.shape[0])](
        src, dst, dst_ids, steps, src_ids, elems, blocks,
        src.stride(0), src.stride(1), src.stride(2), dst.stride(0), dst.stride(1),
        src.shape[1], src.shape[2], dst.shape[1], 1024, num_warps=4)


def around_state_scatter(original, dst, src, dst_indices_raw, step_indices_raw,
                         src_indices_raw=None):
    if not envs.SGLANG_QWEN4_FLAT_STATE_SCATTER.get():
        return original(dst, src, dst_indices_raw, step_indices_raw, src_indices_raw)
    if not (
        dst.is_cuda and dst.dtype == src.dtype == torch.float32
        and dst.device == src.device and get_gcn_arch().startswith("gfx936")
        and dst.ndim == 5 and src.ndim == 6
        and dst.shape[0] == src.shape[0] == 36
        and dst.shape[2:] == src.shape[3:] == (48, 128, 128)
        and dst.is_contiguous() and src.is_contiguous()
        and dst.shape[1] > 0 and src.shape[1] > 0 and src.shape[2] > 0
        and dst_indices_raw.ndim == step_indices_raw.ndim == 1
        and dst_indices_raw.shape == step_indices_raw.shape
        and 0 < step_indices_raw.numel() <= 16
        and dst_indices_raw.device == step_indices_raw.device == dst.device
        and dst_indices_raw.dtype in (torch.int32, torch.int64)
        and step_indices_raw.dtype in (torch.int32, torch.int64)
        and (src_indices_raw is None or (
            src_indices_raw.shape == step_indices_raw.shape
            and src_indices_raw.device == dst.device
            and src_indices_raw.dtype in (torch.int32, torch.int64)
        ))
    ):
        return original(dst, src, dst_indices_raw, step_indices_raw, src_indices_raw)
    return flat_scatter(dst, src, dst_indices_raw, step_indices_raw, src_indices_raw)
