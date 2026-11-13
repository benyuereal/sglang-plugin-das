"""MTP draft lm_head in fp8 (e4m3), served by the hbird fp8 decode tiles.

The draft head is the widest GEMM of a decode step (248320 x 5120); its bf16
copy streams ~2.37 GiB per call. The fp8 copy plus the hbird tiles take it from
1320 to 792 us at M=45. The proxy replaces only the draft model's lm_head and
delegates everything but quant_method, so the target head stays bf16.
"""

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

_MTP_LMHEAD_FP8_CHUNK_DEFAULT = 24576
_MTP_LMHEAD_FP8_E4M3_MAX = 448.0
_MTP_HBIRD_MAX_M_DEFAULT = 192


def _mtp_lmhead_fp8_enabled() -> bool:
    value = os.environ.get("SGLANG_MTP_LMHEAD_FP8", "0")
    return value.strip().lower() not in ("", "0", "false", "off")


def _mtp_lmhead_fp8_chunk() -> int:
    try:
        chunk = int(os.environ.get("SGLANG_MTP_LMHEAD_FP8_CHUNK", ""))
    except ValueError:
        chunk = _MTP_LMHEAD_FP8_CHUNK_DEFAULT
    if chunk <= 0:
        chunk = _MTP_LMHEAD_FP8_CHUNK_DEFAULT
    return max(16, chunk - chunk % 16)


def _mtp_lmhead_fp8_backend() -> str:
    return os.environ.get("SGLANG_MTP_LMHEAD_FP8_BACKEND", "hbird").strip().lower()


def _mtp_hbird_ops():
    try:
        import hbird_ops

        return hbird_ops
    except ImportError:
        import sys

        root = os.environ.get("HBIRD_GEMM_HOME") or ""
        path = os.path.join(root, "plugins")
        if not root or not os.path.isdir(path):
            raise
        if path not in sys.path:
            sys.path.insert(0, path)
        import hbird_ops

        return hbird_ops


class _MTPDraftLMHeadFP8Method:
    def __init__(self, weight_fp8, weight_scale, chunk, hbird_gemm=None, hbird_max_m=0):
        self.weight_fp8 = weight_fp8
        self.weight_scale = weight_scale
        self.chunk = chunk
        self.hbird_gemm = hbird_gemm
        self.hbird_max_m = hbird_max_m

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        assert bias is None, "draft fp8 lm_head does not support a bias"
        xf = x.to(torch.float32)
        xs = (
            xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            / _MTP_LMHEAD_FP8_E4M3_MAX
        )
        xq = (xf / xs).clamp(
            -_MTP_LMHEAD_FP8_E4M3_MAX, _MTP_LMHEAD_FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        vocab = self.weight_fp8.shape[0]
        out = torch.empty((x.shape[0], vocab), dtype=torch.bfloat16, device=x.device)
        if self.hbird_gemm is not None and x.shape[0] <= self.hbird_max_m:
            self.hbird_gemm(
                xq,
                self.weight_fp8,
                xs,
                self.weight_scale,
                out=out,
                prepare_shape=False,
            )
            return out
        for i in range(0, vocab, self.chunk):
            j = min(i + self.chunk, vocab)
            out[:, i:j] = torch._scaled_mm(
                xq,
                self.weight_fp8[i:j].t(),
                scale_a=xs,
                scale_b=self.weight_scale[i:j].t(),
                out_dtype=torch.bfloat16,
            )
        return out


class _MTPDraftLMHeadFP8Proxy:
    def __init__(self, base, method: _MTPDraftLMHeadFP8Method) -> None:
        self._base = base
        self.quant_method = method

    def __getattr__(self, name):
        if name in ("_base", "set_lora", "apply_lora"):
            raise AttributeError(name)
        return getattr(self._base, name)

    @property
    def weight(self) -> torch.Tensor:
        return self._base.weight


def _mtp_quantize_lmhead_rows(weight: torch.Tensor):
    vocab, hidden = weight.shape
    weight_fp8 = torch.empty(
        (vocab, hidden), dtype=torch.float8_e4m3fn, device=weight.device
    )
    weight_scale = torch.empty((vocab, 1), dtype=torch.float32, device=weight.device)
    block = 4096
    for i in range(0, vocab, block):
        j = min(i + block, vocab)
        wf = weight[i:j].to(torch.float32)
        scale = (
            wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            / _MTP_LMHEAD_FP8_E4M3_MAX
        )
        weight_fp8[i:j] = (wf / scale).clamp(
            -_MTP_LMHEAD_FP8_E4M3_MAX, _MTP_LMHEAD_FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        weight_scale[i:j] = scale
        del wf
    return weight_fp8, weight_scale


def _mtp_hbird_draft_head(weight_fp8: torch.Tensor, weight_scale: torch.Tensor):
    if _mtp_lmhead_fp8_backend() in ("torch", "0", "false", "off", "no"):
        return None, 0
    try:
        ops = _mtp_hbird_ops()
        vocab, hidden = weight_fp8.shape
        device = weight_fp8.device
        ms = []
        for m in range(1, _MTP_HBIRD_MAX_M_DEFAULT + 1):
            if ops.guard(m, vocab, hidden) != 0:
                break
            ms.append(m)
        if not ms:
            raise RuntimeError("guard refuses every M")
        started = time.perf_counter()
        ops.prepare_weight(weight_fp8)
        for m in ms:
            ops.prepare(m, vocab, hidden, device)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.info(
            "[mtp_draft_lmhead_fp8] hbird fp8 unavailable (%s); "
            "staying on the chunked _scaled_mm path",
            exc,
        )
        return None, 0
    logger.info(
        "[mtp_draft_lmhead_fp8] draft head on hbird fp8: packed B %.2f GiB, "
        "%d M workspaces (%d..%d), %.2fs",
        weight_fp8.numel() / 2**30,
        len(ms),
        ms[0],
        ms[-1],
        time.perf_counter() - started,
    )
    return ops.hbird_fp8_gemm, ms[-1]


def _eligible(model) -> bool:
    """Cheap gates first: a captured stream cannot allocate, and the proxy must
    not wrap itself."""
    if getattr(model, "_mtp_fp8_installed", False):
        return False
    if not _mtp_lmhead_fp8_enabled():
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None or weight.dim() != 2 or not weight.is_cuda:
        return False
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if isinstance(head, _MTPDraftLMHeadFP8Proxy):
        return False
    logits_processor = getattr(model, "logits_processor", None)
    return not (
        logits_processor is not None
        and getattr(logits_processor, "use_fp32_lm_head", False)
    )


def install(model) -> bool:
    if not _eligible(model):
        return False
    head = model.lm_head
    weight = head.weight
    try:
        started = time.perf_counter()
        weight_fp8, weight_scale = _mtp_quantize_lmhead_rows(weight.detach())
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.warning(
            "[mtp_draft_lmhead_fp8] disabling, quantization failed: %s", exc
        )
        return False
    hbird_gemm, hbird_max_m = _mtp_hbird_draft_head(weight_fp8, weight_scale)
    chunk = _mtp_lmhead_fp8_chunk()
    object.__setattr__(
        model,
        "lm_head",
        _MTPDraftLMHeadFP8Proxy(
            head,
            _MTPDraftLMHeadFP8Method(
                weight_fp8, weight_scale, chunk, hbird_gemm, hbird_max_m
            ),
        ),
    )
    model._mtp_fp8_installed = True
    print(
        "[mtp_draft_lmhead_fp8] draft head %d x %d -> e4m3, %.2fs, "
        "%.2f -> %.2f GiB, chunk=%d, hbird_max_m=%d"
        % (
            weight.shape[0],
            weight.shape[1],
            time.perf_counter() - started,
            weight.numel() * weight.element_size() / 2**30,
            (weight_fp8.numel() + weight_scale.numel() * 4) / 2**30,
            chunk,
            hbird_max_m,
        ),
        flush=True,
    )
    return True


def _drop_proxy(model) -> None:
    try:
        if isinstance(getattr(model, "lm_head", None), _MTPDraftLMHeadFP8Proxy):
            object.__delattr__(model, "lm_head")
    except Exception:  # noqa: BLE001
        pass


def _around_set_lm_head(original, self, target_lm_head):
    out = original(self, target_lm_head)
    try:
        _drop_proxy(self)
        self._mtp_fp8_installed = False
        install(self)
    except Exception:  # noqa: BLE001 - optional path
        logger.warning("[mtp_draft_lmhead_fp8] install failed", exc_info=True)
    return out


def _around_set_embed_and_head(original, self, embed, head):
    _drop_proxy(self)
    self._mtp_fp8_installed = False
    out = original(self, embed, head)
    try:
        install(self)
    except Exception:  # noqa: BLE001
        logger.warning("[mtp_draft_lmhead_fp8] reinstall failed", exc_info=True)
    return out


def register() -> bool:
    if not _mtp_lmhead_fp8_enabled():
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    base = "sglang.srt.models.qwen3_5_mtp.Qwen3_5ForCausalLMMTP."
    HookRegistry.register(
        base + "set_lm_head_from_target", _around_set_lm_head, HookType.AROUND
    )
    HookRegistry.register(
        base + "set_embed_and_head", _around_set_embed_and_head, HookType.AROUND
    )
    logger.info("[mtp_draft_lmhead_fp8] draft head fp8 hooks registered")
    return True
