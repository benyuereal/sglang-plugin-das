from __future__ import annotations

import ctypes
import fcntl
import os
import re
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.models import qwen4_exp as N
from sglang.srt.layers.quantization.compressed_tensors.utils import check_equal_or_regex_match, should_ignore_layer
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens, is_dp_gatherv_active
from hcu_adapter.mapped_host_dtk import _hip_runtime, _device_ptr_for_host_tensor, _wrap_mapped_host_as_cuda_tensor

globals().update({name: value for name, value in vars(N).items() if not name.startswith("__")})
get_attention_dp_size = lambda: get_parallel().attn_dp_size
get_tensor_model_parallel_world_size = lambda: get_parallel().tp_size
_HOST_REGISTER_LOCK = "/tmp/sglang_ple_bf16_hostregister.lock"
_HOST_REGISTER_FLAGS = (0x02, 0x03, 0x00)
_PLE_PIN_ENV = "SGLANG_PLE_PIN_MEMORY"
_PLE_REGISTER_MAX_ATTEMPTS = 8
_PLE_OOM_EXCEPTIONS = (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError))

import functools
from hcu_sglang.layers.dp_attention import _aiter_reduce_scatter_tensor
from hcu_sglang.models.qwen.ple_probe import probe_ngram_quant_flags

@functools.lru_cache(maxsize=1)
def _etp_rs_min_tokens() -> int:
    from sglang.srt.environ import envs

    return envs.SGLANG_PLE_ETP_REDUCE_SCATTER_MIN_TOKENS.get()


def dp_reduce_scatter_etp(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input (unreduced ETP partials)
    forward_batch: ForwardBatch,
):
    """Combine ETP-across-DP partial embeddings (PLE-style sharded lookups).

    Every DP rank gathers its vocab shard for ALL global tokens, producing
    partial sums; the combine must SUM them and keep this rank's token rows.
    The legacy pair (``tensor_model_parallel_all_reduce`` + ``dp_scatter``)
    moves the full buffer twice per rank; a reduce-scatter halves the wire
    traffic and skips the extra full-buffer copy. Layout must mirror
    dp_scatter exactly, so fall back to the legacy pair whenever the
    per-rank sizes are not available or do not match the buffers.
    """
    if global_tokens.shape[0] < _etp_rs_min_tokens():
        # Small-message regime: reduce-scatter (and reduce_scatterv) lose to
        # the fused all-reduce on tiny payloads; keep the legacy pair.
        tensor_model_parallel_all_reduce(global_tokens)
        dp_scatter(local_tokens, global_tokens, forward_batch)
        return
    if get_parallel().attn_cp_size == 1 and _is_etp_reduce_scatter_shape_ok(
        local_tokens, global_tokens
    ):
        if _aiter_reduce_scatter_tensor(local_tokens, global_tokens):
            return
        get_tp_group().reduce_scatter_tensor(local_tokens, global_tokens)
        return
    # The reduce_scatterv sizes contract (get_dp_global_num_tokens) is only
    # authoritative under the SUM_LEN/gatherv regime the MoE combine already
    # uses; everywhere else fall back to the legacy all_reduce + dp_scatter.
    sizes = get_dp_global_num_tokens() if is_dp_gatherv_active() else None
    if (
        sizes is not None
        and sum(sizes) == global_tokens.shape[0]
        and local_tokens.is_contiguous()
        and global_tokens.is_contiguous()
    ):
        n = sizes[get_parallel().attn_dp_rank]
        if 0 < n <= local_tokens.shape[0]:
            local_tokens.fill_(0)
            get_tp_group().reduce_scatterv(
                global_tokens, output=local_tokens[:n], sizes=sizes
            )
            return
    tensor_model_parallel_all_reduce(global_tokens)
    dp_scatter(local_tokens, global_tokens, forward_batch)


def _is_etp_reduce_scatter_shape_ok(
    local_tokens: torch.Tensor, global_tokens: torch.Tensor
) -> bool:
    """Equal-chunk (MAX_LEN) layout check: each DP rank owns one equal slot."""
    dp_size = get_attention_dp_size()
    return (
        get_tensor_model_parallel_world_size() == dp_size
        and dp_size > 1
        and global_tokens.shape[0] == local_tokens.shape[0] * dp_size
        and local_tokens.is_contiguous()
        and global_tokens.is_contiguous()
    )


def _ple_etp_reduce_scatter_enabled(gather_dp_tokens: bool, tp_size: int) -> bool:
    """ETP-across-DP combine via reduce-scatter for the sharded PLE table.

    Under DP attention with attn_tp_size == 1 the PLE table's TP shard group
    IS the DP group, so the per-rank gathers are unreduced partial sums and
    the combine can reduce-scatter directly (half the wire traffic of
    all_reduce + dp_scatter, and no extra full-buffer copy). vLLM reference:
    b681808/#199 (engram_config.embedding_across_dp), which uses all_reduce +
    narrow; this is strictly cheaper.
    """
    p = get_parallel()
    return (gather_dp_tokens and tp_size > 1
            and envs.SGLANG_PLE_ETP_REDUCE_SCATTER.get()
            and p.attn_cp_size == p.attn_tp_size == 1
            and tp_size == p.attn_dp_size == p.tp_size)


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    weight_scale_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    is_int8: tl.constexpr,
    is_per_row_scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_int8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.int8))
    elif is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    if is_per_row_scale:
        weight_scale_ptr = weight_scale_ptr.to(tl.int64).to(
            tl.pointer_type(tl.bfloat16)
        )
        row_scale = tl.load(weight_scale_ptr + local_idx).to(tl.bfloat16)
        values *= row_scale
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


def _drop_module_weight(embedding) -> None:
    params = getattr(embedding, "_parameters", None)
    if isinstance(params, dict) and "weight" in params:
        params.pop("weight")
        return
    if hasattr(embedding, "weight"):
        try:
            delattr(embedding, "weight")
        except AttributeError:
            pass


def _host_register_err_text(err) -> str:
    try:
        code = int(err)
    except Exception:
        code = err
    msg = str(err)
    try:
        raw = torch.cuda.cudart().cudaGetErrorString(err)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        msg = raw
    except Exception:
        pass
    return f"rc={code} {err} {msg}"


def _try_host_register(tensor: torch.Tensor) -> bool:
    """Map existing pageable CPU pages for UVA without a second hipHostMalloc.

    Full-table pin_memory=True OOMs BF16 PLE on HCU (~12.8GB/rank GTT).
    hostRegister reuses the pageable allocation so the Triton UVA gather
    stays CUDA-graph-safe. HCU's CUDA DeviceContext hijack rejects large
    cudaHostRegister on non-0 devices / concurrent ranks, so prefer HIP.
    """
    if tensor.device.type != "cpu" or tensor.numel() == 0:
        return False
    if tensor.is_pinned():
        return True
    if not tensor.is_contiguous():
        return False
    if not torch.cuda.is_available():
        return False
    nbytes = tensor.numel() * tensor.element_size()
    ptr = int(tensor.data_ptr())
    device = torch.cuda.current_device() if torch.cuda.is_initialized() else -1
    hip = _hip_runtime()
    register = getattr(torch.cuda.cudart(), "cudaHostRegister", None)
    for flags in _HOST_REGISTER_FLAGS:
        if hip is not None:
            try:
                rc = int(hip.hipHostRegister(ptr, nbytes, flags))
            except Exception as exc:
                logger.warning(
                    "PLE hipHostRegister raised flags=%s nbytes=%s: %s",
                    flags,
                    nbytes,
                    exc,
                )
            else:
                if rc == 0:
                    logger.info(
                        "PLE hipHostRegister ok flags=%s nbytes=%.2fGB ptr=%#x device=%s",
                        flags,
                        nbytes / (1024**3),
                        ptr,
                        device,
                    )
                    return True
                logger.warning(
                    "PLE hipHostRegister failed flags=%s nbytes=%s ptr=%#x device=%s rc=%s",
                    flags,
                    nbytes,
                    ptr,
                    device,
                    rc,
                )
        if register is None:
            continue
        try:
            err = register(ptr, nbytes, flags)
        except Exception as exc:
            logger.warning(
                "PLE cudaHostRegister raised flags=%s nbytes=%s: %s",
                flags,
                nbytes,
                exc,
            )
            continue
        if err is None or int(err) == 0:
            logger.info(
                "PLE cudaHostRegister ok flags=%s nbytes=%.2fGB ptr=%#x device=%s",
                flags,
                nbytes / (1024**3),
                ptr,
                device,
            )
            return True
        logger.warning(
            "PLE cudaHostRegister failed flags=%s nbytes=%s ptr=%#x device=%s %s",
            flags,
            nbytes,
            ptr,
            device,
            _host_register_err_text(err),
        )
    return False


def _register_table_for_uva(tensor: torch.Tensor) -> bool:
    """Map a pageable CPU PLE table for UVA without a second hipHostMalloc."""
    if tensor.is_pinned():
        return True
    # Serialize across TP ranks. A barrier during construct can no-op, and
    # concurrent large hipHostRegister on HCU returns hipErrorInvalidValue.
    lock_path = os.environ.get("SGLANG_PLE_HOSTREGISTER_LOCK", _HOST_REGISTER_LOCK)
    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    except Exception as exc:
        logger.warning("PLE hostRegister lock unavailable: %s", exc)
        lock_file = None
    try:
        return _try_host_register(tensor)
    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception:
                pass
            lock_file.close()


def _ple_pin_requested() -> bool:
    """Allow ops to skip the pinned attempt on GTT-limited HCU nodes."""
    return os.environ.get(_PLE_PIN_ENV, "auto").strip().lower() != "off"


def _alloc_cpu_table(
    shape: Tuple[int, ...], dtype: torch.dtype, *, name: str
) -> torch.Tensor:
    """Allocate a CPU PLE tensor: pinned when possible, pageable on OOM.

    hipHostMalloc carves pinned pages out of the limited HCU GTT, so a
    multi-GB int8/fp8 ngram table can fail with hipErrorOutOfMemory even
    when host RAM is free (Qwen3.8-Flash-Next ngram-int8, AIOPT-1536).
    Fall back to pageable memory; _maybe_register_uva then hostRegisters
    the same pages for UVA gathers without a second GTT allocation.
    Set SGLANG_PLE_PIN_MEMORY=off to skip the pinned attempt entirely.
    """
    if _ple_pin_requested():
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except _PLE_OOM_EXCEPTIONS as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "hiperroroutofmemory" not in message:
                raise
            logger.warning(
                "PLE pinned host alloc failed for %s (%s); "
                "falling back to pageable memory",
                name,
                exc,
            )
    return torch.empty(shape, dtype=dtype, device="cpu")


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    """CPU-resident PLE table.

    INT8/FP8 prefer pinned host memory for UVA gathers and fall back to a
    pageable table + hostRegister when hipHostMalloc exhausts the HCU GTT.
    BF16 reuses a pageable CPU table and hostRegister-maps it without a
    second multi-GB hipHostMalloc. On HIP the mapped table is wrapped as a
    CUDA tensor and gathered with ATen embedding so full decode CUDA graphs
    replay the current ids; Triton ``tl.load`` from mapped host is not
    graph-safe.

    The table stays in its checkpoint storage dtype (int8/fp8 with per-row
    channel scales, fp8/bf16 with a scalar scale, or bf16); gathers emit bf16.
    """

    _COPIED_ATTRIBUTES = (
        "quant_config",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "num_embeddings",
        "org_vocab_size",
        "padding_size",
        "num_added_embeddings",
        "use_presharded_weights",
        "org_vocab_size_padded",
        "num_embeddings_padded",
        "shard_indices",
        "embedding_dim",
        "num_embeddings_per_partition",
        "num_org_embeddings_per_partition",
        "num_added_embeddings_per_partition",
    )

    def __init__(self, embedding: VocabParallelEmbedding, gather_dp_tokens: bool = False, *, backend="pinned", table_dir=None) -> None:
        nn.Module.__init__(self)
        if backend != "pinned":
            raise NotImplementedError("HCU mapped PLE requires ple_offload_backend=pinned")
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.int8,
        ):
            raise TypeError(
                "PLE embedding offload requires bfloat16, fp8, or int8 weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The unquantized CUDA post-load hook is a no-op. Exclude this CPU-only
        # table so the generic loader does not stage it back to GPU unnecessarily.
        self.quant_method = None
        # Mirror the ETP reduce-scatter combine decision of the source
        # embedding (dp shard group == table TP shard group).
        self.etp_reduce_scatter = _ple_etp_reduce_scatter_enabled(
            gather_dp_tokens, self.tp_size
        )

        source_weight = embedding.weight
        released_source_weight = False
        extras = {
            name: value
            for name, value in vars(source_weight).items()
            if name != "data"
        }
        if source_weight.dtype == torch.bfloat16:
            # BF16 ngram tables are ~12.8GB/rank at TP8. hipHostMalloc of a
            # pinned twin OOMs on HCU GTT. Keep the pageable CPU copy and
            # hostRegister it below for graph-safe UVA. INT8/FP8 stay pinned.
            if source_weight.device.type == "cpu" and not source_weight.is_pinned():
                cpu_weight = source_weight
            else:
                shape = tuple(source_weight.shape)
                dtype = source_weight.dtype
                _drop_module_weight(embedding)
                del source_weight
                released_source_weight = True
                table = torch.empty(shape, dtype=dtype, device="cpu")
                cpu_weight = nn.Parameter(table, requires_grad=False)
                for name, value in extras.items():
                    setattr(cpu_weight, name, value)
        elif source_weight.device.type == "cpu" and source_weight.is_pinned():
            cpu_weight = source_weight
        else:
            # INT8/FP8 prefer a pinned twin for UVA gathers, but hipHostMalloc
            # draws from the limited HCU GTT and a multi-GB table can OOM with
            # hipErrorOutOfMemory even when host RAM is free. _alloc_cpu_table
            # falls back to pageable memory, which _maybe_register_uva
            # hostRegisters for the same UVA gathers.
            table = _alloc_cpu_table(
                tuple(source_weight.shape), source_weight.dtype, name="ngram weight"
            )
            cpu_weight = nn.Parameter(table, requires_grad=False)
            # This wrapper is installed while the model is being constructed;
            # the checkpoint loader fills cpu_weight afterwards. Copying the
            # uninitialized source would touch the full PLE table once for no
            # benefit (more than 50 GB in the Qwen3.8 ngram checkpoints).
        if not released_source_weight and cpu_weight is not source_weight:
            for name, value in extras.items():
                setattr(cpu_weight, name, value)
        cpu_weight.weight_loader = self.weight_loader
        self.register_parameter("weight", cpu_weight)
        source_scale = embedding.weight_scale
        if source_scale.numel() <= 1:
            if source_scale.device.type == "cpu" and source_scale.is_pinned():
                cpu_scale = source_scale
            else:
                cpu_scale = _alloc_cpu_table(
                    source_scale.shape, source_scale.dtype, name="ngram scalar scale"
                )
                cpu_scale.copy_(source_scale.detach().to("cpu"))
        elif source_scale.device.type == "cpu" and source_scale.is_pinned():
            cpu_scale = source_scale
        else:
            cpu_scale = _alloc_cpu_table(
                source_scale.shape, source_scale.dtype, name="ngram per-row scale"
            )
            cpu_scale.copy_(source_scale.to("cpu"))
        self.register_buffer("weight_scale", cpu_scale, persistent=True)
        self._device_scalar_scale = None
        if cpu_scale.numel() <= 1 and torch.cuda.is_available():
            # Graph capture cannot H2D-copy an unpinned CPU scalar.
            self._device_scalar_scale = cpu_scale.to(
                device=f"cuda:{torch.cuda.current_device()}",
                dtype=cpu_scale.dtype,
            )
        if (not released_source_weight) and cpu_weight is not source_weight:
            del embedding.weight
        self._kernel_weight_ptr = None
        self._kernel_scale_ptr = None
        self._device_weight_view = None
        self._device_weight_holder = None
        self._device_scale_view = None
        self._device_scale_holder = None
        self._uva_register_pending = False
        self._uva_register_attempts = 0
        scale_needs_register = (
            cpu_scale.device.type == "cpu" and not cpu_scale.is_pinned()
        )
        if not cpu_weight.is_pinned() or scale_needs_register:
            # Pageable table (BF16 reuse, or the pinned-alloc OOM fallback):
            # hostRegister it for UVA gathers. Registered pages dereference
            # like pinned GTT, so the int8/fp8 Triton path works unchanged.
            self._use_uva_gather = False
            self._uva_register_pending = True
            self._maybe_register_uva()
        else:
            self._use_uva_gather = True
            self._kernel_weight_ptr = _device_ptr_for_host_tensor(cpu_weight)
            if cpu_scale.device.type == "cpu" and cpu_scale.is_pinned():
                self._kernel_scale_ptr = _device_ptr_for_host_tensor(cpu_scale)
            self._bind_mapped_cuda_view()
        self._pageable_id_scratch = None
        self._pageable_row_scratch = None
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def _maybe_register_uva(self) -> None:
        if not getattr(self, "_uva_register_pending", False):
            return
        # hostRegister is a host-side map; skipping it during CUDA graph
        # capture leaves the pageable D2H path inside the graph.
        logger.info(
            "PLE hostRegister begin dtype=%s nbytes=%.2fGB device=%s",
            self.weight.dtype,
            self.weight.numel() * self.weight.element_size() / (1024**3),
            torch.cuda.current_device() if torch.cuda.is_available() else None,
        )
        registered = True
        for name, tensor in (("weight", self.weight), ("scale", self.weight_scale)):
            if tensor.device.type != "cpu" or tensor.is_pinned() or tensor.numel() == 0:
                continue
            if not _register_table_for_uva(tensor):
                logger.warning(
                    "PLE %s hostRegister failed nbytes=%.2fGB",
                    name,
                    tensor.numel() * tensor.element_size() / (1024**3),
                )
                registered = False
        if registered:
            self._uva_register_pending = False
            self._use_uva_gather = True
            self._kernel_weight_ptr = _device_ptr_for_host_tensor(self.weight)
            if self.weight_scale.device.type == "cpu":
                self._kernel_scale_ptr = _device_ptr_for_host_tensor(self.weight_scale)
            self._bind_mapped_cuda_view()
            return
        self._uva_register_attempts += 1
        if self._uva_register_attempts >= _PLE_REGISTER_MAX_ATTEMPTS:
            # Stop retrying in the decode loop; stay on the pageable gather.
            self._uva_register_pending = False
            logger.warning(
                "PLE hostRegister abandoned after %s attempts; using the "
                "pageable gather fallback",
                self._uva_register_attempts,
            )
        else:
            logger.warning(
                "PLE hostRegister failed; will retry, else pageable gather "
                "(not full-graph-safe)"
            )

    def _bind_mapped_cuda_view(self) -> None:
        if torch.version.hip is None:
            return
        if self.weight.dtype not in (torch.bfloat16, torch.int8, torch.float8_e4m3fn):
            return
        if not torch.cuda.is_available():
            return
        ptr = self._kernel_weight_ptr or _device_ptr_for_host_tensor(self.weight)
        try:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            view, holder = _wrap_mapped_host_as_cuda_tensor(
                ptr, tuple(self.weight.shape), self.weight.dtype, device
            )
        except Exception as exc:
            logger.warning("PLE mapped CUDA view wrap failed: %s", exc)
            self._device_weight_view = None
            self._device_weight_holder = None
            self._device_scale_view = None
            self._device_scale_holder = None
            if self.weight.dtype in (torch.int8, torch.float8_e4m3fn):
                # int8/fp8 tables may be served by FULL decode graphs by
                # default; without a mapped view the gather would fall back
                # to the pinned Triton kernel, which is NOT full-graph-safe
                # on HIP (replay collapses to token 0).
                logger.warning(
                    "PLE %s table has no mapped CUDA view; if decode CUDA "
                    "graphs are FULL, set SGLANG_PLE_DECODE_GRAPH_BREAKABLE=1 "
                    "to avoid silent gather corruption.",
                    self.weight.dtype,
                )
            return
        self._device_weight_view = view
        self._device_weight_holder = holder
        self._bind_mapped_scale_view(device)
        logger.info(
            "PLE %s mapped CUDA view ok shape=%s ptr=%#x device=%s%s",
            str(self.weight.dtype).replace("torch.", ""),
            tuple(view.shape),
            int(view.data_ptr()),
            view.device,
            " (per-row scale view bound)" if self._device_scale_view is not None else "",
        )

    def _bind_mapped_scale_view(self, device: torch.device) -> None:
        """Bind a mapped CUDA view over the pinned per-row scale buffer.

        Scalar scales stay on the existing device-copy path. The per-row
        scale is pinned at PinnedHostEmbedding construction for int8/fp8
        tables; loaders that expand a scalar scale later must keep it pinned
        (see load_qwen4_exp_ple_shard), and this helper is also called
        lazily from the gather when the expansion happened after init.
        """
        if self._device_weight_view is None:
            return
        if self._device_scale_view is not None or self.weight_scale.numel() <= 1:
            return
        if self.weight_scale.device.type != "cpu" or not self.weight_scale.is_pinned():
            return
        try:
            scale_ptr = self._kernel_scale_ptr or _device_ptr_for_host_tensor(
                self.weight_scale
            )
            scale_view, scale_holder = _wrap_mapped_host_as_cuda_tensor(
                scale_ptr,
                tuple(self.weight_scale.shape),
                self.weight_scale.dtype,
                device,
            )
        except Exception as exc:
            logger.warning("PLE mapped scale view wrap failed: %s", exc)
            self._device_scale_view = None
            self._device_scale_holder = None
            self._device_weight_view = None
            self._device_weight_holder = None
            logger.warning(
                "PLE per-row scale gather has no mapped CUDA view; if decode "
                "CUDA graphs are FULL, set SGLANG_PLE_DECODE_GRAPH_BREAKABLE=1 "
                "to avoid silent gather corruption."
            )
            return
        self._device_scale_view = scale_view
        self._device_scale_holder = scale_holder

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        super().weight_loader(param, loaded_weight)
        if int(param.data_ptr()) == int(self.weight.data_ptr()):
            self._maybe_register_uva()

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            # The gather kernel emits bf16 rows regardless of the table dtype.
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def _ensure_pageable_scratch(self, rows: int) -> None:
        if self._pageable_id_scratch is None or self._pageable_id_scratch.numel() < rows:
            self._pageable_id_scratch = torch.empty(
                rows, dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._pageable_row_scratch = torch.empty(
                rows,
                self.embedding_dim,
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )

    def _gather_from_pageable_host(
        self, flat_ids: torch.Tensor, output: torch.Tensor
    ) -> None:
        # Breakable CUDA graphs re-run this Python body on replay so the CPU
        # index sees the current ids. Full-graph capture only records the
        # small pinned H2D; callers should graph-break around this path.
        n = int(flat_ids.numel())
        self._ensure_pageable_scratch(n)
        self._pageable_id_scratch[:n].copy_(flat_ids, non_blocking=False)
        host_ids = self._pageable_id_scratch[:n]
        tp_start = self.shard_indices.org_vocab_start_index
        tp_end = self.shard_indices.org_vocab_end_index
        in_range = (host_ids >= tp_start) & (host_ids < tp_end)
        local = torch.where(in_range, host_ids - tp_start, torch.zeros_like(host_ids))
        local_rows = local.long()
        rows = self.weight.detach()[local_rows]
        if rows.dtype != torch.bfloat16:
            # Match the pinned/registered UVA kernel: dequantize int8/fp8 rows
            # with the per-row scale (a scalar scale is applied downstream).
            rows = rows.to(dtype=torch.bfloat16)
            if self.weight_scale.numel() > 1:
                scale = self.weight_scale.detach().to(dtype=torch.bfloat16)
                rows = rows * scale[local_rows].unsqueeze(-1)
        rows = torch.where(in_range.unsqueeze(-1), rows, torch.zeros_like(rows))
        self._pageable_row_scratch[:n].copy_(rows)
        output.view(n, self.embedding_dim).copy_(
            self._pageable_row_scratch[:n], non_blocking=False
        )

    def _gather_from_mapped_cuda_view(
        self, flat_ids: torch.Tensor, output: torch.Tensor
    ) -> None:
        weight = self._device_weight_view
        start = self.shard_indices.org_vocab_start_index
        end = self.shard_indices.org_vocab_end_index
        if self._device_scale_view is None and self.weight_scale.numel() > 1:
            # A scalar scale expanded to per-row after init (fp8 auto-switch):
            # bind the mapped view lazily, host-side, before capture records.
            self._bind_mapped_scale_view(
                torch.device(f"cuda:{torch.cuda.current_device()}")
            )
            if self._device_weight_view is None:
                # Scale bind failed: the weight view was dropped too, so let
                # the caller take the pinned-Triton (breakable-only) path.
                self._gather_from_pinned_triton(flat_ids, output)
                return
        in_range = (flat_ids >= start) & (flat_ids < end)
        local = torch.where(in_range, flat_ids - start, torch.zeros_like(flat_ids))
        nrows = int(weight.shape[0])
        if nrows > 0:
            local = local.clamp(0, nrows - 1)
        rows = F.embedding(local, weight)
        if rows.dtype != torch.bfloat16:
            rows = rows.to(torch.bfloat16)
        scale_view = self._device_scale_view
        if scale_view is not None:
            # Per-row int8/fp8 tables: dequantize inside the gather exactly like
            # the pinned Triton kernel (row scale of the OWNING shard). All-ATen,
            # so the whole sequence replays inside full decode CUDA graphs.
            scale = F.embedding(local, scale_view).to(torch.bfloat16)
            rows = rows * scale
        rows = torch.where(in_range.unsqueeze(-1), rows, torch.zeros_like(rows))
        output.view(rows.shape).copy_(rows)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        self._maybe_register_uva()
        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            if not self._use_uva_gather:
                self._gather_from_pageable_host(flat_ids, output)
            elif self._device_weight_view is not None:
                self._gather_from_mapped_cuda_view(flat_ids, output)
            else:
                self._gather_from_pinned_triton(flat_ids, output)
        return output

    def _gather_from_pinned_triton(
        self, flat_ids: torch.Tensor, output: torch.Tensor
    ) -> None:
        weight_ptr = self._kernel_weight_ptr or int(self.weight.data_ptr())
        scale_ptr = self._kernel_scale_ptr or int(
            self.weight_scale.data_ptr()
        )
        _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
            weight_ptr,
            scale_ptr,
            flat_ids,
            output,
            embedding_dim=self.embedding_dim,
            tp_vocab_start=self.shard_indices.org_vocab_start_index,
            tp_vocab_end=self.shard_indices.org_vocab_end_index,
            is_fp8=self.weight.dtype == torch.float8_e4m3fn,
            is_int8=self.weight.dtype == torch.int8,
            is_per_row_scale=self.weight_scale.numel() > 1,
            BLOCK_D=self._block_d,
        )

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.etp_reduce_scatter:
                # Unreduced ETP partials: _finish_embedding_lookup combines
                # them with dp_reduce_scatter_etp instead.
                return output
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))


class NGramMethods:
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )
        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(
            config.make_ngram_vocab_size_divisible_by
        )
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.enable_ple_fusion = envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = (
            self._build_head_vocab_and_offsets()
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.use_attn_tp_ngram = _use_attn_tp_ngram()
        self.gather_dp_tokens = (
            is_dp_attention_enabled()
            and get_attention_dp_size() > 1
            and not self.use_attn_tp_ngram
        )
        # etp_reduce_scatter is set below, once ngram_embedding exists (its
        # tp_size feeds the gate); see _ple_etp_reduce_scatter_enabled.
        # Probe the quantization layout of the ngram table; naming rules and
        # branch semantics live in ple_probe (kept sglang-free for offline
        # tests, matchers injected from the real imports above).
        ple_not_ignored, ple_int8, ple_fp8_channelwise = probe_ngram_quant_flags(
            prefix,
            quant_config,
            config,
            should_ignore_layer=should_ignore_layer,
            check_equal_or_regex_match=check_equal_or_regex_match,
        )
        self.ple_per_row_scale = ple_int8 or ple_fp8_channelwise
        # Allocate the large PLE table on CPU from the start when offload is
        # requested. This avoids a transient GPU allocation without requiring
        # a separate process-wide environment variable.
        allocation_context = (
            torch.device("meta") if config.ple_offload_embedding else nullcontext()
        )
        with allocation_context:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                params_dtype=(
                    torch.int8
                    if ple_int8
                    else (
                        torch.float8_e4m3fn
                        if (
                            ple_fp8_channelwise
                            or (
                                quant_config is not None
                                and quant_config.get_name() == "fp8"
                            )
                            or getattr(config, "ple_embedding_dtype", None)
                            == "float8_e4m3fn"
                        )
                        else torch.bfloat16
                    )
                ),
                output_dtype=torch.bfloat16,
                use_attn_tp_group=self.use_attn_tp_ngram,
            )
        self.etp_reduce_scatter = _ple_etp_reduce_scatter_enabled(
            self.gather_dp_tokens, self.ngram_embedding.tp_size
        )
        scale_shape = (
            (self.ngram_embedding.num_embeddings_per_partition, 1)
            if self.ple_per_row_scale
            else (1,)
        )
        self.ngram_embedding.register_buffer(
            "weight_scale",
            torch.ones(scale_shape, dtype=torch.bfloat16),
            persistent=True,
        )
        if config.ple_offload_embedding:
            self.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                self.ngram_embedding, gather_dp_tokens=self.gather_dp_tokens,
                backend=getattr(config, "ple_offload_backend", "pinned"),
                table_dir=getattr(config, "ple_offload_dir", None),
            )


    def _scale_ple_embeddings(
        self,
        embeddings: torch.Tensor,
        lookup_ids: torch.Tensor,
    ) -> torch.Tensor:
        ngram_embedding = self.ngram_embedding
        if not self._uses_per_row_scale():
            # BF16 PLE offload keeps the scalar scale on CPU. CUDA graph
            # capture forbids unpinned H2D copies, so prefer the GPU replica
            # installed by Qwen4ExpPinnedHostEmbedding.
            scale = getattr(ngram_embedding, "_device_scalar_scale", None)
            if scale is None:
                scale = ngram_embedding.weight_scale
                if scale.device != embeddings.device or scale.dtype != embeddings.dtype:
                    if scale.device.type == "cpu" and not scale.is_pinned():
                        scale = scale.pin_memory()
                    scale = scale.to(
                        device=embeddings.device, dtype=embeddings.dtype
                    )
            return embeddings * scale
        if isinstance(ngram_embedding, Qwen4ExpPinnedHostEmbedding):
            return embeddings
        global_ids = lookup_ids.long()
        local_ids = global_ids
        if ngram_embedding.tp_size > 1:
            start = ngram_embedding.shard_indices.org_vocab_start_index
            end = ngram_embedding.shard_indices.org_vocab_end_index
            in_range = (global_ids >= start) & (global_ids < end)
            local_ids = torch.where(in_range, global_ids - start, 0)
        return embeddings * ngram_embedding.weight_scale[local_ids]


    def _uses_per_row_scale(self) -> bool:
        # The loader can discover a sharded scale tensor after construction.
        # Use the registered buffer as the source of truth so that the forward
        # path stays correct when a scalar placeholder is expanded at load time.
        return self.ngram_embedding.weight_scale.numel() > 1


    def _lookup_ngram_embeddings(self, lookup_ids: torch.Tensor) -> torch.Tensor:
        ngram_embedding = self.ngram_embedding
        if self._uses_per_row_scale() and not isinstance(
            ngram_embedding, Qwen4ExpPinnedHostEmbedding
        ):
            # VocabParallelEmbedding.forward() reduces the local vocab shards.
            # A row scale belongs to the same shard as its weight, so dequantize
            # the local (masked) rows before that reduction. Scaling after the
            # all-reduce makes non-owning TP ranks use their unrelated row 0
            # scale and leaves the replicated PLE output inconsistent.
            embeddings = ngram_embedding._embed_local_shard(lookup_ids)
            embeddings = self._scale_ple_embeddings(embeddings, lookup_ids)
            if (
                ngram_embedding.tp_size > 1
                and not get_attn_tp_context().input_scattered
            ):
                if getattr(self, "etp_reduce_scatter", False):
                    # Unreduced ETP partials: _finish_embedding_lookup
                    # combines them with dp_reduce_scatter_etp instead.
                    pass
                elif ngram_embedding.use_attn_tp_group:
                    embeddings = attn_tp_all_reduce(embeddings)
                else:
                    embeddings = tensor_model_parallel_all_reduce(embeddings)
            return embeddings

        embeddings = ngram_embedding(lookup_ids)
        return self._scale_ple_embeddings(embeddings, lookup_ids)


    def _embed_ngram_ids(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        embeddings = self._lookup_ngram_embeddings(lookup_ids)
        return self._finish_embedding_lookup(
            embeddings, semantic_tokens, forward_batch, physical_tokens
        )


    def _finish_embedding_lookup(
        self,
        embeddings: torch.Tensor,
        semantic_tokens: int,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        if not self.gather_dp_tokens:
            return embeddings
        local_embeddings = embeddings.new_empty(
            (physical_tokens, *embeddings.shape[1:])
        )
        if self.etp_reduce_scatter:
            # Sharded-table partials: sum across the DP group and keep this
            # rank's rows in one collective (replaces all_reduce + dp_scatter).
            dp_reduce_scatter_etp(
                local_embeddings, embeddings.contiguous(), forward_batch
            )
        else:
            dp_scatter(local_embeddings, embeddings.contiguous(), forward_batch)
        return local_embeddings[:semantic_tokens]
