"""HCU page-major MHA device KV pool."""

from __future__ import annotations

from contextlib import nullcontext
from inspect import signature
from math import prod
from typing import Any, Callable

import torch

from hcu_sglang.compat.aiter import install_aiter_compat
from hcu_sglang.utils.detection import is_hcu

install_aiter_compat()


def _layout_enabled() -> bool:
    from sglang.srt.utils import get_bool_env_var

    # default mirrors hcu_sglang.environ (single source): official layout
    # unless the HCU FA perf-A/B path is explicitly requested.
    return get_bool_env_var("SGLANG_KV_LAYOUT_HCU_FA", default="false")


def _mha_pool_base():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool


class HCUMHATokenToKVPool(_mha_pool_base()):
    """MHA KV pool matching HCU FlashAttention's K/V page layout."""

    def __init__(self, *args, **kwargs):
        initialize = super().__init__
        arguments = signature(initialize).bind(*args, **kwargs)
        arguments.arguments["enable_alt_stream"] = False
        initialize(*arguments.args, **arguments.kwargs)

    def _kv_buffer_shapes(self):
        page_num = (self.size + self.page_size) // self.page_size
        return (
            (page_num, self.head_num, self.page_size, self.head_dim),
            (page_num, self.head_num, self.page_size, self.v_head_dim),
        )

    def _kv_tokens_per_row(self) -> int:
        # K and V are both BHSD: [pages, heads, page, dim].
        # The base layout label may still be NHD; descriptors follow storage.
        return self.page_size

    def _create_buffers_normal(self):
        from sglang.srt.mem_cache.memory_pool import GPU_MEMORY_TYPE_KV_CACHE

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                k_shape, v_shape = self._kv_buffer_shapes()
                self.k_buffer = [
                    torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]

    def _init_data_ptrs_and_strides(self):
        self.k_data_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        pointer_buffers = self._slot_move_pointer_buffers()
        self.data_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in pointer_buffers],
            dtype=torch.uint64,
            device=self.device,
        )
        kv_buffers = [*self.k_buffer, *self.v_buffer]
        kv_strides = [
            self.head_num
            * (self.head_dim if index < len(self.k_buffer) else self.v_head_dim)
            * tensor.dtype.itemsize
            for index, tensor in enumerate(kv_buffers)
        ]
        extra_strides = [
            prod(tensor.shape[1:]) * tensor.dtype.itemsize
            for tensor in pointer_buffers[len(kv_buffers) :]
        ]
        self.data_strides = torch.tensor(
            kv_strides + extra_strides,
            device=self.device,
        )

    def _store_kv_layer(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        page_indices = loc // self.page_size
        offsets = loc % self.page_size
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode() and self.alt_stream is not None:
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_idx][page_indices, :, offsets, :] = cache_k
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_idx][page_indices, :, offsets, :] = cache_v
            current_stream.wait_stream(self.alt_stream)
            return
        self.k_buffer[layer_idx][page_indices, :, offsets, :] = cache_k
        self.v_buffer[layer_idx][page_indices, :, offsets, :] = cache_v

    def _move_kv_cache_impl(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        target_pages = tgt_loc // self.page_size
        target_offsets = tgt_loc % self.page_size
        source_pages = src_loc // self.page_size
        source_offsets = src_loc % self.page_size
        for key_buffer, value_buffer in zip(self.k_buffer, self.v_buffer):
            key_buffer[target_pages, :, target_offsets, :] = key_buffer[
                source_pages, :, source_offsets, :
            ]
            value_buffer[target_pages, :, target_offsets, :] = value_buffer[
                source_pages, :, source_offsets, :
            ]

    def get_cpu_copy(self, indices, mamba_indices=None):
        from sglang.srt.platforms import current_platform

        current_platform.synchronize()
        result = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            result.append([])
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start : start + chunk_size]
                page_indices = chunk_indices // self.page_size
                offsets = chunk_indices % self.page_size
                key = self.k_buffer[layer_id][page_indices, :, offsets, :]
                value = self.v_buffer[layer_id][page_indices, :, offsets, :]
                result[-1].append(
                    [
                        key.to("cpu", non_blocking=True),
                        value.to("cpu", non_blocking=True),
                    ]
                )
        current_platform.synchronize()
        return result

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        from sglang.srt.platforms import current_platform

        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start : start + chunk_size]
                key_cpu, value_cpu = kv_cache_cpu[layer_id][start // chunk_size]
                assert key_cpu.shape[0] == value_cpu.shape[0] == len(chunk_indices)
                key = key_cpu.to(self.k_buffer[0].device, non_blocking=True)
                value = value_cpu.to(self.v_buffer[0].device, non_blocking=True)
                page_indices = chunk_indices // self.page_size
                offsets = chunk_indices % self.page_size
                self.k_buffer[layer_id][page_indices, :, offsets, :] = key
                self.v_buffer[layer_id][page_indices, :, offsets, :] = value
        current_platform.synchronize()


def _replace_mha_pool_class(args: tuple[Any, ...], kwargs: dict[str, Any]):
    if "mha_pool_class" in kwargs:
        kwargs["mha_pool_class"] = HCUMHATokenToKVPool
        return args, kwargs
    if args:
        args = (*args[:-1], HCUMHATokenToKVPool)
    return args, kwargs


def around_build_with_mha_pool(
    original: Callable[..., Any],
    configurator: Any,
    *args,
    **kwargs,
) -> Any:
    if not is_hcu() or not _layout_enabled():
        return original(configurator, *args, **kwargs)
    if getattr(configurator, "kv_cache_dtype_str", None) == "mxfp8":
        return original(configurator, *args, **kwargs)
    if kwargs.get("quant_method") is not None:
        return original(configurator, *args, **kwargs)
    args, kwargs = _replace_mha_pool_class(args, kwargs)
    return original(configurator, *args, **kwargs)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    for method in (
        "_build_mha_kv_pool",
        "_build_hybrid_swa_kv_pool",
        "_build_hybrid_linear_kv_pool",
    ):
        HookRegistry.register(
            f"{target}.{method}",
            around_build_with_mha_pool,
            HookType.AROUND,
        )
