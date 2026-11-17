"""HCU page-major MHA HiCache host pool."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from hcu_sglang.compat.aiter import install_aiter_compat
from hcu_sglang.utils.detection import is_hcu

install_aiter_compat()


def _host_base():
    from sglang.srt.mem_cache.pool_host.base import HostKVCache

    return HostKVCache


class MHATokenToKVPoolHostHCU(_host_base()):
    """Host KV pool matching HCU's page-major FlashAttention layout."""

    def __init__(
        self,
        device_pool: Any,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence[Any] = (),
        pool_label: str = "kv",
    ):
        if layout != "layout_hcu":
            raise ValueError(
                "MHATokenToKVPoolHostHCU requires hicache_mem_layout=layout_hcu, "
                f"got {layout!r}"
            )
        if device_pool.layer_shard_enabled:
            raise ValueError(
                "layout_hcu HiCache does not support layer-sharded KV pools"
            )
        if mtp_draft_device_pools:
            raise ValueError(
                "layout_hcu HiCache does not support packed MTP draft KV pools"
            )
        self.mtp_draft_device_pools = ()
        self.target_layer_num = device_pool.layer_num
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )
        k_by_layer = self.k_buffer.transpose(0, 1)
        v_by_layer = self.v_buffer.transpose(0, 1)
        self.k_data_refs = [k_by_layer[index] for index in range(self.layer_num)]
        self.v_data_refs = [v_by_layer[index] for index in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self) -> int:
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self) -> int:
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        k_shape = (
            self.page_num,
            self.layer_num,
            self.head_num,
            self.page_size,
            self.head_dim,
        )
        v_shape = (
            self.page_num,
            self.layer_num,
            self.head_num,
            self.page_size,
            self.head_dim,
        )
        self.token_stride_size = (
            self.page_size * self.head_num * self.head_dim * self.dtype.itemsize
        )
        allocate = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        kwargs = dict(
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return allocate(k_shape, **kwargs), allocate(v_shape, **kwargs)

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise ValueError("layout_hcu HiCache does not support draft KV pools")
        if io_backend == "kernel":
            from sgl_kernel.kvcacheio import (
                transfer_kv_per_layer_kernel_pf_lf_H2D_hcu,
            )

            transfer_kv_per_layer_kernel_pf_lf_H2D_hcu(
                src_k=self.k_buffer,
                dst_k=device_pool.k_buffer[layer_id],
                src_v=self.v_buffer,
                dst_v=device_pool.v_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                item_size=self.token_stride_size,
                src_layout_dim=self.token_stride_size * self.layer_num,
                page_size=self.page_size,
                layer_id=layer_id,
            )
            return
        if io_backend == "direct":
            from sgl_kernel.kvcacheio import transfer_kv_all_direct_pf_lf_H2D_hcu

            transfer_kv_all_direct_pf_lf_H2D_hcu(
                src_ptrs_k=self.k_buffer,
                src_ptrs_v=self.v_buffer,
                dst_ptrs_k=device_pool.k_buffer,
                dst_ptrs_v=device_pool.v_buffer,
                src_indices=host_indices,
                dst_indices=device_indices,
                start_layer_id=layer_id,
                page_size=self.page_size,
            )
            return
        raise ValueError(f"Unsupported layout_hcu IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
    ):
        if io_backend == "kernel":
            from sgl_kernel.kvcacheio import transfer_kv_all_kernel_lf_pf_D2H_hcu

            transfer_kv_all_kernel_lf_pf_D2H_hcu(
                src_k=device_pool.k_data_ptrs,
                dst_k=self.k_buffer,
                src_v=device_pool.v_data_ptrs,
                dst_v=self.v_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.token_stride_size,
                srt_layout_dim=self.token_stride_size * self.page_num,
                dst_layout_dim=self.token_stride_size * self.layer_num,
                page_size=self.page_size,
                layer_num=self.layer_num,
            )
            return
        if io_backend == "direct":
            from sgl_kernel.kvcacheio import transfer_kv_all_direct_lf_pf_D2H_hcu

            transfer_kv_all_direct_lf_pf_D2H_hcu(
                src_ptrs_k=device_pool.k_buffer,
                src_ptrs_v=device_pool.v_buffer,
                dst_ptrs_k=self.k_buffer,
                dst_ptrs_v=self.v_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                start_layer_id=0,
                page_size=self.page_size,
            )
            return
        raise ValueError(f"Unsupported layout_hcu IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True):
        page = int(index) // self.page_size
        k_page = self.k_buffer[page : page + 1]
        v_page = self.v_buffer[page : page + 1]
        if flat:
            return k_page.flatten(), v_page.flatten()
        return k_page, v_page

    def get_dummy_flat_data_page(self):
        k_shape = (1, self.layer_num, self.head_num, self.page_size, self.head_dim)
        v_shape = (1, self.layer_num, self.head_num, self.page_size, self.head_dim)
        kwargs = dict(dtype=self.dtype, device=self.device, pin_memory=self.pin_memory)
        return (
            torch.zeros(k_shape, **kwargs).flatten(),
            torch.zeros(v_shape, **kwargs).flatten(),
        )

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        k_page, v_page = data_page
        page = int(index) // self.page_size
        self.k_buffer[page : page + 1].copy_(
            k_page.reshape(
                1,
                self.layer_num,
                self.head_num,
                self.page_size,
                self.head_dim,
            )
        )
        self.v_buffer[page : page + 1].copy_(
            v_page.reshape(
                1,
                self.layer_num,
                self.head_num,
                self.page_size,
                self.head_dim,
            )
        )

    def get_split_heads_page_buffer_meta(self, indices, split_factor: int):
        raise NotImplementedError("layout_hcu does not support split-head metadata")

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        k_base = self.k_buffer.data_ptr()
        v_base = self.v_buffer.data_ptr()
        pointers = []
        for token_index in indices.tolist()[:: self.page_size]:
            offset = (
                token_index
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            pointers.extend((k_base + offset, v_base + offset))
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        return pointers, [element_size] * len(pointers)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        page_stride = (
            self.layer_num
            * self.head_num
            * self.page_size
            * self.head_dim
            * self.dtype.itemsize
        )
        return (
            self.k_buffer.data_ptr() % page_size_bytes == 0
            and self.v_buffer.data_ptr() % page_size_bytes == 0
            and page_stride % page_size_bytes == 0
        )


def around_get_mha_host_pool_cls(
    original: Callable[..., type],
    device_pool: Any,
    layout: str | None = None,
) -> type:
    if not is_hcu():
        return original(device_pool)
    if layout is None:
        from sglang.srt.runtime_context import get_memory

        layout = get_memory().hicache_mem_layout
    if layout == "layout_hcu":
        return MHATokenToKVPoolHostHCU
    return original(device_pool)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mha.get_mha_host_pool_cls",
        around_get_mha_host_pool_cls,
        HookType.AROUND,
    )
