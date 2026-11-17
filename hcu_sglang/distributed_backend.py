# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/parallel_state.py

from __future__ import annotations

from sglang.srt.distributed import parallel_state as P
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.config import resolved_custom_all_reduce_backend

_CUSTOM_ALL_REDUCE_BACKEND = "auto"
use_quick_custom_allreduce = get_bool_env_var(
    "SGLANG_USE_QUICK_CUSTOM_ALLREDUCE", default="false"
)


def coordinator_init(
    self,
    group_ranks: P.List[P.List[int]],
    local_rank: int,
    torch_distributed_backend: P.Union[str, P.Backend],
    use_pynccl: bool,
    use_mscclpp: bool,
    use_custom_allreduce: bool,
    use_torch_symm_mem_all_reduce: bool,
    use_hpu_communicator: bool,
    use_xpu_communicator: bool,
    use_npu_communicator: bool,
    use_message_queue_broadcaster: bool = False,
    group_name: P.Optional[str] = None,
    gloo_timeout: P.timedelta = P.timedelta(seconds=120 * 60),
    recovered_rank: bool = False,
    rank_offset: int = 0,
    max_world_size: P.Optional[int] = None,
    custom_all_reduce_backend: str = "auto",
):
    # Set group info
    group_name = group_name or "anonymous"
    self.unique_name = P._get_unique_name(group_name)
    P._register_group(self)

    # Set rank info
    self.rank = P.torch.distributed.get_rank()
    # Joiner group ranks are local; shift them into global rank space.
    if rank_offset > 0:
        group_ranks = [[r + rank_offset for r in ranks] for ranks in group_ranks]
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None
    # Which FlashInfer fusion workspace this group owns, or None when the
    # group is not eligible for the allreduce-only kAllReduce path. Stamped
    # by _tag_groups_for_flashinfer_allreduce_only() after group init.
    self._fi_workspace_hint: P.Optional[str] = None
    self.local_size = P.get_int_env_var("LOCAL_SIZE", 0)
    # Scale cohorts create these groups without the serving ranks.
    use_local_synchronization = rank_offset > 0 and not recovered_rank

    if P.is_cuda_alike():
        device_id = (
            0 if P.envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() else local_rank
        )
        self.device = P.torch.device(f"cuda:{device_id}")
    elif P._is_npu:
        self.device = P.torch.device(f"npu:{local_rank}")
    elif P._is_xpu:
        self.device = P.torch.device(f"xpu:{local_rank}")
    elif P._is_musa:
        self.device = P.torch.device(f"musa:{local_rank}")
    else:
        self.device = P.torch.device("cpu")
    self.device_module = P.torch.get_device_module(self.device)

    for ranks in group_ranks:
        subgroup_timeout = P._MODEL_PARALLEL_GROUP_TIMEOUT
        if "mooncake" in torch_distributed_backend:
            from mooncake.pg import MooncakeBackendOptions

            pg_active_size = len(ranks)
            if max_world_size is not None:
                assert max_world_size >= len(ranks), (
                    f"max_world_size ({max_world_size}) must be >= "
                    f"group size ({len(ranks)})"
                )
                pg_active_size = max_world_size

            pg_active_ranks = P.torch.zeros(
                pg_active_size, dtype=P.torch.int32, device=self.device
            )
            pg_active_ranks[: len(ranks)] = 1
            pg_active_ranks_cpu = P.torch.zeros(pg_active_size, dtype=P.torch.int32)
            pg_active_ranks_cpu[: len(ranks)] = 1

            if max_world_size is not None:
                dev_opts = MooncakeBackendOptions(
                    pg_active_ranks, recovered_rank, max_world_size
                )
                cpu_opts = MooncakeBackendOptions(
                    pg_active_ranks_cpu, recovered_rank, max_world_size
                )
            else:
                dev_opts = MooncakeBackendOptions(pg_active_ranks, recovered_rank)
                cpu_opts = MooncakeBackendOptions(
                    pg_active_ranks_cpu, recovered_rank
                )

            active_ranks = pg_active_ranks[: len(ranks)]
            active_ranks_cpu = pg_active_ranks_cpu[: len(ranks)]
            device_group = P.torch.distributed.new_group(
                ranks,
                backend="mooncake",
                pg_options=dev_opts,
                timeout=subgroup_timeout,
                group_desc=f"{group_name}:device",
                use_local_synchronization=use_local_synchronization,
            )
            cpu_group = P.torch.distributed.new_group(
                ranks,
                backend="mooncake-cpu",
                pg_options=cpu_opts,
                timeout=subgroup_timeout,
                group_desc=f"{group_name}:cpu",
                use_local_synchronization=use_local_synchronization,
            )
        else:
            active_ranks = P.torch.ones(
                len(ranks), dtype=P.torch.int32, device=self.device
            )
            active_ranks_cpu = P.torch.ones(len(ranks), dtype=P.torch.int32)
            pg_options = P.get_torch_distributed_pg_options(group_name)
            device_group = P.torch.distributed.new_group(
                ranks,
                backend=torch_distributed_backend,
                pg_options=pg_options,
                timeout=subgroup_timeout,
                group_desc=f"{group_name}:device",
                use_local_synchronization=use_local_synchronization,
            )
            # a group with `gloo` backend, to allow direct coordination
            # between processes through the CPU.
            cpu_group = P.torch.distributed.new_group(
                ranks,
                backend="gloo",
                timeout=gloo_timeout,
                group_desc=f"{group_name}:cpu",
                use_local_synchronization=use_local_synchronization,
            )
        if self.rank in ranks:
            self.ranks = ranks
            self.world_size = len(ranks)
            self.rank_in_group = ranks.index(self.rank)
            self.device_group = device_group
            self.cpu_group = cpu_group
            self.active_ranks = active_ranks
            self.active_ranks_cpu = active_ranks_cpu

    assert self.cpu_group is not None
    assert self.device_group is not None

    # Import communicators
    self.use_pynccl = use_pynccl
    self.use_mscclpp = use_mscclpp
    self.use_custom_allreduce = use_custom_allreduce
    self.custom_all_reduce_backend = custom_all_reduce_backend
    self.use_torch_symm_mem_all_reduce = use_torch_symm_mem_all_reduce
    self.use_hpu_communicator = use_hpu_communicator
    self.use_xpu_communicator = use_xpu_communicator
    self.use_npu_communicator = use_npu_communicator
    self.use_message_queue_broadcaster = use_message_queue_broadcaster

    # Lazy import to avoid documentation build error
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        dispatch_custom_allreduce,
    )
    from sglang.srt.distributed.device_communicators.pymscclpp import (
        PyMscclppCommunicator,
    )
    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        debug_check_symmetric_mempool,
        is_symmetric_memory_enabled,
        use_symmetric_memory,
    )
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric

    self.is_symmetric_memory_enabled = is_symmetric_memory_enabled
    self.use_symmetric_memory = use_symmetric_memory
    self.is_allocation_symmetric = is_allocation_symmetric
    self.debug_check_symmetric_mempool = debug_check_symmetric_mempool
    if P.is_hip():
        from sglang.srt.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
            qr_rocm_arch_available,
        )

    self.pynccl_comm: P.Optional[PyNcclCommunicator] = None
    if use_pynccl and self.world_size > 1:
        self.pynccl_comm = PyNcclCommunicator(
            group=self.cpu_group,
            device=self.device,
            is_symmetric_memory_enabled=self.is_symmetric_memory_enabled(),
        )

    self.pymscclpp_comm: P.Optional[PyMscclppCommunicator] = None
    if use_mscclpp and self.world_size > 1:
        self.pymscclpp_comm = PyMscclppCommunicator(
            group=self.cpu_group,
            device=self.device,
            group_name=group_name,
        )

    self.ca_comm: P.Optional[P.Any] = None
    self.qr_comm: P.Optional[QuickAllReduce] = None

    self.pcie_ipc_comm: P.Optional[P.Any] = None
    from sglang.srt.distributed.device_communicators.pcie_ipc_ar import (
        PcieIpcCommunicator,
        eligible_group,
    )

    if eligible_group(
        group_name=group_name,
        world_size=self.world_size,
        deterministic=self._deterministic_collectives_enabled(),
    ):
        try:
            # The IPC handshake needs the CUDA (NCCL) group, not the CPU one.
            # Autotuning is the other way round: it rendezvouses on the host.
            self.pcie_ipc_comm = PcieIpcCommunicator(
                group=self.device_group,
                device=self.device,
                cpu_group=self.cpu_group,
            )
        except Exception as e:
            P.logger.warning(f"Setup FlashInfer PCIe-IPC allreduce failed with {e}.")
    ca_backend = self.custom_all_reduce_backend
    aiter_transport_env = P.os.getenv("AITER_AR_TRANSPORT", "ipc").lower()
    strict_fabric = ca_backend == "aiter" and aiter_transport_env == "fabric"
    if use_custom_allreduce and self.world_size > 1 and ca_backend != "off":
        # Initialize a custom fast all-reduce implementation.
        try:
            CAClass = dispatch_custom_allreduce(
                group=self.cpu_group,
                device=self.device,
                backend=ca_backend,
            )
            if CAClass is not None:
                self.ca_comm = CAClass(
                    group=self.cpu_group,
                    device=self.device,
                )
        except Exception as e:
            if strict_fabric:
                P.logger.error(
                    "[AR] Strict Fabric init failed on TP group ranks=%s: %s",
                    self.ranks,
                    e,
                )
                raise
            P.logger.warning(
                f"Setup Custom allreduce failed with {e} "
                f"(backend={ca_backend}, transport={aiter_transport_env}). "
                "Falling back to RCCL/PyNccl. To silence this warning, "
                "specify --disable-custom-all-reduce explicitly."
            )
            self.ca_comm = None

        # aiter+fabric silently landing on disabled=True is treated as a
        # strict-Fabric failure.
        if self.ca_comm is not None and strict_fabric and self.ca_comm.disabled:
            raise RuntimeError(
                f"[AR] Strict Fabric requested but aiter CA is disabled "
                f"(ranks={self.ranks}). AITER_AR_TRANSPORT=fabric must not "
                "silently degrade to RCCL."
            )

        # QuickAllReduce is IPC-only. When aiter is going to run over
        # Fabric/auto, don't let QR construct alongside it.
        suppress_qr = ca_backend in ("aiter", "auto") and aiter_transport_env in (
            "fabric",
            "auto",
        )
        if P.is_hip() and use_quick_custom_allreduce and not suppress_qr:
            try:
                # Initialize a custom quick all-reduce implementation on the
                # HIP stack when rocm >= gfx942. Quick reduce is designed as a
                # complement to custom allreduce.
                # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                if qr_rocm_arch_available():
                    self.qr_comm = QuickAllReduce(
                        group=self.cpu_group, device=self.device
                    )
            except Exception as e:
                P.logger.warning(f"Failed to initialize QuickAllReduce: {e}")

        # Canonical startup log: greppable per-rank so operators can prove
        # Fabric was actually selected.
        requested_transport = (
            aiter_transport_env if ca_backend in ("aiter", "auto") else "n/a"
        )
        selected_transport = (
            getattr(self.ca_comm, "transport", "n/a")
            if self.ca_comm is not None
            else "n/a"
        )
        disabled = True if self.ca_comm is None else self.ca_comm.disabled
        P.logger.info(
            "[AR] custom_all_reduce_backend=%s requested_transport=%s "
            "selected_transport=%s disabled=%s tp_ranks=%s world_size=%s",
            ca_backend,
            requested_transport,
            selected_transport,
            disabled,
            self.ranks,
            self.world_size,
        )
        if (
            ca_backend == "aiter"
            and requested_transport == "fabric"
            and selected_transport != "fabric"
        ):
            P.logger.error(
                "[AR] Requested transport=fabric but selected=%s. "
                "This should not happen under strict-Fabric semantics.",
                selected_transport,
            )
    elif self.world_size > 1 and P.is_hip():
        P.logger.info(
            "[AR] All-reduce call path: NCCL (custom AR disabled, backend=%s)",
            ca_backend,
        )

    self.torch_symm_mem_comm: P.Optional[TorchSymmMemCommunicator] = None
    if self.use_torch_symm_mem_all_reduce and self.world_size > 1:
        self.torch_symm_mem_comm = TorchSymmMemCommunicator(
            group=self.cpu_group,
            device=self.device,
        )

    # Create communicator for other hardware backends
    from sglang.srt.distributed.device_communicators.hpu_communicator import (
        HpuCommunicator,
    )
    from sglang.srt.distributed.device_communicators.npu_communicator import (
        NpuCommunicator,
    )
    from sglang.srt.distributed.device_communicators.xpu_communicator import (
        XpuCommunicator,
    )

    self.hpu_communicator: P.Optional[HpuCommunicator] = None
    if use_hpu_communicator and self.world_size > 1:
        self.hpu_communicator = HpuCommunicator(group=self.device_group)

    self.xpu_communicator: P.Optional[XpuCommunicator] = None
    if use_xpu_communicator and self.world_size > 1:
        self.xpu_communicator = XpuCommunicator(group=self.device_group)

    self.npu_communicator: P.Optional[NpuCommunicator] = None
    if use_npu_communicator and self.world_size > 1:
        self.npu_communicator = NpuCommunicator(group=self.device_group)

    # Create message queue
    from sglang.srt.distributed.device_communicators.shm_broadcast import (
        MessageQueue,
    )

    self.mq_broadcaster: P.Optional[MessageQueue] = None
    if use_message_queue_broadcaster and self.world_size > 1 and not recovered_rank:
        # Recovered ranks create their mq_broadcaster in elastic_ep.py
        self.mq_broadcaster = MessageQueue.create_from_process_group(
            self.cpu_group, 1 << 22, 6
        )


def init_world_group(
    ranks: P.List[int],
    local_rank: int,
    backend: str,
    recovered_rank: bool = False,
    max_world_size: P.Optional[int] = None,
) -> P.GroupCoordinator:
    return P.GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
        recovered_rank=recovered_rank,
        max_world_size=max_world_size,
        custom_all_reduce_backend="off",
    )


def init_model_parallel_group(
    group_ranks: P.List[P.List[int]],
    local_rank: int,
    backend: str,
    use_pynccl: P.Optional[bool] = None,
    use_custom_allreduce: P.Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: P.Optional[str] = None,
    use_mscclpp: P.Optional[bool] = None,
    use_torch_symm_mem_allreduce: P.Optional[bool] = None,
    recovered_rank: bool = False,
    rank_offset: int = 0,
    max_world_size: P.Optional[int] = None,
    custom_all_reduce_backend: P.Optional[str] = None,
) -> P.GroupCoordinator:
    if group_name == "attention_tp" and not getattr(P, "_ATTN_TP_USE_AITER_CUSTOM_COMM", False):
        use_custom_allreduce = False
        custom_all_reduce_backend = "off"
    if use_custom_allreduce is None:
        use_custom_allreduce = P._ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp is None:
        use_mscclpp = P._ENABLE_MSCCLPP
    if use_torch_symm_mem_allreduce is None:
        use_torch_symm_mem_allreduce = P._ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    if custom_all_reduce_backend is None:
        custom_all_reduce_backend = _CUSTOM_ALL_REDUCE_BACKEND
    return P.GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=(
            not (P._is_npu or P._is_xpu or backend == "mooncake")
            if use_pynccl is None
            else use_pynccl
        ),
        use_mscclpp=use_mscclpp,
        use_custom_allreduce=use_custom_allreduce,
        use_torch_symm_mem_all_reduce=use_torch_symm_mem_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        recovered_rank=recovered_rank,
        rank_offset=rank_offset,
        max_world_size=max_world_size,
        custom_all_reduce_backend=custom_all_reduce_backend,
    )


def set_custom_all_reduce_backend(backend: str) -> None:
    global _CUSTOM_ALL_REDUCE_BACKEND
    _CUSTOM_ALL_REDUCE_BACKEND = backend


def get_custom_all_reduce_backend() -> str:
    return _CUSTOM_ALL_REDUCE_BACKEND


def around_coordinator_init(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return coordinator_init(self, *args, **kwargs)


def around_init_world_group(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return init_world_group(*args, **kwargs)


def around_init_model_parallel_group(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return init_model_parallel_group(*args, **kwargs)


def after_set_all_reduce_flags(result):
    if is_hcu():
        from sglang.srt.runtime_context import get_exec
        set_custom_all_reduce_backend(
            resolved_custom_all_reduce_backend(get_exec().comm.disable_custom_all_reduce)
        )
    return result



def get_attn_dp_tp_group():
    group = getattr(P, '_ATTN_DP_TP', None)
    assert group is not None, 'attention DP-TP group is not initialized'
    return group


def around_initialize_model_parallel(original, backend=None, duplicate_tp_group=False,
        duplicate_attn_cp_group=False, enable_symm_mem=False, recovered_rank=False, rank_offset=0, max_world_size=None):
    result = original(backend=backend, duplicate_tp_group=duplicate_tp_group,
        enable_symm_mem=enable_symm_mem, recovered_rank=recovered_rank,
        rank_offset=rank_offset, max_world_size=max_world_size)
    if not is_hcu():
        return result
    parallel = P.get_parallel()
    tp_size = parallel.tp_size
    cp_size = parallel.attn_cp_size
    dp_size = parallel.attn_dp_size
    attn_tp_size = tp_size // cp_size // dp_size
    # The native HCU initializer already owns this group; fill only missing groups.
    if getattr(P, '_ATTN_DP_TP', None) is None:
        if cp_size == 1:
            P._ATTN_DP_TP = P._TP
        elif dp_size == 1:
            P._ATTN_DP_TP = P._ATTN_TP
        else:
            world_size = tp_size * parallel.pp_size
            group_ranks = []
            for tp_group_idx in range(world_size // tp_size):
                tp_base = tp_group_idx * tp_size
                for cp_idx in range(cp_size):
                    ranks = []
                    for dp_idx in range(dp_size):
                        start = tp_base + (dp_idx * cp_size + cp_idx) * attn_tp_size
                        ranks.extend(range(start, start + attn_tp_size))
                    group_ranks.append(ranks)
            backend = backend or P.torch.distributed.get_backend(P.get_world_group().device_group)
            P._ATTN_DP_TP = P.init_model_parallel_group(
                group_ranks, P.get_world_group().local_rank, backend,
                use_custom_allreduce=False, group_name='attention_dp_tp',
                recovered_rank=recovered_rank, rank_offset=rank_offset,
                max_world_size=max_world_size)
    if duplicate_attn_cp_group and getattr(P, '_ATTN_CP_OVERLAP', None) is None:
        _SOURCE_COMM_METHODS['_init_attn_cp_overlap_group'](
            world_size=tp_size * parallel.pp_size, attn_cp_size=cp_size,
            attn_tp_size=attn_tp_size, backend=backend,
            recovered_rank=recovered_rank, rank_offset=rank_offset,
            max_world_size=max_world_size)
    return result


def get_attn_cp_overlap_group():
    return getattr(P, '_ATTN_CP_OVERLAP', None) or P.get_attn_cp_group()


def around_sync_eagle_verify_outputs(original, verify_state):
    if not is_hcu():
        return original(verify_state)
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    parallel = P.get_parallel()
    if not is_dp_attention_enabled() or parallel.attn_dp_size == 1:
        return original(verify_state)
    # Attention DP ranks own different requests; only TP/CP peers share this state.
    for group in (parallel.attn_tp_group, parallel.attn_cp_group):
        if group.world_size > 1:
            group.broadcast(verify_state, src=0)


_SOURCE_COMM_METHODS = {}


def _around_source_method(name):
    def around(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return _SOURCE_COMM_METHODS[name](*args, **kwargs)
    return around


def around_destroy_model_parallel(original):
    if not is_hcu():
        return original()
    from sglang.srt.distributed.bootstrap import reset_parallel_initialised

    reset_parallel_initialised()
    P.get_parallel().clear_stamp()
    dwdp_mgr = P.get_global_dwdp_manager()
    if dwdp_mgr is not None:
        dwdp_mgr.cleanup()
        P.set_global_dwdp_manager(None)
    names = ('_SHARED_EXPERTS_TP', '_TP', '_PP', '_DCP', '_MOE_EP', '_MOE_TP',
        '_MOE_DP', '_ATTN_CP_OVERLAP', '_ATTN_CP', '_ATTN_DP_TP', '_ATTN_TP',
        '_PDMUX_PREFILL_TP_GROUP')
    destroyed = set()
    for name in names:
        group = getattr(P, name, None)
        if group is not None and id(group) not in destroyed:
            group.destroy()
            destroyed.add(id(group))
    for name in names:
        setattr(P, name, None)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if is_hcu():
        import sglang.srt.distributed as distributed

        import types
        from hcu_sglang import distributed_methods

        P.get_bool_env_var = get_bool_env_var
        if not hasattr(P, "_ATTN_CP_OVERLAP"):
            P._ATTN_CP_OVERLAP = None
        if not hasattr(P, "_ATTN_DP_TP"):
            P._ATTN_DP_TP = None
        P._is_hcu = True
        P._ATTN_TP_USE_AITER_CUSTOM_COMM = get_bool_env_var('SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM')
        P._use_fused_reshape_to_float = get_bool_env_var('SGLANG_USE_FUSED_RESHAPE_TO_FLOAT')
        for name in ('_reduce_scatter_tensor', '_maybe_aiter_reduce_scatter',
                '_all_gather_into_tensor', 'all_gather', '_all_to_all_single',
                'all_to_all_single', '_init_attn_cp_overlap_group'):
            fn = getattr(distributed_methods, name)
            _SOURCE_COMM_METHODS[name] = types.FunctionType(
                fn.__code__, vars(P), name, fn.__defaults__)
            _SOURCE_COMM_METHODS[name].__kwdefaults__ = fn.__kwdefaults__
            if name != '_init_attn_cp_overlap_group':
                HookRegistry.register('sglang.srt.distributed.parallel_state.GroupCoordinator.' + name,
                    _around_source_method(name), HookType.AROUND)
        P.get_attn_cp_overlap_group = get_attn_cp_overlap_group
        distributed.get_attn_cp_overlap_group = get_attn_cp_overlap_group
        P.get_attn_dp_tp_group = get_attn_dp_tp_group
        distributed.get_attn_dp_tp_group = get_attn_dp_tp_group
        P.set_custom_all_reduce_backend = set_custom_all_reduce_backend
        P.get_custom_all_reduce_backend = get_custom_all_reduce_backend
        distributed.set_custom_all_reduce_backend = set_custom_all_reduce_backend
        distributed.get_custom_all_reduce_backend = get_custom_all_reduce_backend
        for name in ("get_attn_dp_tp_group", "get_attn_cp_overlap_group",
                "set_custom_all_reduce_backend", "get_custom_all_reduce_backend"):
            if name not in P.__all__:
                P.__all__.append(name)
    for target, wrapper, kind in (
        ("sglang.srt.distributed.parallel_state.initialize_model_parallel", around_initialize_model_parallel, HookType.AROUND),
        ("sglang.srt.distributed.parallel_state.destroy_model_parallel", around_destroy_model_parallel, HookType.AROUND),
        ("sglang.srt.distributed.parallel_state.GroupCoordinator.__init__", around_coordinator_init, HookType.AROUND),
        ("sglang.srt.distributed.parallel_state.init_world_group", around_init_world_group, HookType.AROUND),
        ("sglang.srt.distributed.parallel_state.init_model_parallel_group", around_init_model_parallel_group, HookType.AROUND),
        ("sglang.srt.distributed.bootstrap._set_all_reduce_flags", after_set_all_reduce_flags, HookType.AFTER),
        ("sglang.srt.speculative.eagle_utils._sync_eagle_verify_outputs", around_sync_eagle_verify_outputs, HookType.AROUND),
    ):
        HookRegistry.register(target, wrapper, kind)
