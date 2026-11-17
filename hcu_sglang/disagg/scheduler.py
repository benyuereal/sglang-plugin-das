# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scheduler that manages a tensor parallel GPU worker."""

from __future__ import annotations

def init_disaggregation(self):
    from sglang.srt.managers import scheduler as native
    from hcu_sglang.disagg.utils import resolve_disagg_metadata_config

    self.mm_receiver = None
    self.disagg_prefill_bootstrap_queue = None
    self.disagg_prefill_inflight_queue = None
    self.disagg_decode_prealloc_queue = None
    self.disagg_decode_transfer_queue = None

    self.disaggregation_mode = native.DisaggregationMode(native.get_disagg().disaggregation_mode)
    self.transfer_backend = native.TransferBackend(
        native.get_disagg().disaggregation_transfer_backend
    )

    # Dedicated Gloo group for epoch-tagged StepInfo/MLPSync all-gather.
    # Do not mix recv-control or model-forward collectives into it.
    # A process-group timeout is replica-fatal; no plain-barrier fallback.
    self.dp_scheduler_cpu_group = None
    self._dp_scheduler_epoch = 0
    if (
        self.disaggregation_mode == native.DisaggregationMode.DECODE
        and native.get_parallel().attn_dp_enabled
    ):
        if not self.require_mlp_sync:
            raise RuntimeError("PD Decode DP sync requires require_mlp_sync=True")
        if native.get_parallel().pp_size != 1:
            raise RuntimeError(
                "PD Decode DP sync currently supports pp_size=1 only"
            )
        # if get_parallel().attn_tp_size != 1 or get_parallel().attn_cp_size != 1:
        #     raise RuntimeError(
        #         "PD Decode DP sync currently supports attn_tp_size=1 and "
        #         "attn_cp_size=1 only"
        #     )

        tp_ranks = list(self.tp_group.ranks)
        expected_world = (
            native.get_parallel().attn_dp_size
            * native.get_parallel().attn_tp_size
            * native.get_parallel().attn_cp_size
        )
        default_world = native.torch.distributed.get_world_size()
        if len(tp_ranks) != expected_world or len(tp_ranks) != default_world:
            raise RuntimeError(
                "PD Decode DP sync topology check failed: "
                f"tp_ranks={len(tp_ranks)} expected={expected_world} "
                f"default_world={default_world}. "
                "Supported only on PP1 DP-attention Decode topology."
            )
        expected_ranks = list(range(default_world))
        if tp_ranks != expected_ranks:
            raise RuntimeError(
                "PD Decode DP sync requires tp_group.ranks to be the ordered "
                f"full default world: actual={tp_ranks} "
                f"expected={expected_ranks}"
            )

        from datetime import timedelta

        # Dedicated scheduler-group timeout for PD Decode DP sync.
        timeout_s = 60.0
        self.dp_scheduler_cpu_group = native.torch.distributed.new_group(
            ranks=tp_ranks,
            backend="gloo",
            timeout=timedelta(seconds=timeout_s),
        )
        if native.get_parallel().tp_rank == 0:
            native.logger.info(
                "PD Decode single-clock enabled: dedicated Gloo scheduler "
                "group, world=%s timeout=%.1fs",
                len(tp_ranks),
                timeout_s,
            )

    self._init_waiting_queue_prefetch_runtime()

    # The external-KV abort/cleanup bookkeeping is initialised by the Hygon
    # internal fork's Scheduler.__init__; official v0.5.22 has no such state.
    # This module's prefill hooks add, discard and test these two sets
    # unconditionally, so establish them here for every mode.
    if not hasattr(self, "_external_kv_abort_rids"):
        self._external_kv_abort_rids = set()
    if not hasattr(self, "_external_kv_cleanup_rids"):
        self._external_kv_cleanup_rids = set()

    # In rust-server mode the KV bootstrap registry is already serving on
    # the rust api listener (maybe_init_rust_server runs before this
    # method — the PrefillBootstrapQueue's KVManager below registers to it
    # synchronously, and a failed registration only retries ~60s then logs,
    # leaving every PD request unroutable). Only the ascend config store,
    # which start_disagg_service would otherwise create, is left to do.
    if (
        self.disaggregation_mode == native.DisaggregationMode.PREFILL
        and self._hosts_rust_server()
    ):
        native.maybe_create_ascend_config_store(
            transfer_backend=self.transfer_backend,
        )

    # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
    draft_token_to_kv_pool = (
        self.draft_worker.primary_draft_kv_pool
        if self.draft_worker is not None
        else None
    )

    if self.spec_algorithm.carries_draft_hidden_states():
        # Derive the rank-uniform PD wire schema from config because only the
        # last prefill PP stage owns a draft runner.
        draft_model_config = native.ModelConfig.from_server_args(
            self.server_args,
            model_path=native.get_spec().speculative_draft_model_path,
            model_revision=native.get_spec().speculative_draft_model_revision,
            is_draft_model=True,
        )
        disagg_hidden_size, disagg_hidden_states_dtype = (
            native.get_draft_recurrent_hidden_state_spec_from_config(
                draft_model_config, self.spec_algorithm
            )
        )
    else:
        disagg_hidden_size = 16  # minimal padding size for RDMA
        disagg_hidden_states_dtype = native.torch.float32

    # DSpark PD hidden-state transfer widens the metadata buffers and adds
    # its own receive pool. NULL mode has no PD wire to size, and resolving
    # it there would inspect speculative workers that need not exist.
    metadata_buffer_kwargs = {}
    if self.disaggregation_mode != native.DisaggregationMode.NULL:
        disagg_metadata_config = resolve_disagg_metadata_config(
            hidden_size=disagg_hidden_size,
            hidden_states_dtype=disagg_hidden_states_dtype,
            disaggregation_mode=self.disaggregation_mode,
            transfer_backend=self.transfer_backend,
            spec_algorithm=self.spec_algorithm,
            model_config=self.model_config,
            server_args=self.server_args,
            model_runner=self.tp_worker.model_runner,
            pp_rank=native.get_parallel().pp_rank,
            pp_size=native.get_parallel().pp_size,
            gpu_id=native.get_device().gpu_id,
            max_prefill_tokens=self.max_prefill_tokens,
        )
        disagg_hidden_size = disagg_metadata_config.hidden_size
        disagg_hidden_states_dtype = disagg_metadata_config.hidden_states_dtype
        metadata_buffer_kwargs = disagg_metadata_config.metadata_buffer_kwargs

    # The PD metadata wire schema must match on P and D even when only D
    # enables spec decoding; a seedless prefill writes the invalid sentinel.
    output_dsa_topk_indices_dim = native.get_dsa_seed_metadata_dim(
        self.model_config.hf_config
    )

    if (
        self.disaggregation_mode == native.DisaggregationMode.DECODE
    ):  # *8 headroom for MiniMax-M3; *2 for other models.
        buffer_multiplier = (
            8 if native.is_minimax_sparse(self.model_config.hf_config) else 2
        )
        buffer_size = (self.req_to_token_pool.size) * buffer_multiplier
        self.req_to_metadata_buffer_idx_allocator = native.ReqToMetadataIdxAllocator(
            buffer_size
        )
        self.disagg_metadata_buffers = native.MetadataBuffers(
            buffer_size,
            hidden_size=disagg_hidden_size,
            hidden_states_dtype=disagg_hidden_states_dtype,
            max_sampling_mask_tokens=self.server_args.sampling_mask_max_tokens,
            custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
            kv_checksum_enabled=native.get_disagg().disaggregation_enable_kv_checksum,
            **metadata_buffer_kwargs,
        )

        # The decode requests polling kv cache
        self.disagg_decode_transfer_queue = native.DecodeTransferQueue(
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.disagg_metadata_buffers,
            scheduler=self,
            tree_cache=self.tree_cache,
        )

        # The decode requests pending for pre-allocation
        self.disagg_decode_prealloc_queue = native.DecodePreallocQueue(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.disagg_metadata_buffers,
            scheduler=self,
            transfer_queue=self.disagg_decode_transfer_queue,
            tree_cache=self.tree_cache,
            gpu_id=native.get_device().gpu_id,
            bootstrap_port=native.get_disagg().disaggregation_bootstrap_port,
            max_total_num_tokens=self.max_total_num_tokens,
            num_reserved_decode_tokens=native.get_disagg().num_reserved_decode_tokens,
            transfer_backend=self.transfer_backend,
        )

    elif self.disaggregation_mode == native.DisaggregationMode.PREFILL:
        # *2 for the headroom.
        buffer_size = self.max_running_requests * 2
        self.req_to_metadata_buffer_idx_allocator = native.ReqToMetadataIdxAllocator(
            buffer_size
        )
        self.disagg_metadata_buffers = native.MetadataBuffers(
            buffer_size,
            hidden_size=disagg_hidden_size,
            hidden_states_dtype=disagg_hidden_states_dtype,
            max_sampling_mask_tokens=self.server_args.sampling_mask_max_tokens,
            custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
            kv_checksum_enabled=native.get_disagg().disaggregation_enable_kv_checksum,
            **metadata_buffer_kwargs,
        )

        self.disagg_prefill_bootstrap_queue = native.PrefillBootstrapQueue(
            token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.disagg_metadata_buffers,
            gpu_id=native.get_device().gpu_id,
            bootstrap_port=native.get_disagg().disaggregation_bootstrap_port,
            max_total_num_tokens=self.max_total_num_tokens,
            scheduler=self,
            scheduler_stage_metrics=self.scheduler_stage_metrics,
            transfer_backend=self.transfer_backend,
        )
        # The prefill requests that are in the middle of kv sending
        self.disagg_prefill_inflight_queue: native.List[native.Req] = []
        # Requests with a sent chunk that are not yet on the inflight queue.
        self.disagg_prefill_pending_chunk_rids: native.Set[str] = set()

        self.enable_staging = native.envs.SGLANG_DISAGG_STAGING_BUFFER.get()

    if (
        self.enable_unified_memory
        and self.disaggregation_mode != native.DisaggregationMode.NULL
    ):
        self.token_to_kv_pool_allocator.set_disagg_move_gate(
            native.unified_memory_disagg_move_gate(self)
        )

    # Init mm receiver for EPD disaggregation mode
    if native.get_disagg().language_only and native.get_disagg().encoder_transfer_backend in [
        "zmq_to_scheduler",
        "mooncake",
    ]:
        self.mm_receiver = native.create_mm_receiver(
            self.server_args,
            dtype=self.model_config.dtype,
            hf_config=self.model_config.hf_config,
            scheduler=self,
        )



def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return
    from sglang.srt.managers import scheduler as native
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    def _waiting_queue_prefetch_runtime_default(self):
        import logging

        logging.getLogger(__name__).warning(
            "Scheduler._init_waiting_queue_prefetch_runtime is not provided"
            " by this sglang build; waiting-queue prefetch stays disabled"
        )

    if not hasattr(native.Scheduler, "_init_waiting_queue_prefetch_runtime"):
        native.Scheduler._init_waiting_queue_prefetch_runtime = (
            _waiting_queue_prefetch_runtime_default
        )

    # get_next_disagg_prefill_batch_to_run calls _begin_scheduler_iteration before
    # any mode check. That hook exists only in the Hygon-internal fork, where it
    # drains background cache-prefetch acknowledgements and opens the next prefetch
    # epoch. Neither this repository nor official sglang ships the external cache
    # linker prefetch path, so on official v0.5.22 every PD prefill launch dies at
    # the first scheduler iteration with AttributeError. Install a no-op default at
    # hook-registration time; a host build that provides the method keeps it.
    def _begin_scheduler_iteration_default(self):
        return None

    if not hasattr(native.Scheduler, "_begin_scheduler_iteration"):
        native.Scheduler._begin_scheduler_iteration = (
            _begin_scheduler_iteration_default
        )

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.init_disaggregation",
        init_disaggregation,
        HookType.REPLACE,
    )
