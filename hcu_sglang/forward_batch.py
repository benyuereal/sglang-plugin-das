from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.kv_loc_plan import Cols, KVLocPlan
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.model_executor import forward_batch_info as N
from sglang.srt.distributed.utils import all_gather_single
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.disagg.hidden_state import get_pd_hidden_capture_layer_ids

_is_hcu = is_hcu()


def _pin_host_metadata(device):
    return _is_hcu and N.is_pin_memory_available(device)


class ForwardBatchMethods:
    def init_mlp_sync_metadata(
        self,
        batch: ScheduleBatch,
        device: Union[str, N.torch.device],
        *,
        is_draft_worker: bool = False,
        materialize_device_tensors: bool = True,
    ) -> None:
        """Populate per-rank token counts for DP-attention MLP synchronization."""
        self.dp_spec_prefill_coordination_applied = (
            batch.dp_spec_prefill_coordination_applied
        )
        global_num_tokens_source = batch.global_num_tokens
        global_num_tokens_for_logprob_source = batch.global_num_tokens_for_logprob
        # A DP spec/prefill coordination plan already wrote this phase's counts.
        if (
            is_draft_worker
            and batch.draft_global_num_tokens is not None
            and not batch.dp_spec_prefill_coordination_applied
        ):
            global_num_tokens_source = batch.draft_global_num_tokens
            global_num_tokens_for_logprob_source = (
                batch.draft_global_num_tokens_for_logprob
            )

        if global_num_tokens_source is None:
            return

        assert global_num_tokens_for_logprob_source is not None
        if (
            self.spec_info is not None
            and not batch.dp_spec_prefill_coordination_applied
        ):
            from sglang.srt.speculative.spec_info import spec_scale_global_num_tokens

            global_num_tokens, global_num_tokens_for_logprob = (
                spec_scale_global_num_tokens(
                    self.spec_info,
                    global_num_tokens_source,
                    global_num_tokens_for_logprob_source,
                )
            )
        else:
            global_num_tokens = global_num_tokens_source
            global_num_tokens_for_logprob = global_num_tokens_for_logprob_source

        self.original_global_num_tokens_cpu = global_num_tokens_source
        self.global_num_tokens_cpu = global_num_tokens
        self.global_num_tokens_for_logprob_cpu = global_num_tokens_for_logprob
        if materialize_device_tensors:
            self.global_num_tokens_gpu = N.torch.tensor(
                global_num_tokens,
                dtype=N.torch.int64,
                pin_memory=_pin_host_metadata(device),
            ).to(device, non_blocking=True)
            self.global_num_tokens_for_logprob_gpu = N.torch.tensor(
                global_num_tokens_for_logprob,
                dtype=N.torch.int64,
                pin_memory=_pin_host_metadata(device),
            ).to(device, non_blocking=True)
        else:
            # Draft-extend CUDA Graph replay consumes its own persistent token
            # count buffers.  Building these transient tensors from pageable
            # host memory can block the CPU behind the target stream and leave
            # a launch bubble before replay.  Preserve the CPU values needed
            # for graph selection and materialize the tensors only if the
            # caller ultimately falls back to eager execution.
            self.global_num_tokens_gpu = None
            self.global_num_tokens_for_logprob_gpu = None
        from sglang.srt.layers.cp.utils import cp_v2_dp_token_counts_for_strategy

        self.global_cp_num_tokens_cpu = cp_v2_dp_token_counts_for_strategy(
            batch.global_cp_num_tokens
        )
        self.can_run_decode_cuda_graph = batch.can_run_decode_cuda_graph
        self.can_run_dp_draft_cuda_graph = batch.can_run_dp_draft_cuda_graph



    def materialize_deferred_device_metadata(
        self, device: Union[str, N.torch.device]
    ) -> None:
        """Materialize device metadata deferred by a graph-first prepare path."""
        if (
            self.global_num_token_non_padded is None
            and self.global_num_token_non_padded_cpu is not None
            and N.enable_num_token_non_padded()
        ):
            self.global_num_token_non_padded = N.torch.tensor(
                self.global_num_token_non_padded_cpu, dtype=N.torch.int32, device=device
            )

        if (
            self.global_num_tokens_cpu is not None
            and self.global_num_tokens_gpu is None
        ):
            self.global_num_tokens_gpu = N.torch.tensor(
                self.global_num_tokens_cpu, dtype=N.torch.int64, device=device
            )
        if (
            self.global_num_tokens_for_logprob_cpu is not None
            and self.global_num_tokens_for_logprob_gpu is None
        ):
            self.global_num_tokens_for_logprob_gpu = N.torch.tensor(
                self.global_num_tokens_for_logprob_cpu,
                dtype=N.torch.int64,
                device=device,
            )



    @classmethod
    def init_new(
        cls,
        batch: ScheduleBatch,
        model_runner: ModelRunner,
        *,
        capture_hidden_mode: Optional[N.CaptureHiddenMode] = None,
        return_hidden_states_before_norm: bool,
        extend_position_info=None,
        spec_mrope_positions: Optional[N.torch.Tensor] = None,
        kv_loc_plan: Optional[KVLocPlan] = None,
        write_cols: Optional[Cols] = None,
        defer_device_metadata: bool = False,
    ):
        # init_new must not mutate the input ScheduleBatch; per-forward
        # overrides go through explicit keyword arguments.
        #
        # `kv_loc_plan`: the iteration's plan when this forward is one of
        # several over the same slots (a speculative iteration), `write_cols`
        # the columns of its window this forward writes. Without one, this
        # forward is its own iteration and builds the plan itself.

        pd_hidden_capture_layer_ids = get_pd_hidden_capture_layer_ids(batch.reqs)
        if (
            model_runner.server_args.disaggregation_mode == "prefill"
            and N.get_parallel().attn_cp_size > 1
        ):
            gathered_capture_layers = [None] * N.get_parallel().attn_cp_size
            N.torch.distributed.all_gather_object(
                gathered_capture_layers,
                pd_hidden_capture_layer_ids,
                group=N.get_parallel().attn_cp_group.cpu_group,
            )
            nonempty_capture_layers = [
                [int(x) for x in layer_ids]
                for layer_ids in gathered_capture_layers
                if layer_ids
            ]
            if nonempty_capture_layers:
                expected_capture_layers = nonempty_capture_layers[0]
                if any(
                    layer_ids != expected_capture_layers
                    for layer_ids in nonempty_capture_layers[1:]
                ):
                    raise RuntimeError(
                        "PD hidden capture layers disagree across prefill CP ranks: "
                        f"{gathered_capture_layers}"
                    )
                pd_hidden_capture_layer_ids = expected_capture_layers
        # capture_hidden_mode=None means no override: capture the server's
        # configured maximum so lower-mode requests can share one graph.
        if capture_hidden_mode is None:
            request_capture_hidden_mode = (
                N.CaptureHiddenMode.NULL
                if model_runner.is_draft_worker
                else max(
                    batch.return_hidden_states_mode,
                    N.get_server_return_hidden_states_mode(),
                )
            )
            if pd_hidden_capture_layer_ids:
                request_capture_hidden_mode = max(
                    request_capture_hidden_mode, N.CaptureHiddenMode.FULL
                )
            capture_hidden_mode = N.get_required_capture_hidden_mode(
                request_capture_hidden_mode,
                batch.spec_info,
            )

        # extend-mode-only fields are None on decode/idle
        if batch.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = batch.extend_lens
            extend_prefix_lens = batch.prefix_lens
            extend_logprob_start_lens = batch.extend_logprob_start_lens

        # Mirror the grammars-population behavior previously done in
        # ScheduleBatch.get_model_worker_batch.
        if batch.sampling_info is not None:
            if batch.has_grammar:
                batch.sampling_info.grammars = [req.grammar for req in batch.reqs]
            else:
                batch.sampling_info.grammars = None

        # ScheduleBatch.sampling_info is already swapped to the forward-only
        # copy by Scheduler.run_batch under overlap mode (see save/restore
        # block there). Use it directly.
        seq_lens_cpu = batch.seq_lens_cpu

        # TODO(seq-lens-removal): the whole ScheduleBatch seq_lens family
        # (incl. seq_lens_sum) is slated for removal in favor of kv-committed
        # lengths, so this init_new-time backfill onto the ScheduleBatch is
        # tolerated for now despite the init_new-must-not-mutate-SB rule.
        if batch.seq_lens_sum is None and seq_lens_cpu is not None:
            batch.seq_lens_sum = int(seq_lens_cpu.sum())

        from sglang.srt.model_executor.model_runner import ModelRunner

        # Only a real ModelRunner owns watermark state; lightweight stand-ins skip it.
        watermark_state = (
            model_runner.watermark_state
            if isinstance(model_runner, ModelRunner)
            else None
        )
        ret = cls(
            # Required core inputs
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            # Inputs aliased by reference from ScheduleBatch
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices_cpu=(
                getattr(batch, "req_pool_indices_cpu", None)
                if batch.forward_mode.is_extend_without_speculative()
                else None
            ),
            orig_seq_lens=batch.orig_seq_lens,
            orig_seq_lens_cpu=(
                getattr(batch, "orig_seq_lens_cpu", None)
                if batch.forward_mode.is_extend_without_speculative()
                else None
            ),
            out_cache_loc_dsv4=batch.out_cache_loc_dsv4,
            engram_history=batch.engram_history,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_track_mask=batch.mamba_track_mask,
            mamba_track_seqlens=batch.mamba_track_seqlens,
            mamba_prefill_track_mask_cpu=(
                list(batch.mamba_prefill_track_mask_cpu)
                if batch.mamba_prefill_track_mask_cpu is not None
                else None
            ),
            mamba_track_seqlens_cpu=(
                list(batch.mamba_track_seqlens_cpu)
                if batch.mamba_track_seqlens_cpu is not None
                else None
            ),
            mamba_cow_src_indices=batch.mamba_cow_src_indices,
            mamba_cow_dst_indices=batch.mamba_cow_dst_indices,
            mamba_clear_indices=batch.mamba_clear_indices,
            encoder_lens=batch.encoder_lens,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            input_embeds=batch.input_embeds,
            replace_embeds=batch.replace_embeds,
            replace_positions=batch.replace_positions,
            # Scalar config / flags
            return_logprob=batch.return_logprob,
            is_extend_in_batch=batch.is_extend_in_batch,
            dp_pure_prefill=getattr(batch, "dp_pure_prefill", False),
            can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
            can_run_dp_draft_cuda_graph=batch.can_run_dp_draft_cuda_graph,
            can_run_dp_prefill_cuda_graph=batch.can_run_dp_prefill_cuda_graph,
            dp_prefill_cuda_graph_max_prefix_len=batch.dp_prefill_cuda_graph_max_prefix_len,
            global_forward_mode=batch.global_forward_mode,
            is_prefill_only=batch.is_prefill_only,
            spec_algorithm=batch.spec_algorithm,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=return_hidden_states_before_norm,
            pd_hidden_capture_layer_ids=pd_hidden_capture_layer_ids,
            tbo_split_seq_index=batch.tbo_split_seq_index,
            # Host-side metadata
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            mm_inputs=batch.multimodal_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            lora_ids=[req.lora_id for req in batch.reqs],
            rids=[req.rid for req in batch.reqs],
            watermark_prompt_tail_ids=(
                watermark_state.prompt_tails(batch)
                if watermark_state is not None
                and batch.sampling_info.has_watermark_candidates
                else None
            ),
            watermark_context_hash_history=(
                watermark_state.retracted_context_hashes(batch)
                if watermark_state is not None
                and batch.sampling_info.has_watermark_candidates
                else None
            ),
            # Compound (carry their own device tensors)
            sampling_info=batch.sampling_info,
            spec_info=batch.spec_info,
        )

        # ScheduleBatch and req_to_token keep allocator-global slot identities,
        # and so does the plan's window. An NPU DCP target writes rank-local
        # slots, which its translator maps the window to at `bind` below; keep
        # the global view for the consumers that read it.
        if N._is_npu and N.get_parallel().dcp_enabled and not model_runner.is_draft_worker:
            ret.origin_out_cache_loc = ret.out_cache_loc
        ret._maybe_init_non_generation_fields(batch)

        device = model_runner.device
        pin_host_metadata = _pin_host_metadata(device)

        ret.mm_token_modalities = N._maybe_build_forward_token_modalities(
            model_runner.model_config,
            ret.mm_inputs,
            extend_prefix_lens if isinstance(extend_prefix_lens, list) else None,
            extend_seq_lens if isinstance(extend_seq_lens, list) else None,
            len(batch.input_ids) if batch.input_ids is not None else 0,
            device,
        )

        translator = model_runner.kv_index_translator
        if kv_loc_plan is None:
            kv_loc_plan = translator.own_plan(ret)
        kv_loc_plan.bind(ret, translator, cols=write_cols)

        if N.envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            hashed = N._hash_rids_to_tensor(
                rids=[req.rid for req in batch.reqs],
                device=device,
            )
            bootstrap_room_ids = N._bootstrap_rooms_to_tensor(
                bootstrap_rooms=[req.bootstrap_room for req in batch.reqs],
                device=device,
            )
            batch.sampling_info.rids_int = hashed
            batch.sampling_info.bootstrap_room_ids_int = bootstrap_room_ids
            ret.rids_int = hashed
            ret.bootstrap_room_ids_int = bootstrap_room_ids

        if N.envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT.get():
            ret.req_all_ids_flat, ret.req_all_ids_lens = N.compute_req_all_ids_info(
                batch.reqs
            )

        if batch.extend_input_logprob_token_ids is not None:
            extend_input_logprob_token_ids = batch.extend_input_logprob_token_ids
            if (
                pin_host_metadata
                and extend_input_logprob_token_ids.device.type == "cpu"
                and not extend_input_logprob_token_ids.is_pinned()
            ):
                extend_input_logprob_token_ids = (
                    extend_input_logprob_token_ids.pin_memory()
                )
            ret.extend_input_logprob_token_ids_gpu = extend_input_logprob_token_ids.to(
                device, non_blocking=True
            )

        num_tokens = len(batch.input_ids) if batch.input_ids is not None else 0
        if N.enable_num_token_non_padded():
            if not defer_device_metadata:
                ret.global_num_token_non_padded = N.torch.tensor(
                    num_tokens,
                    dtype=N.torch.int32,
                    pin_memory=pin_host_metadata,
                ).to(device, non_blocking=True)
        ret.global_num_token_non_padded_cpu = num_tokens

        ret.init_mlp_sync_metadata(
            batch,
            device,
            is_draft_worker=model_runner.is_draft_worker,
            materialize_device_tensors=not defer_device_metadata,
        )

        if ret.forward_mode.is_idle():
            ret.positions = N.torch.empty((0,), dtype=N.torch.int64, device=device)
            if model_runner.model_config.model_is_mrope:
                ret.mrope_positions = ret.positions.new_empty((3, 0))
            if model_runner.lora_manager is not None:
                model_runner.lora_manager.reset_lora_batch()
            return ret

        # A dLLM denoise pass rewrites a fixed-size block in place. Context
        # encoding uses the ordinary EXTEND position path below.
        if batch.dllm_config is not None and ret.forward_mode.is_dllm_extend():
            block_size = batch.dllm_config.block_size
            positions_dtype = N.torch.int64 if N.is_hip() or N._is_npu else N.torch.int32
            ret.positions = N.torch.tensor(
                [
                    i
                    for block_offset in (req.dllm_block_offset for req in batch.reqs)
                    for i in range(block_offset, block_offset + block_size)
                ],
                dtype=positions_dtype,
            ).to(device, non_blocking=True)
        elif (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions

        # Init position information
        if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
            if ret.positions is None:
                ret.positions = N.clamp_position(batch.seq_lens)
        else:
            if isinstance(extend_seq_lens, list):
                # Main path: H2D from host lists; populate *_cpu mirrors.
                assert isinstance(extend_prefix_lens, list)
                pin_memory = N.is_pin_memory_available(device)
                ret.extend_seq_lens = N.torch.tensor(
                    extend_seq_lens, dtype=N.torch.int32, pin_memory=pin_memory
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens = N.torch.tensor(
                    extend_prefix_lens, dtype=N.torch.int32, pin_memory=pin_memory
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens_cpu = list(extend_prefix_lens)
                ret.extend_seq_lens_cpu = list(extend_seq_lens)
            else:
                # gpu_only: device tensors handed in directly; leave *_cpu unset.
                assert isinstance(extend_seq_lens, N.torch.Tensor)
                ret.extend_seq_lens = extend_seq_lens
                ret.extend_prefix_lens = extend_prefix_lens
            ret.extend_num_tokens = batch.extend_num_tokens
            if extend_position_info is None:
                positions, ret.extend_start_loc = N.compute_position(
                    model_runner.prefill_attention_backend_str,
                    ret.extend_prefix_lens,
                    ret.extend_seq_lens,
                    ret.extend_num_tokens,
                )
            else:
                positions, ret.extend_start_loc, _ = extend_position_info
            if ret.positions is None:
                ret.positions = positions
            ret.extend_logprob_start_lens_cpu = extend_logprob_start_lens

        if model_runner.ngram_embedding_manager.enabled:
            ret._init_ngram_embedding_info(batch, device)

        if model_runner.model_config.model_is_mrope:
            if spec_mrope_positions is not None:
                ret.mrope_positions = spec_mrope_positions
            elif (
                extend_position_info is not None and extend_position_info[2] is not None
            ):
                ret.mrope_positions = extend_position_info[2]
            elif (
                ret.spec_info is not None
                and getattr(ret.spec_info, "positions", None) is not None
            ):
                ret.compute_spec_mrope_positions(model_runner, batch)
            elif ret.forward_mode.is_draft_extend_v2():
                # Draft-extend tokens are uniform text continuation; reuse the
                # spec mrope path with the input-consistent `ret.positions` rather
                # than the per-request rebuild (which mis-sizes mm requests).
                ret.compute_spec_mrope_positions(
                    model_runner, batch, seq_positions=ret.positions
                )
            else:
                ret._compute_mrope_positions(model_runner, batch)

        # Init lora information
        if (
            model_runner.lora_manager is not None
            and not model_runner.lora_manager.attn_dp_enabled
        ):
            # In the non-LoRA overlap loading case, we fetch LoRA adapters into the memory pool
            # as a batch, right before running the batch
            if not N.get_lora().enable_lora_overlap_loading:
                model_runner.lora_manager.fetch_new_loras(set(ret.lora_ids))

            model_runner.lora_manager.prepare_lora_batch(ret)

        parallel = N.get_parallel()
        if parallel.attn_dcp_size > 1 and ret.out_cache_loc is not None and N.is_hip():
            ret.dcp_kv_mask = (
                ret.positions % parallel.attn_dcp_size == parallel.attn_dcp_rank
            )

        return ret



    def set_local_num_token_non_padded(self, *, sharded: bool) -> None:
        """Derive the LOCAL num_token_non_padded from the invariant GLOBAL scalar.

        A replicated (``sharded=False``) forward keeps the full DP-group count.
        """
        if self.global_num_tokens_cpu is not None:
            # DP / MLP-sync path: per-DP padded width.
            if len(self.global_num_tokens_cpu) > 1:
                num_tokens_per_dp = self.global_num_tokens_cpu[
                    N.dp_slot_in(self.global_num_tokens_cpu)
                ]
            else:
                num_tokens_per_dp = self.global_num_tokens_cpu[0]
        else:
            # Pure TP+SP: local input width.
            num_tokens_per_dp = self._forward_num_tokens()

        self.num_token_non_padded = N.compute_local_num_token_non_padded(
            global_num_token_non_padded=self.global_num_token_non_padded,
            num_tokens_per_dp=num_tokens_per_dp,
            sharded=sharded,
        )



    def moe_num_token_non_padded(self) -> Optional[N.torch.Tensor]:
        """Bound for masking a sparse MoE's padded rows, or None when the MoE
        input is a gathered buffer whose real rows are not a prefix of it."""
        from sglang.srt.layers.cp.utils import enable_cp_v2
        from sglang.srt.layers.layer_boundary import batch_gathers_over_moe_cp
        from sglang.srt.layers.moe.utils import is_moe_input_scattered_across_dp_ranks

        if (
            is_moe_input_scattered_across_dp_ranks()
            and self.attn_cp_metadata is not None
            and self.forward_mode.is_context_parallel_extend()
        ):
            from sglang.srt.layers.cp.base import get_cp_strategy

            if enable_cp_v2():
                strategy = get_cp_strategy()
                if strategy is not None:
                    return strategy.moe_num_token_non_padded(self)
            elif _is_hcu:
                from sglang.srt.layers.attention.dsa.utils import (
                    dsa_use_prefill_cp,
                    is_dsa_prefill_cp_round_robin_split,
                )

                if dsa_use_prefill_cp(self) and is_dsa_prefill_cp_round_robin_split():
                    # Legacy HCU metadata is not InterleaveCPStrategy metadata.
                    # Its real rows still form a prefix of each strided shard.
                    metadata = self.attn_cp_metadata
                    if metadata.moe_local_token_count is None:
                        total = self.global_num_token_non_padded_cpu
                        if total is None:
                            total = sum(self.extend_seq_lens_cpu)
                        parallel = N.get_parallel()
                        local = max(
                            0,
                            (total + parallel.attn_cp_size - 1 - parallel.attn_cp_rank)
                            // parallel.attn_cp_size,
                        )
                        # Fill on device instead of synchronously uploading a CPU scalar.
                        metadata.moe_local_token_count = N.torch.full(
                            (), local, dtype=N.torch.int32, device=self.input_ids.device
                        )
                    return metadata.moe_local_token_count

        if self.num_token_non_padded is None:
            return None

        if is_moe_input_scattered_across_dp_ranks():
            # a2a dispatch, FP4 all-gather and dwdp all route the local shard.
            return self.num_token_non_padded
        if batch_gathers_over_moe_cp(self):
            return None
        # DSA / MLA CP all-gather across attention CP on a prefill, which leaves
        # the real rows zigzag-permuted rather than in a prefix.
        if N.get_parallel().attn_cp_size > 1 and self._moe_input_gathered_across_cp():
            return None
        if N.get_parallel().attn_dp_size != 1:
            # dp_gather concatenates one padded slot per attention-DP rank.
            return None
        # A single attention-DP group all-reduces instead of gathering, so the
        # buffer holds the whole sequence and the GLOBAL count bounds it.
        if not self.attn_tp_sequence_sharded:
            return self.num_token_non_padded
        # None under cuda-graph replay: its static batch carries no GLOBAL count.
        return self.global_num_token_non_padded



    @property
    def logical_forward_mode(self) -> ForwardMode:
        """The mode these rows actually carry.

        ``prepare_mlp_sync_batch`` may relabel a decode batch as a 1-token
        extend so every DP rank presents one shape to the collectives.
        Consumers that need the decode semantics behind the label -- mamba
        state updates, linear-attention gates -- read through it here.
        """
        original = self._original_forward_mode
        if (
            original is not None
            and original.is_decode()
            and self.forward_mode.is_extend()
        ):
            return original
        return self.forward_mode



    def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
        from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer

        # Local imports: module-level CP helper imports here are circular (#27014).
        from sglang.srt.layers.cp.padding import get_cp_padding_align_size
        from sglang.srt.layers.cp.utils import enable_cp_v2

        assert self.global_num_tokens_cpu is not None
        assert self.global_num_tokens_for_logprob_cpu is not None

        self._original_batch_size = self.batch_size
        global_num_tokens = list(self.global_num_tokens_cpu)
        # MegaMoEv2 dispatch is rank-synchronous, so an idle rank still carries the
        # one row _run_mega_routed fabricates for it. HIP-only, False elsewhere.
        mega_moe_idle_materialize = N._mega_moe_materializes_idle_rank(self)
        if mega_moe_idle_materialize:
            global_num_tokens = [1] * len(global_num_tokens)
        sync_group_size = len(global_num_tokens)
        attn_tp_size = N.get_parallel().attn_tp_size

        for i in range(sync_group_size):
            # make sure that the padded length is divisible by attn_tp_size because we may need reduce-scatter across attn_tp dim.
            # there is no reduce-scatter in LM logprob, so we do not need to adjust the padded length for logprob
            global_num_tokens[i] = N.ceil_align(global_num_tokens[i], attn_tp_size)

        # make sure that each rank has the same number of tokens to do collective communication.
        # Zigzag (in-seq-split) CP pads to 2 * attn_cp_size for load balance; other CP modes
        # pad to attn_cp_size; CP off pads nothing (extra padding breaks EAGLE/MTP draft
        # prefill with NaN draft logits, see #23269).
        # FIXME(kpham-sgl): revisit so draft prefill-extend tolerates padded dummy tokens.
        if not enable_cp_v2():
            cp_align_size = get_cp_padding_align_size()
            for i in range(sync_group_size):
                global_num_tokens[i] = N.ceil_align(global_num_tokens[i], cp_align_size)

        dp_padding_mode = N.DpPaddingMode.get_dp_padding_mode(
            self.is_extend_in_batch, global_num_tokens
        )
        from hcu_sglang.models.qwen.qwen4_exp import compact_layout
        compact = compact_layout(self, model_runner, global_num_tokens)
        if compact is not None:
            global_num_tokens = compact
            dp_padding_mode = N.DpPaddingMode.SUM_LEN
        if N._elastic_should_preserve_local_token_counts(
            model_runner=model_runner,
            dp_padding_mode=dp_padding_mode,
            global_num_tokens=global_num_tokens,
        ):
            # Joined ranks require real token counts instead of MAX_LEN padding.
            dp_padding_mode = N.DpPaddingMode.SUM_LEN
        # Prefill CUDA graphs require every DP rank to run the same captured
        # shape. Under SUM_LEN each rank pads to its own local token
        # count and can select a different capture bucket. This mismatches the
        # rank-coupled communication geometry: DP gather/combine uses
        # all_gather_into_tensor / reduce_scatter_tensor, while MoE backends may
        # use A2A dispatch/combine. Force MAX_LEN so every rank pads to the global
        # max and picks the same bucket.
        #
        # Larger prefills fall back to eager and keep the memory-efficient
        # SUM_LEN. global_num_tokens is identical across ranks (all-gathered),
        # so the decision is consistent cluster-wide.
        prefill_cg = N.get_exec().graph.cuda_graph_config.prefill
        if (
            self.can_run_dp_prefill_cuda_graph
            and self.is_extend_in_batch
            and prefill_cg.bs
            and max(global_num_tokens) <= max(prefill_cg.bs)
            and not N.prefill_graph_tolerates_sum_len()
        ):
            dp_padding_mode = N.DpPaddingMode.MAX_LEN
        if self.dp_spec_prefill_coordination_applied:
            # Preserve each rank's forward mode on coordinated steps.
            dp_padding_mode = N.DpPaddingMode.SUM_LEN
        self.dp_padding_mode = dp_padding_mode
        # Read once here, where dp_padding_mode is final: the idle-row branch
        # below runs after another arm may have rewritten self.forward_mode.
        materializes_dummy_extend = mega_moe_idle_materialize or (
            self.is_extend_in_batch and (dp_padding_mode.is_max_len() or self.qwen4_compact_prefill)
        )

        if dp_padding_mode.is_max_len():
            # when DP gather mode is all gather, we will use
            # all_gather_into_tensor to gather hidden states, where transferred
            # tokens should be padded to the same length. We will also use
            # reduce-scatter instead of all-reduce after MLP.
            max_num_tokens = max(global_num_tokens)
            global_num_tokens = [max_num_tokens] * sync_group_size
            buffer_len = max_num_tokens * sync_group_size
        else:
            buffer_len = sum(global_num_tokens)

        num_tokens = global_num_tokens[N.dp_slot_in(global_num_tokens)]

        self.attn_tp_sequence_sharded = model_runner.attn_tp_sequence_sharded(
            num_tokens
        )

        self.global_dp_buffer_len = buffer_len
        self.global_num_tokens_padded_cpu = global_num_tokens
        N.set_dp_buffer_len_from_batch(self)
        N.set_is_extend_in_batch(self.is_extend_in_batch)

        bs = self.batch_size

        if (
            self.forward_mode.is_decode()
            or self.forward_mode.is_target_verify()
            or self.forward_mode.is_draft_extend_v2()
            or self.forward_mode.is_idle()
        ):
            # TARGET_VERIFY counts as extend; hybrid-SSM verify batches keep their
            # verify layout. MTP draft workers ("*E", no mamba layer) count as hybrid.
            hybrid_ssm = N.mambaish_config(model_runner.model_config) is not None or (
                model_runner.is_draft_worker
                and getattr(
                    model_runner.model_config.hf_config,
                    "mtp_hybrid_override_pattern",
                    None,
                )
                is not None
            )
            if (
                hybrid_ssm
                and not self.qwen4_compact_prefill
                and self.spec_info is not None
                and not self.spec_info.is_draft_input()
            ):
                if self.forward_mode.is_idle():
                    self._original_forward_mode = self.forward_mode
                    self.forward_mode = N.ForwardMode.TARGET_VERIFY
                # Invert the spec_scale_global_num_tokens scaling.
                bs = self.batch_size = num_tokens // self.spec_info.num_tokens_per_req
            elif materializes_dummy_extend:
                self._original_forward_mode = self.forward_mode
                self.forward_mode = N.ForwardMode.EXTEND
                # Fabricate a single dummy request covering num_tokens for an
                # empty (idle) rank. Ranks reach this once MAX_LEN is forced for
                # the prefill breakable CUDA graph (idle + prefill), which needs
                # every DP rank to run the same captured shape. The `else`
                # branch handles decode rows padded to a 1-token extend: full
                # attention runs them as real 1-token extends, while the mamba
                # layers read back through the relabel via logical_forward_mode.
                if self.seq_lens.shape[0] == 0:
                    dev = self.seq_lens.device
                    self.extend_num_tokens = num_tokens
                    self.extend_seq_lens = N.torch.tensor(
                        [num_tokens], dtype=N.torch.int32, device=dev
                    )
                    self.extend_prefix_lens = N.torch.zeros(
                        1, dtype=self.seq_lens.dtype, device=dev
                    )
                    self.extend_start_loc = N.torch.zeros(
                        1, dtype=N.torch.int32, device=dev
                    )
                    self.seq_lens = N.torch.tensor(
                        [num_tokens], dtype=self.seq_lens.dtype, device=dev
                    )
                    # orig_seq_lens is not padded by _pad_inputs_to_size, so
                    # fabricate it to match the dummy request (the breakable
                    # prefill CUDA graph runner reads it).
                    self.orig_seq_lens = N.torch.tensor(
                        [num_tokens], dtype=self.orig_seq_lens.dtype, device=dev
                    )
                    self.orig_seq_lens_cpu = None
                    self.seq_lens_sum = int(num_tokens)
                    if self.seq_lens_cpu is not None:
                        self.seq_lens_cpu = N.torch.tensor(
                            [num_tokens], dtype=self.seq_lens.dtype
                        )
                    self.extend_prefix_lens_cpu = [0]
                    self.extend_seq_lens_cpu = [int(num_tokens)]
                    self.extend_logprob_start_lens_cpu = [0]
                    bs = self.batch_size = 1
                    # Keep idle fabricated rows masked. Bump the GLOBAL scalar;
                    # the LOCAL count is derived from it downstream.
                    # (global_num_token_non_padded is None unless
                    # moe_ep_size > 1.)
                    if self._original_forward_mode.is_idle():
                        if self.global_num_token_non_padded is not None:
                            self.global_num_token_non_padded.fill_(0)
                        self.global_num_token_non_padded_cpu = 0
                    else:
                        if self.global_num_token_non_padded is not None:
                            self.global_num_token_non_padded.fill_(num_tokens)
                        self.global_num_token_non_padded_cpu = num_tokens
                else:
                    self.extend_num_tokens = bs
                    self.extend_seq_lens = N.torch.full_like(self.seq_lens, 1)
                    self.extend_prefix_lens = self.seq_lens - 1
                    self.extend_start_loc = N.torch.arange(
                        bs, dtype=N.torch.int32, device=self.seq_lens.device
                    )
                    self.extend_prefix_lens_cpu = self.extend_prefix_lens.cpu().tolist()
                    self.extend_seq_lens_cpu = self.extend_seq_lens.cpu().tolist()
                    self.extend_logprob_start_lens_cpu = self.extend_prefix_lens_cpu
            else:
                if self.spec_info is not None:
                    # Invert the spec_scale_global_num_tokens scaling.
                    bs = self.batch_size = (
                        num_tokens // self.spec_info.num_tokens_per_req
                    )
                else:
                    bs = self.batch_size = num_tokens
        elif self.forward_mode.is_extend():
            self.extend_num_tokens = num_tokens

        # padding
        self._pad_inputs_to_size(model_runner, num_tokens, bs)
        self.global_num_tokens_cpu = global_num_tokens
        self.use_pin_memory = not N._is_cpu
        global_num_tokens_pinned = N.torch.tensor(
            global_num_tokens, pin_memory=self.use_pin_memory
        )
        self.global_num_tokens_gpu.copy_(
            global_num_tokens_pinned, non_blocking=self.use_pin_memory
        )
        # Exchanged here, not in the model forward: every rank reaches this point
        # after reshaping its batch and before it picks a prefill graph or eager.
        if (
            self.is_extend_in_batch
            and sync_group_size > 1
            and not model_runner.is_draft_worker
        ):
            local_late_rows = model_runner.attn_backend.late_layer_num_tokens(self)
            if local_late_rows is not None:
                late_rows = N.torch.empty(sync_group_size, dtype=N.torch.int64)
                all_gather_single(
                    late_rows,
                    N.torch.tensor([local_late_rows], dtype=N.torch.int64),
                    group=N.get_parallel().tp_group.cpu_group,
                )
                self.global_late_layer_num_tokens_cpu = late_rows.tolist()

        TboForwardBatchPreparer.prepare(
            batch=self, is_draft_worker=model_runner.is_draft_worker
        )
        # TODO: The following is added to make sure sub-batch input_ids are padded
        # to the multiple of attn_tp_size. It can likely be removed after this
        # function is refactored and merged into the Scheduler.
        if self.tbo_children:
            for child in self.tbo_children:
                child._pad_inputs_to_size(
                    model_runner, child.tbo_padded_len, child.batch_size
                )



    def prepare_attn_tp_scatter_input(self, model_runner: ModelRunner):
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        # Pure TP+SP has no MLP-sync pass, so stamp the decision here.
        self.attn_tp_sequence_sharded = model_runner.attn_tp_sequence_sharded(
            self._forward_num_tokens()
        )

        attn_tp_context = get_attn_tp_context()
        input_scattered = attn_tp_context.use_input_scattered(self)
        model_sp = N.get_parallel().minimax_opt or N.get_parallel().hy3_sp

        if not input_scattered and not model_sp:
            return

        if model_sp and not self.forward_mode.is_extend():
            return

        if input_scattered:
            assert self.forward_mode.is_extend()

        tokens = self.input_ids.shape[0]
        rank_size = N.get_parallel().tp_size
        tokens_padded = (tokens + rank_size - 1) // rank_size * rank_size
        self._pad_inputs_to_size(model_runner, tokens_padded, self.batch_size)



    def post_forward_mlp_sync_batch(self, logits_output: LogitsProcessorOutput):
        if self._original_forward_mode is not None:
            self.forward_mode = self._original_forward_mode
        if self._original_batch_size is not None:
            self.batch_size = self._original_batch_size
        bs = self.batch_size

        # MLP-sync padding appended dummy rows after the real ones; slice the
        # per-request tensors back so post-forward consumers (seeded sampling,
        # ngram token-table updates) never see the padding. The draft-decode
        # branch below does the same for speculative batches.
        if self.spec_info is None and self._original_num_tokens is not None:
            self.positions = self.positions[: self._original_num_tokens]
            self.seq_lens = self.seq_lens[:bs]
            self.req_pool_indices = self.req_pool_indices[:bs]
            if self.req_pool_indices_cpu is not None:
                self.req_pool_indices_cpu = self.req_pool_indices_cpu[:bs]
            if self.seq_lens_cpu is not None:
                self.seq_lens_cpu = self.seq_lens_cpu[:bs]

        if self.spec_info is not None:
            if self.forward_mode.is_decode():  # draft
                num_tokens = self.hidden_states_backup.shape[0]
                self.positions = self.positions[:num_tokens]
                self.seq_lens = self.seq_lens[:bs]
                self.req_pool_indices = self.req_pool_indices[:bs]
                if self.req_pool_indices_cpu is not None:
                    self.req_pool_indices_cpu = self.req_pool_indices_cpu[:bs]
                if self.seq_lens_cpu is not None:
                    self.seq_lens_cpu = self.seq_lens_cpu[:bs]
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :num_tokens
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
                if getattr(logits_output, "draft_top1_token_ids", None) is not None:
                    logits_output.draft_top1_token_ids = (
                        logits_output.draft_top1_token_ids[:num_tokens]
                    )
                    logits_output.draft_top1_probs = logits_output.draft_top1_probs[
                        :num_tokens
                    ]
            elif self.forward_mode.is_target_verify():  # verify
                num_tokens = bs * self.spec_info.num_tokens_per_req
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :num_tokens
                    ]
                if logits_output.hidden_states is not None:
                    logits_output.hidden_states = logits_output.hidden_states[
                        :num_tokens
                    ]
            elif (
                self.spec_info.spec_input_type == N.SpecInputType.EAGLE_DRAFT_EXTEND
                and not self.forward_mode.is_draft_extend_v2()
            ):
                if self.spec_info.num_correct_drafts is not None:
                    self.spec_info.num_correct_drafts = (
                        self.spec_info.num_correct_drafts[:bs]
                    )
                if self.spec_info.num_accept_tokens is not None:
                    self.spec_info.num_accept_tokens = self.spec_info.num_accept_tokens[
                        :bs
                    ]
                if self.extend_seq_lens is not None:
                    self.extend_seq_lens = self.extend_seq_lens[:bs]
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
                if logits_output.hidden_states is not None:
                    logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_draft_extend_v2():  # draft extend_v2
                bs = bs * self.spec_info.num_tokens_per_req
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_extend() or self.forward_mode.is_idle():
                if self.forward_mode.is_idle():
                    self.positions = self.positions[:bs]
                    self.seq_lens = self.seq_lens[:bs]
                    self.req_pool_indices = self.req_pool_indices[:bs]
                    if self.seq_lens_cpu is not None:
                        self.seq_lens_cpu = self.seq_lens_cpu[:bs]
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:bs]

            if hasattr(self, "hidden_states_backup"):
                self.spec_info.hidden_states = self.hidden_states_backup
            if hasattr(self, "output_cache_loc_backup"):
                self.out_cache_loc = self.output_cache_loc_backup

        elif self.forward_mode.is_decode() or self.forward_mode.is_idle():
            logits_output.next_token_logits = logits_output.next_token_logits[:bs]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:bs]
        elif self.forward_mode.is_extend():
            num_tokens = self._original_num_tokens
            logits_output.next_token_logits = logits_output.next_token_logits[
                :num_tokens
            ]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]



def register_hooks():
    if not is_hcu():
        return
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.forward_batch import ForwardBatch
    for name in ('init_mlp_sync_metadata', 'init_new', 'logical_forward_mode', 'materialize_deferred_device_metadata', 'moe_num_token_non_padded', 'post_forward_mlp_sync_batch', 'prepare_attn_tp_scatter_input', 'prepare_mlp_sync_batch', 'set_local_num_token_non_padded'):
        setattr(ForwardBatch, name, ForwardBatchMethods.__dict__[name])
