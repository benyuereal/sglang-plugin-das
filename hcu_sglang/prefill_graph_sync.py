# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.scheduler_components.dp_attn import DPSyncWaitCarry
from dataclasses import dataclass
from typing import Callable, Optional, List
from sglang.srt.managers.scheduler_components import dp_attn as N
from sglang.srt.managers.schedule_batch import ScheduleBatch as _NativeScheduleBatch
from sglang.srt.managers import schedule_batch as SB
from hcu_sglang.encoder_swa_replay import resets_swa_window
from sglang.srt.runtime_context import get_platform
from hcu_sglang.cp_token_counts import folded_replay_rows
from hcu_sglang.utils.detection import is_hcu

_NATIVE_INFO = [N.MLPSyncBatchInfo]


@dataclass
class ScheduleBatch(_NativeScheduleBatch):
    dp_pure_prefill: bool = False
    global_cp_num_tokens: Optional[List[int]] = None
    orig_seq_lens_cpu: Optional[List[int]] = None
    loc_tensor: Optional[N.torch.Tensor] = None

    def prepare_for_extend(self):
        self.forward_mode = SB.ForwardMode.EXTEND

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            self.forward_mode = SB.ForwardMode.DLLM_EXTEND

        # Init tensors
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[r.prefix_len :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_end for r in reqs]
        orig_seq_lens = [max(r.extend_end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [r.prefix_len for r in reqs]
        extend_lens = [r.extend_len for r in reqs]
        extend_logprob_start_lens = [
            SB.compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = SB.is_pin_memory_available(self.device)
        # Stay on pinned CPU; H2D is deferred to forward stream via
        # resolve_forward_inputs.
        pinned_input_ids = SB.flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = SB.torch.tensor(seq_lens, dtype=SB.torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = SB.torch.tensor(seq_lens, dtype=SB.torch.int64)
        orig_seq_lens_tensor = SB.torch.tensor(
            orig_seq_lens, dtype=SB.torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens
        self.loc_tensor = SB.torch.tensor([-1], device=self.device)

        if SB.get_exec().features.enable_encoder_swa_bounded_replay:
            for req in reqs:
                if req.multimodal_inputs is not None and (
                    not get_platform().is_cuda
                    or any(
                        item.modality != SB.Modality.IMAGE
                        for item in req.multimodal_inputs.mm_items
                    )
                ):
                    raise ValueError(
                        "encoder SWA replay supports multimodal inputs only for "
                        "image requests on CUDA"
                    )
                if (
                    req.input_embeds is not None
                    or req.positional_embed_overrides is not None
                ):
                    raise ValueError(
                        "encoder SWA replay does not support input_embeds or "
                        "positional_embed_overrides"
                    )
            self.encoder_swa_reset = [resets_swa_window(r) for r in reqs]
        # Allocate memory
        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = SB.alloc_for_extend(
            self
        )

        # Set fields
        input_embeds = []
        all_replace_embeds: SB.List[SB.torch.Tensor] = []
        all_replace_positions: SB.List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.extend_batch_idx += 1

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # Slice to match extend_input_len — PrefillAdder truncates
                # fill_len/extend_input_len on chunk overflow but not input_embeds.
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_len]
                )

            if req.positional_embed_overrides is not None:
                # Override positions are absolute in the full sequence.
                # Convert to extend-tensor coordinates by subtracting pre_len,
                # then skip any that fall within the cached prefix.
                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_len:
                        continue  # Outside current extend chunk, skip
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)

            # Only calculate cached_tokens once. Once retracted, the 'retracted_stain'
            # flag will always True
            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached

                # Calculate detailed breakdown of cached tokens by source (for HiCache)
                # Only compute once on FIRST chunk - subsequent chunks in chunked prefill
                # would incorrectly count previously computed tokens as cache hits.
                if not req._cache_breakdown_computed:
                    # storage_hit_length is set after pop_prefetch_loaded_span()
                    # returns a completed prefetch.
                    (
                        req.cached_tokens_device,
                        req.cached_tokens_host,
                        req.cached_tokens_storage,
                    ) = SB.split_cached_prefix_by_tier(
                        prefix_len=req.prefix_len,
                        host_hit_len=req.materialized_host_hit_len(),
                        storage_hit_len=req.storage_hit_length,
                        storage_hit_start=req.storage_hit_start,
                        host_hit_is_storage=req.host_hit_is_storage,
                    )
                    req._cache_breakdown_computed = True

                req.already_computed = seq_len
            req.is_retracted = False

            if SB.get_exec().mamba.enable_mamba_extra_buffer:
                track_entry = self._mamba_radix_cache_v2_req_prepare_for_extend(req)
                mamba_track_mask_cpu.append(track_entry.track_mask)
                mamba_track_indices_cpu.append(track_entry.track_index)
                mamba_track_seqlens_cpu.append(track_entry.track_seqlen)

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    req.prefix_len,
                    req.extend_end,
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # Apply logprob_start_len
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_len - extend_logprob_start_lens[i] number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_len
                        - extend_logprob_start_lens[i]
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            # Pin so the non_blocking H2D copy in ForwardBatch.init_new is stream-ordered
            extend_input_logprob_token_ids = SB.torch.tensor(
                extend_input_logprob_token_ids,
                pin_memory=SB.is_pin_memory_available(self.device),
            )
            # Clamp placeholder or out-of-range token IDs (e.g., multimodal hashes)
            # so they stay within the vocab boundary before being sent to GPU.
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        if has_replace_embeds:
            replace_embeds_tensor = SB.torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = SB.torch.tensor(
                all_replace_positions, dtype=SB.torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        self.input_ids = None
        self.prefill_input_ids_cpu = pinned_input_ids
        self.req_pool_indices = req_pool_indices_tensor
        self.req_pool_indices_cpu = req_pool_indices_cpu
        self.orig_seq_lens = orig_seq_lens_tensor
        self.orig_seq_lens_cpu = orig_seq_lens
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            SB.torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, SB.torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, SB.torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = extend_logprob_start_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if SB.get_exec().mamba.enable_mamba_extra_buffer:
            self.mamba_prefill_track_mask_cpu = mamba_track_mask_cpu
            self.mamba_track_seqlens_cpu = mamba_track_seqlens_cpu
            self.mamba_track_indices = SB.torch.tensor(
                mamba_track_indices_cpu,
                dtype=SB.torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = SB.torch.tensor(
                mamba_track_mask_cpu,
                dtype=SB.torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = SB.torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=SB.torch.int64,
                device=self.device,
            )

        # Collect mamba init info for deferred ops on forward stream
        if any(req.kv.holds_mamba for req in reqs):
            self._collect_deferred_mamba_cow_and_clear(reqs)

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SB.SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )




    def convert_decode_to_extend(self):
        """View every decode request as a 1-token extend with its context as
        prefix, making this a plain extend (prefill) batch. Called after the
        DP mlp-sync when a peer rank runs extend, so this rank replays the
        extend graphs; requires prepare_for_decode(); token count unchanged.
        """
        bs = self.batch_size()
        self.forward_mode = SB.ForwardMode.EXTEND
        # Stale residue from this object's life as a prefill batch; the
        # extend-merge path would exclude (silently drop) that req otherwise.
        self.chunked_req = None
        # Also stale residue; None keeps the prefill result path from
        # re-reporting old prefill stats for what is decode work.
        self.prefill_stats = None
        # Also stale residue: decode rows are never fresh placements, so they
        # reset no encoder SWA window and fold no replay rows the vote missed.
        if self.encoder_swa_reset is not None:
            self.encoder_swa_reset = [False] * bs
        # A 1-token extend's position is arange(prefix, prefix + 1), so prefix
        # must be seq_len - 1; output_ids trails the row by one or zero and
        # cannot stand in for it. Rows past bs are a beam tail, not requests.
        seq_lens = self.seq_lens_cpu[:bs].tolist()
        for req, seq_len in zip(self.reqs, seq_lens, strict=True):
            req.refresh_fill_ids()
            # end runs one past full_untruncated_fill_ids while output_ids
            # trails; safe only while decoding_reqs suppresses the checkpoint insert.
            # As in mix_with_running, read prefix and extend length off the batch.
            req.extend_end = seq_len

        self.prefix_lens = [seq_len - 1 for seq_len in seq_lens]
        self.extend_lens = [1] * bs
        self.extend_num_tokens = bs
        self.extend_logprob_start_lens = [0] * bs
        self.decoding_reqs = self.reqs
        self.is_prefill_only = False




    def prepare_for_idle(self):
        self.forward_mode = SB.ForwardMode.IDLE
        self.input_ids = SB.torch.empty(0, dtype=SB.torch.int64, device=self.device)
        self.seq_lens = SB.torch.empty(0, dtype=SB.torch.int64, device=self.device)
        self.seq_lens_cpu = SB.torch.empty(0, dtype=SB.torch.int64)
        self.orig_seq_lens = SB.torch.empty(0, dtype=SB.torch.int32, device=self.device)
        self.orig_seq_lens_cpu = None
        self.out_cache_loc = SB.torch.empty(0, dtype=SB.torch.int64, device=self.device)
        self.req_pool_indices = SB.torch.empty(0, dtype=SB.torch.int64, device=self.device)
        self.req_pool_indices_cpu = SB.torch.empty(0, dtype=SB.torch.int64)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SB.SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )




    def prepare_for_decode(self):
        self.forward_mode = SB.ForwardMode.DECODE
        self.mamba_track_seqlens_cpu = None
        self.mamba_prefill_track_mask_cpu = None
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensors so they don't leak into ForwardBatch.
        self.input_embeds = None
        self.replace_embeds = None
        self.replace_positions = None

        self.mamba_cow_src_indices = None
        self.mamba_cow_dst_indices = None
        self.mamba_clear_indices = None

        # Clear context parallel metadata - CP is only for prefill, not decode
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if not self.spec_algorithm.is_none():
            # Spec decoding owns decode preparation (allocation, seq-lens bookkeeping).
            from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

            self.mamba_track_mask_cpu = None
            self.mamba_track_mask_next_cpu = None
            self.mamba_decode_batch_idx_cpu = None
            spec_prepare_for_decode(self)
            return

        # Beam member rows ride this decode batch: append them to the row
        # tensors before allocation so alloc_for_decode covers them too.
        SB.strip_beam_tail(self)
        SB.append_beam_tail(self)

        if self.sampling_info.penalizer_orchestrator.is_required:
            self.cumulate_penalty_output_tokens()

        # input_ids is set at end of previous run_batch (placeholder for
        # overlap; next_token_ids cast for non-overlap).

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory (DSV4-NPU c{4,128}_state alloc lens are computed inside
        # the allocator, triggered from mem_cache/common.py.)
        self.out_cache_loc = SB.alloc_for_decode(self, token_per_req=1)

        for req in self.reqs:
            req.decode_batch_idx += 1

        # New-tensor avoids racing model_worker_batch refs queued for
        # overlap forward.
        self.seq_lens = self.seq_lens + 1
        self.seq_lens_cpu = self.seq_lens_cpu + 1
        self.orig_seq_lens = self.orig_seq_lens + 1
        self.orig_seq_lens_cpu = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
                self.req_pool_indices_cpu,
            )

        if SB.get_exec().mamba.enable_mamba_extra_buffer:
            mamba_track_interval = SB.mamba_track_grid(self.tree_cache.page_size)

            if len(self.reqs) == 0:
                self.mamba_track_indices = SB.torch.empty(
                    (0,), dtype=SB.torch.int64, device=self.device
                )
                self.mamba_track_buffer_indices = []
            else:
                if SB.get_exec().mamba.enable_mamba_extra_buffer_lazy:
                    self.mamba_lazy_prealloc_at_boundary(mamba_track_interval)
                SB.set_mamba_track_indices_from_reqs(self)

            track_remainders_cpu = self.seq_lens_cpu % mamba_track_interval
            track_mask_cpu = track_remainders_cpu == 0
            self.mamba_track_mask_cpu = track_mask_cpu.tolist()
            self.mamba_track_mask_next_cpu = (
                (track_remainders_cpu == mamba_track_interval - 1).tolist()
                if self.enable_overlap
                else None
            )
            # ScheduleBatch.copy() snapshots the list of requests, but the Req
            # objects remain shared. The next overlapped decode can therefore
            # advance their counters before this batch's result is processed.
            self.mamba_decode_batch_idx_cpu = [
                req.decode_batch_idx for req in self.reqs
            ]
            # async H2D
            self.mamba_track_mask = track_mask_cpu.pin_memory().to(
                device=self.device, non_blocking=True
            )




    def filter_batch(
        self,
        reqs_to_exclude: Optional[Union[SB.Req, List[SB.Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        SB.strip_beam_tail(self)
        if keep_indices is None:
            if isinstance(reqs_to_exclude, SB.Req):
                reqs_to_exclude = [reqs_to_exclude]
            elif reqs_to_exclude is None:
                reqs_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished() and self.reqs[i] not in reqs_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests. Stale tensors are left as-is: is_empty()
            # keys off reqs, so callers drop the batch before a forward reads them.
            self.reqs = []
            self.return_hidden_states = False
            self.return_hidden_states_mode = SB.CaptureHiddenMode.NULL
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = SB.torch.tensor(
            keep_indices,
            dtype=SB.torch.int64,
            pin_memory=SB.is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.orig_seq_lens_cpu = None
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None

        if self.input_ids is not None:
            self.input_ids = self.input_ids[keep_indices_device]
        # Optional under no-verify-sync; resolve_seq_lens repopulates before forward.
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]

        self.mamba_track_indices = None
        self.mamba_track_buffer_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_track_seqlens_cpu = None
        self.mamba_prefill_track_mask_cpu = None
        self.mamba_track_mask_cpu = None
        self.mamba_track_mask_next_cpu = None
        self.mamba_decode_batch_idx_cpu = None
        self.mamba_lazy_spec_track_positions_cpu = None
        self.mamba_cow_src_indices = None
        self.mamba_cow_dst_indices = None
        self.mamba_clear_indices = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_grammar = any(req.grammar for req in self.reqs)
        self.return_hidden_states_mode = SB.get_batch_return_hidden_states_mode(self.reqs)
        self.return_hidden_states = self.return_hidden_states_mode.need_capture()

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                new_indices_cpu=keep_indices,
            )




    def merge_batch(self, other: ScheduleBatch):
        SB.strip_beam_tail(self)
        SB.strip_beam_tail(other)
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = SB.torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu = self.encoder_lens_cpu + other.encoder_lens_cpu
        self.req_pool_indices = SB.torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.req_pool_indices_cpu = SB.torch.cat(
            [self.req_pool_indices_cpu, other.req_pool_indices_cpu]
        )
        self.seq_lens = SB.torch.cat([self.seq_lens, other.seq_lens])
        self.orig_seq_lens = SB.torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.orig_seq_lens_cpu = None
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None
        # Cat only when both sides hold a real token tensor; otherwise drop to
        # None and let resolve_forward_inputs rebuild from the merged
        # req_pool_indices. Mismatch arises e.g. with spec_v1, which keeps its
        # tensor while a relay-staged side is None -- there the worker rebuilds.
        if self.input_ids is not None and other.input_ids is not None:
            self.input_ids = SB.torch.cat([self.input_ids, other.input_ids])
        else:
            self.input_ids = None
        # Optional under no-verify-sync; drop the mirror if either side absent.
        if self.seq_lens_cpu is None or other.seq_lens_cpu is None:
            self.seq_lens_cpu = None
        else:
            self.seq_lens_cpu = SB.torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.mamba_track_indices = None
        self.mamba_track_buffer_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_track_seqlens_cpu = None
        self.mamba_prefill_track_mask_cpu = None
        self.mamba_track_mask_cpu = None
        self.mamba_track_mask_next_cpu = None
        self.mamba_decode_batch_idx_cpu = None
        self.mamba_lazy_spec_track_positions_cpu = None
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums = self.top_logprobs_nums + other.top_logprobs_nums
            self.token_ids_logprobs = self.token_ids_logprobs + other.token_ids_logprobs
        elif self.return_logprob:
            self.top_logprobs_nums = self.top_logprobs_nums + [0] * len(other.reqs)
            self.token_ids_logprobs = self.token_ids_logprobs + [None] * len(other.reqs)
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs = self.reqs + other.reqs
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = self.multimodal_inputs + other.multimodal_inputs

        self.return_logprob = self.return_logprob or other.return_logprob
        self.has_grammar = self.has_grammar or other.has_grammar
        self.return_hidden_states_mode = max(
            self.return_hidden_states_mode, other.return_hidden_states_mode
        )
        self.return_hidden_states = self.return_hidden_states_mode.need_capture()
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)




    def copy(self):
        # Only contain fields that will be used by process_batch_result.
        # Shallow-copy the reqs list as a defensive snapshot. filter_batch and
        # merge_batch historically mutated the list in place; they now rebind
        # new lists, but the slice stays so this snapshot never aliases the
        # original.
        return SB.ScheduleBatch(
            reqs=self.reqs[:],
            extend_lens=self.extend_lens,
            prefix_lens=self.prefix_lens,
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            req_pool_indices_cpu=self.req_pool_indices_cpu,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            has_grammar=self.has_grammar,
            return_hidden_states=self.return_hidden_states,
            return_hidden_states_mode=self.return_hidden_states_mode,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            dp_spec_prefill_coordination_metadata=self.dp_spec_prefill_coordination_metadata,
            dp_spec_prefill_coordination_applied=self.dp_spec_prefill_coordination_applied,
            global_num_tokens=self.global_num_tokens,
            dp_pure_prefill=self.dp_pure_prefill,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_decode_cuda_graph=self.can_run_decode_cuda_graph,
            can_run_dp_draft_cuda_graph=self.can_run_dp_draft_cuda_graph,
            can_run_dp_prefill_cuda_graph=self.can_run_dp_prefill_cuda_graph,
            dp_prefill_cuda_graph_max_prefix_len=self.dp_prefill_cuda_graph_max_prefix_len,
            global_cp_num_tokens=self.global_cp_num_tokens,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_buffer_indices=self.mamba_track_buffer_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            mamba_track_seqlens_cpu=self.mamba_track_seqlens_cpu,
            mamba_prefill_track_mask_cpu=self.mamba_prefill_track_mask_cpu,
            mamba_track_mask_cpu=self.mamba_track_mask_cpu,
            mamba_track_mask_next_cpu=self.mamba_track_mask_next_cpu,
            mamba_decode_batch_idx_cpu=self.mamba_decode_batch_idx_cpu,
            mamba_lazy_spec_track_positions_cpu=self.mamba_lazy_spec_track_positions_cpu,
            dp_cooperation_info=self.dp_cooperation_info,
            dp_balance_stats=self.dp_balance_stats,
            prefill_stats=self.prefill_stats,
            fpm_start_time=self.fpm_start_time,
            forward_iter=self.forward_iter,
            launch_ts=self.launch_ts,
            after_idle_gap=self.after_idle_gap,
            split_prefill_start=self.split_prefill_start,
            extend_num_tokens=self.extend_num_tokens,
        )




    def maybe_evict_swa(self):
        if self.tree_cache.supports_swa():
            retain_window = self.tree_cache.swa_retain_window
            # Auxiliary windows check their own cursors and prefix locks.
            has_auxiliary_swa = self.tree_cache.supports_auxiliary_swa()

            release_leaf_lock = SB.envs.SGLANG_OPT_RELEASE_PREFILL_SWA.get() and hasattr(
                self.tree_cache, "release_swa_prefix_lock"
            )

            eviction_interval = max(1, SB.envs.SGLANG_SWA_EVICTION_INTERVAL.get())
            swa_evict_due = self._swa_eviction_trigger(retain_window, eviction_interval)
            self.token_to_kv_pool_allocator.free_group_begin()
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # In overlap scheduler, we cannot evict swa when
                    # req.decode_batch_idx == 0 since the prev extend batch is
                    # still running. `_swa_eviction_trigger` owns the rest.
                    if (
                        req.decode_batch_idx >= 1
                        and req.kv.holds_kv
                        and (has_auxiliary_swa or swa_evict_due(req))
                    ):
                        # The scheduler already gated the primary window,
                        # including backends that use a forward-based cadence.
                        self._evict_swa(
                            req,
                            req.seqlen - 1,
                            eviction_interval=(
                                eviction_interval if has_auxiliary_swa else 1
                            ),
                        )

                    # Past the retain window the request no longer reads its prefill's
                    # SWA; release that part of the tree lock so SWA LRU can reclaim it.
                    if release_leaf_lock and req.decode_batch_idx >= retain_window:
                        self.tree_cache.release_swa_prefix_lock(req)
                elif (
                    self.forward_mode.is_extend()
                    and not self.tree_cache.supports_prefix_sharing()
                ):
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # In chunked prefill case, when the second extend batch is scheduling, the first extend batch is still running, so we cannot evict swa tokens
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - SB.get_schedule().chunked_prefill_size
                                if SB.get_schedule().chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)
            self.token_to_kv_pool_allocator.free_group_end()





@dataclass
class MLPSyncBatchInfo(_NATIVE_INFO[0]):
    cp_num_tokens: int = 0
    global_cp_num_tokens: Optional[List[int]] = None

    def _get_local_tensor(self, device, dtype=N.torch.int64) -> N.torch.Tensor:
        return N.torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_run_decode_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                int(self.can_run_prefill_cuda_graph),
                self.prefill_cuda_graph_max_prefix_len,
                int(self.can_run_draft_cuda_graph),
                int(self.is_health_check),
                self.attention_pairs,
                self.cp_num_tokens,
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=N.torch.int64) -> N.torch.Tensor:
        return N.torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_run_decode_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                N.ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_prefill_cuda_graph
                0,  # prefill_cuda_graph_max_prefix_len
                1,  # can_run_draft_cuda_graph
                0,  # is_health_check
                0,  # attention_pairs
                0,  # cp_num_tokens
            ],
            device=device,
            dtype=dtype,
        )

    def finalize_local(self):
        """Populate gather-derived metadata from the sole attention-DP rank."""
        self.tp0_info_cpu = self._get_local_tensor(device="cpu").view(1, -1)
        self.global_num_tokens = [self.num_tokens]
        self.global_num_tokens_for_logprob = [self.num_tokens_for_logprob]
        self.global_attention_pairs = [self.attention_pairs]
        self.global_cp_num_tokens = [self.cp_num_tokens]
        self.any_health_check = self.is_health_check
        if N._ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = N.DPCooperationInfo.create(
                self.tp0_info_cpu[:, 5].tolist()
            )

    def all_gather(
        self,
        device,
        group: N.torch.distributed.ProcessGroup,
        use_all_reduce: bool = False,
    ):
        sync_start = N.time.perf_counter()
        local_info_tensor = self._get_local_tensor(device=device)
        fallback_tensor = self._get_fallback_tensor(device=device)
        info_width = local_info_tensor.numel()
        # Inactive max_world_size slots must decode as IDLE. repeat() (not
        # expand().contiguous()) so the buffer never aliases fallback_tensor:
        # at world size 1 the expanded view is already contiguous, contiguous()
        # is a no-op, and the masked fallback writes below would then read and
        # write the same storage.
        global_info_tensor = fallback_tensor.repeat(
            self.num_dp_ranks, self.tp_size * self.cp_size, 1
        )

        if use_all_reduce:
            # Admission can expose different WORLD sizes; use fixed global slots.
            global_info_tensor.zero_()
            flat_info = global_info_tensor.view(-1, info_width)
            rank = N.torch.distributed.get_rank(group)
            if 0 <= rank < flat_info.shape[0]:
                flat_info[rank] = local_info_tensor
            N.torch.distributed.all_reduce(
                global_info_tensor,
                op=N.torch.distributed.ReduceOp.SUM,
                group=group,
            )
            missing = flat_info.abs().sum(dim=1) == 0
            flat_info[missing] = fallback_tensor
        else:
            exchange_sync_rows(
                global_info_tensor.flatten(),
                local_info_tensor,
                group=group,
            )

        tp_info = global_info_tensor.view(
            self.num_dp_ranks * self.tp_size * self.cp_size, info_width
        )
        num_ranks_in_tp_info = tp_info.shape[0]
        if device == "cpu":
            tp_active_ranks = N.get_parallel().tp_group.active_ranks_cpu
        else:
            tp_active_ranks = N.get_parallel().tp_group.active_ranks
        if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
            tp_active_ranks = N.torch.ones(
                num_ranks_in_tp_info,
                dtype=tp_active_ranks.dtype,
                device=tp_active_ranks.device,
            )
        tp_info[tp_active_ranks[:num_ranks_in_tp_info] == 0] = fallback_tensor

        # One D2H for every field: each `.item()` / `.tolist()` on a device
        # tensor is its own stream sync. Copy the whole tensor, not the
        # `[:, 0, :]` slice -- that slice is non-contiguous once
        # attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
        tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
        self.tp0_info_cpu = tp0_info_cpu
        self.global_num_tokens = tp0_info_cpu[:, 0].tolist()
        self.global_num_tokens_for_logprob = tp0_info_cpu[:, 1].tolist()
        self.global_cp_num_tokens = tp0_info_cpu[:, 11].tolist()
        if self.cp_size > 1:
            from hcu_sglang.cp_token_counts import normalize_dp_cp_token_counts

            self.global_cp_num_tokens = normalize_dp_cp_token_counts(
                self.global_num_tokens, self.global_cp_num_tokens
            )
        self.can_run_decode_cuda_graph = bool(tp0_info_cpu[:, 2].min())
        self.is_extend_in_batch = bool(tp0_info_cpu[:, 3].max())
        self.can_run_prefill_cuda_graph = bool(tp0_info_cpu[:, 6].min())
        self.prefill_cuda_graph_max_prefix_len = int(tp0_info_cpu[:, 7].max())
        self.can_run_draft_cuda_graph = bool(tp0_info_cpu[:, 8].min())
        self.any_health_check = bool(tp0_info_cpu[:, 9].max())
        self.global_attention_pairs = tp0_info_cpu[:, 10].tolist()
        self.sync_wait_seconds = N.time.perf_counter() - sync_start
        if N._ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = N.DPCooperationInfo.create(
                tp0_info_cpu[:, 5].tolist()
            )


def exchange_sync_rows(
    global_flat: N.torch.Tensor,
    local_row: N.torch.Tensor,
    group: N.torch.distributed.ProcessGroup,
) -> None:
    """Fill ``global_flat`` with every rank's ``local_row`` in group-rank order,
    like ``all_gather_single``. On gloo the all-gather is a ring of world_size - 1
    serial hops; gather + broadcast is one hop plus a log-depth tree. At two
    ranks the ring is the single hop, so it stays."""
    world_size = N.torch.distributed.get_world_size(group)
    if local_row.device.type != "cpu" or world_size <= 2:
        N.all_gather_single(global_flat, local_row, group=group)
        return
    root = N.torch.distributed.get_global_rank(group, 0)
    rows = global_flat.view(world_size, local_row.numel())
    if N.torch.distributed.get_rank(group) == 0:
        N.torch.distributed.gather(
            local_row, gather_list=list(rows.unbind(0)), dst=root, group=group
        )
    else:
        N.torch.distributed.gather(local_row, dst=root, group=group)
    N.torch.distributed.broadcast(global_flat, src=root, group=group)


def _resolve_elastic_world_num_dp_ranks(
    num_dp_ranks: int,
    *,
    group: N.torch.distributed.ProcessGroup,
    local_num_tokens: int,
    local_forward_mode: int,
) -> int:
    if not N.world_dp_gather_enabled():
        return num_dp_ranks

    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    live_num_dp_ranks = N.dp_gather_width()
    effective_ep_size = ElasticEPStateManager.get_effective_ep_size()
    # Query live membership because elastic joins can expand WORLD.
    from hcu_sglang.cp_token_counts import attn_replica_size

    replica_size = attn_replica_size()
    world_size = N.torch.distributed.get_world_size(group)

    if live_num_dp_ranks * replica_size != effective_ep_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync num_dp_ranks is out of sync: "
            f"rank={N.torch.distributed.get_rank(group)} "
            f"live_num_dp_ranks={live_num_dp_ranks} "
            f"effective_ep_size={effective_ep_size} "
            f"attn_replica_size={replica_size} "
            f"world_size={world_size} configured_num_dp_ranks={num_dp_ranks} "
            f"local_num_tokens={local_num_tokens} "
            f"local_forward_mode={local_forward_mode}"
        )
    if effective_ep_size > world_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync EP size exceeds WORLD size: "
            f"rank={N.torch.distributed.get_rank(group)} "
            f"live_num_dp_ranks={live_num_dp_ranks} world_size={world_size} "
            f"effective_ep_size={effective_ep_size} "
            f"attn_replica_size={replica_size}"
        )

    return live_num_dp_ranks


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    draft_require_mlp_tp_gather: Optional[bool] = None,
    skip_global_metadata=False,
):
    if not N.require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
        batch.global_cp_num_tokens = [mlp_sync_info.cp_num_tokens]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
        batch.global_cp_num_tokens = mlp_sync_info.global_cp_num_tokens
    if N.get_parallel().enable_cp_tp_group_sharing:
        batch.global_cp_num_tokens = None

    # Reuse the same all-gather result for a draft model whose A2A backend
    # requires a different local/full token-count representation.
    if draft_require_mlp_tp_gather is not None:
        if draft_require_mlp_tp_gather:
            batch.draft_global_num_tokens = mlp_sync_info.global_num_tokens
            batch.draft_global_num_tokens_for_logprob = (
                mlp_sync_info.global_num_tokens_for_logprob
            )
        else:
            batch.draft_global_num_tokens = [mlp_sync_info.num_tokens]
            batch.draft_global_num_tokens_for_logprob = [
                mlp_sync_info.num_tokens_for_logprob
            ]
    if N.envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get():
        # Fresh counts have not yet been adjusted by the coordination plan.
        batch.dp_spec_prefill_coordination_applied = False
        info = mlp_sync_info.tp0_info_cpu
        batch.dp_spec_prefill_coordination_metadata = (
            mlp_sync_info.global_num_tokens,
            mlp_sync_info.global_num_tokens_for_logprob,
            info[:, 3],
        )
    if not skip_global_metadata:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_decode_cuda_graph = mlp_sync_info.can_run_decode_cuda_graph
    batch.can_run_dp_draft_cuda_graph = mlp_sync_info.can_run_draft_cuda_graph
    batch.can_run_dp_prefill_cuda_graph = mlp_sync_info.can_run_prefill_cuda_graph
    batch.dp_prefill_cuda_graph_max_prefix_len = (
        mlp_sync_info.prefill_cuda_graph_max_prefix_len
    )


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    model_runner: ModelRunner,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    draft_require_mlp_tp_gather: Optional[bool] = None,
    dwdp: bool = False,
    sync_wait_carry: Optional[DPSyncWaitCarry] = None,
):
    parallel = N.get_parallel()
    num_dp_ranks = parallel.num_dp_ranks
    attn_tp_size = parallel.attn_tp_size
    tp_group = parallel.tp_group
    # Check if other DP workers have running batches
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        if (
            local_batch.encoder_swa_reset is not None
            and getattr(model_runner.attn_backend, "folds_encoder_swa_replay", False)
        ):
            num_tokens += folded_replay_rows(local_batch)
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    cp_num_tokens = 0
    if local_batch is not None and local_batch.forward_mode.is_extend():
        from hcu_sglang.cp_token_counts import get_cp_v2_physical_token_count

        cp_num_tokens = get_cp_v2_physical_token_count(
            num_tokens=num_tokens,
            extend_seq_lens=local_batch.extend_lens,
            cp_size=N.get_parallel().attn_cp_size,
        )

    # With a single DP replica and no residual attention-TP/MLP-TP gather,
    # every CP rank receives the same work through the CP control broadcast.
    # The scheduler metadata all-gather is therefore redundant and can hang
    # on HCU CPU process groups during the idle loop.  Reuse the established
    # skip path, which records each rank's local token count as required by CP.
    skip_all_gather = N.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get() or (
        N.get_parallel().num_dp_ranks == 1
        and attn_tp_size == 1
        and not N.require_mlp_tp_gather
    )
    can_run_decode_cuda_graph = N._local_decode_cuda_graph_vote(
        local_batch=local_batch, disable_cuda_graph=disable_cuda_graph
    )
    can_run_draft_cuda_graph = N._spec_input_cuda_graph_compatible(local_batch)
    breakable_prefill = N.check_cuda_graph_backend(N.Phase.PREFILL, N.Backend.BREAKABLE)
    full_prefill = N.check_cuda_graph_backend(N.Phase.PREFILL, N.Backend.FULL)
    coordinated_prefill = breakable_prefill or full_prefill
    prefill_graph_runner = (
        model_runner.prefill_cuda_graph_runner if coordinated_prefill else None
    )
    prefill_cuda_graph_max_prefix_len = (
        max(local_batch.prefix_lens, default=0)
        if full_prefill
        and local_batch is not None
        and local_batch.forward_mode in (N.ForwardMode.EXTEND, N.ForwardMode.MIXED)
        else 0
    )
    can_run_prefill_cuda_graph = N._local_prefill_cuda_graph_vote(
        local_batch=local_batch,
        num_tokens=num_tokens,
        prefill_graph_runner=prefill_graph_runner,
        coordinated_prefill=coordinated_prefill,
        breakable_prefill=breakable_prefill,
        spec_algorithm=model_runner.spec_algorithm,
        model_config=model_runner.model_config,
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    is_health_check = bool(
        local_batch is not None
        and local_batch.reqs
        and all(N.is_health_check_generate_req(req) for req in local_batch.reqs)
    )
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = N.TboDPAttentionPreparer()
    use_world_group = N.world_dp_gather_enabled()
    if use_world_group:
        world = parallel.world_group
        group = N.torch.distributed.group.WORLD
        device = world.device
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or N.envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    if use_world_group:
        num_dp_ranks = _resolve_elastic_world_num_dp_ranks(
            num_dp_ranks,
            group=group,
            local_num_tokens=num_tokens,
            local_forward_mode=local_forward_mode,
        )
    skip_all_gather = N.should_skip_scheduler_all_gather(num_dp_ranks)

    mlp_sync_info = MLPSyncBatchInfo(
        num_dp_ranks=num_dp_ranks,
        tp_size=attn_tp_size,
        cp_size=parallel.attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        cp_num_tokens=cp_num_tokens,
        can_run_decode_cuda_graph=can_run_decode_cuda_graph,
        can_run_draft_cuda_graph=can_run_draft_cuda_graph,
        can_run_prefill_cuda_graph=can_run_prefill_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        prefill_cuda_graph_max_prefix_len=prefill_cuda_graph_max_prefix_len,
        is_health_check=is_health_check,
        # Gathered unconditionally like the other columns, so a per-process
        # metrics flag cannot make ranks disagree on what the tensor holds.
        attention_pairs=N._local_attention_pairs(local_batch),
    )

    if num_dp_ranks == 1:
        mlp_sync_info.finalize_local()
    elif not skip_all_gather:
        mlp_sync_info.all_gather(
            device=device,
            group=group,
            use_all_reduce=use_world_group,
        )

    metadata_ready = mlp_sync_info.tp0_info_cpu is not None
    if metadata_ready:
        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info_cpu[:, 4:6],
            )
        )
    # Decide whether to emit idle batch
    if skip_all_gather:
        # Skip idle batch when attn-dp=1 (and always under DWDP: ranks run independently)
        need_idle_batch = not dwdp and num_dp_ranks > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        N._update_gather_batch(
            batch_to_gather,
            mlp_sync_info,
            N.require_mlp_tp_gather,
            draft_require_mlp_tp_gather,
            skip_global_metadata=not metadata_ready,
        )

    # Set on `local_batch`, not `batch_to_gather`: for PREBUILT batches the
    # scheduler's `last_batch` is the prebuilt batch, not its inner idle batch.
    if local_batch is not None and metadata_ready:
        local_batch.recv_skipper_forward_mode = (
            N.SchedulerRecvSkipper.derive_forward_mode(
                mlp_sync_info.tp0_info_cpu[:, 5].tolist()
            )
        )

    if N._ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info
        local_batch.dp_balance_stats = None
    if N._ENABLE_METRICS_DP_ATTENTION and num_dp_ranks > 1 and not skip_all_gather:
        sync_wait_seconds = mlp_sync_info.sync_wait_seconds
        if sync_wait_carry is not None:
            sync_wait_seconds += sync_wait_carry.seconds
            sync_wait_carry.seconds = 0.0
        # Health-check probes run on an otherwise idle engine; recording them
        # would show maximal imbalance where there is no load.
        if need_idle_batch and not mlp_sync_info.any_health_check:
            local_batch.dp_balance_stats = N.DPBalanceStats.create(
                local_tokens=num_tokens,
                global_num_tokens=mlp_sync_info.global_num_tokens,
                local_attention_pairs=mlp_sync_info.attention_pairs,
                global_attention_pairs=mlp_sync_info.global_attention_pairs,
                sync_wait_seconds=sync_wait_seconds,
            )
        elif local_batch is None and sync_wait_carry is not None:
            # No batch yet: this iteration may gather again, or end in on_idle.
            sync_wait_carry.seconds = sync_wait_seconds

    return local_batch


def around__resolve_elastic_world_num_dp_ranks(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _resolve_elastic_world_num_dp_ranks(*args, **kwargs)


def around__update_gather_batch(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _update_gather_batch(*args, **kwargs)


def around_prepare_mlp_sync_batch_raw(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return prepare_mlp_sync_batch_raw(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn._resolve_elastic_world_num_dp_ranks", around__resolve_elastic_world_num_dp_ranks, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn._update_gather_batch", around__update_gather_batch, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn.prepare_mlp_sync_batch_raw", around_prepare_mlp_sync_batch_raw, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn.MLPSyncBatchInfo", MLPSyncBatchInfo, HookType.REPLACE)
    HookRegistry.register("sglang.srt.managers.schedule_batch.ScheduleBatch", ScheduleBatch, HookType.REPLACE)
