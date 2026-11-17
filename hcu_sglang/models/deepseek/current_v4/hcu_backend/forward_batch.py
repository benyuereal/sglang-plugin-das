"""Extend the mutable forward record with current HCU fields, preserving dataclass copies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional,List,Any
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch as _OfficialForwardBatch

@dataclass
class ForwardBatch(_OfficialForwardBatch):
    orig_seq_lens_cpu: Optional[List[int]] = None
    rms_quant_flag: bool = False
    pd_hidden_capture_layer_ids: Optional[List[int]] = None
    dsa_layer_split_main_kv_page_plan: Optional[Any] = None
    global_cp_num_tokens_cpu: Optional[List[int]] = None
    dp_local_token_index: Optional[int] = None
    cp_local_dp_layout: bool = False
    global_late_layer_num_tokens_cpu: Optional[List[int]] = None
    encoder_swa_row_floor: Optional[torch.Tensor] = None
    encoder_swa_compress_rows: Optional[torch.Tensor] = None
    dp_pure_prefill: bool = False
    qwen4_compact_prefill: bool = False
