# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref

import torch

from hcu_sglang.utils.detection import is_hcu
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_pin_memory_available


_delta_cache = {}


def _source_version(source):
    try:
        return source._version
    except RuntimeError:
        return None


def _mrope_delta_device(mm_input, device):
    source = mm_input.mrope_position_delta
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    key = id(mm_input)
    version = _source_version(source)
    entry = _delta_cache.get(key)
    if (entry is None or entry[0]() is not mm_input
            or entry[1] is not source or entry[2] != version):
        def release(reference):
            current = _delta_cache.get(key)
            if current is not None and current[0] is reference:
                del _delta_cache[key]
        entry = (weakref.ref(mm_input, release), source, version, {})
        _delta_cache[key] = entry
    cached = entry[3].get(target)
    if cached is None:
        pinned = source
        non_blocking = False
        if source.device.type == "cpu" and is_pin_memory_available(target):
            pinned = source if source.is_pinned() else source.pin_memory()
            non_blocking = True
        value = pinned.reshape(-1).to(
            device=target, dtype=torch.int64, non_blocking=non_blocking
        )
        ready = None
        streams = set()
        if target.type == "cuda":
            stream = torch.cuda.current_stream(target)
            ready = torch.cuda.Event()
            ready.record(stream)
            streams.add(stream.cuda_stream)
        cached = (value, pinned, ready, streams)
        entry[3][target] = cached
    value, _, ready, streams = cached
    if ready is not None:
        stream = torch.cuda.current_stream(target)
        if stream.cuda_stream not in streams:
            stream.wait_event(ready)
            streams.add(stream.cuda_stream)
        value.record_stream(stream)
    return value


def around_compute_spec_mrope_positions(
    original, self, model_runner, batch, seq_positions=None
):
    mm_inputs = batch.multimodal_inputs
    if not is_hcu() or all(mm is None for mm in mm_inputs):
        return original(self, model_runner, batch, seq_positions=seq_positions)
    batch_size = self.seq_lens.shape[0]
    if seq_positions is None:
        seq_positions = batch.spec_info.positions
    seq_positions = seq_positions.view(batch_size, -1)
    deltas = torch.zeros(
        (batch_size, 1), dtype=torch.int64, device=model_runner.device
    )
    for i, mm in enumerate(mm_inputs):
        if mm is not None:
            deltas[i] = _mrope_delta_device(mm, model_runner.device)
    self.mrope_positions = (
        (seq_positions + deltas).flatten().unsqueeze(0).repeat(3, 1)
    )


def around_compute_mrope_positions_decode(original, self, model_runner, batch):
    if not is_hcu():
        return original(self, model_runner, batch)
    mm_inputs = batch.multimodal_inputs
    if (get_exec().deterministic.rl_on_policy_target is not None
            or not any(mm is not None for mm in mm_inputs)
            or any(mm is not None and mm.mrope_positions is not None
                   and mm.mrope_positions.shape[1] >= self.seq_lens_cpu[i]
                   for i, mm in enumerate(mm_inputs))):
        return original(self, model_runner, batch)
    deltas = torch.zeros(
        self.seq_lens.shape[0], dtype=torch.int64, device=model_runner.device
    )
    for i, mm in enumerate(mm_inputs):
        if mm is not None:
            deltas[i] = _mrope_delta_device(mm, model_runner.device)
    positions = (deltas - 1) + self.seq_lens.to(torch.int64)
    self.mrope_positions = positions.unsqueeze(0).repeat(3, 1)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    prefix = "sglang.srt.model_executor.forward_batch_info.ForwardBatch."
    HookRegistry.register(
        prefix + "compute_spec_mrope_positions",
        around_compute_spec_mrope_positions,
        HookType.AROUND,
    )
    HookRegistry.register(
        prefix + "_compute_mrope_positions_decode",
        around_compute_mrope_positions_decode,
        HookType.AROUND,
    )
