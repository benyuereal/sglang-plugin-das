from types import SimpleNamespace

import pytest

from hcu_sglang import server_args_hooks as H
from hcu_sglang.environ import register_envs
from sglang.srt.arg_groups import attention_hook, overrides
from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch


register_envs()
TARGET = "sglang.srt.arg_groups.overrides._attention_backend_fa3_fp8_fallback"


@pytest.mark.parametrize("hcu,arch,model,page,dtype,layout,backend,kv,expected", [
    (True, "gfx936", "Qwen4ExpForConditionalGeneration", 64, "bfloat16", True, "fa3", "fp8_e5m2", "fa3"),
    (True, "gfx936", "Qwen4ExpForCausalLMMTP", 64, "bfloat16", True, "fa3", "fp8_e5m2", "fa3"),
    (False, "gfx936", "Qwen4ExpForConditionalGeneration", 64, "bfloat16", True, "fa3", "fp8_e5m2", "triton"),
    (True, "gfx938", "Qwen4ExpForConditionalGeneration", 64, "bfloat16", True, "fa3", "fp8_e5m2", "triton"),
    (True, "gfx936", "DeepseekV4ForCausalLM", 64, "bfloat16", True, "fa3", "fp8_e5m2", "triton"),
    (True, "gfx936", "Qwen4ExpForConditionalGeneration", 16, "bfloat16", True, "fa3", "fp8_e5m2", "triton"),
    (True, "gfx936", "Qwen4ExpForConditionalGeneration", 64, "float16", True, "fa3", "fp8_e5m2", "triton"),
    (True, "gfx936", "Qwen4ExpForConditionalGeneration", 64, "bfloat16", False, "fa3", "fp8_e5m2", "triton"),
    (True, "gfx936", "Qwen4ExpForConditionalGeneration", 64, "bfloat16", True, "triton", "fp8_e5m2", "triton"),
    (True, "gfx936", "Qwen4ExpForConditionalGeneration", 64, "bfloat16", True, "fa3", "fp8_e4m3", "fa3"),
])
def test_registered_pass_updates_preimported_consumer_without_mutating_inputs(
        monkeypatch, hcu, arch, model, page, dtype, layout, backend, kv, expected):
    from hcu_sglang.utils import detection
    from sglang.srt.environ import envs

    monkeypatch.setattr(H, "is_hcu", lambda: hcu)
    monkeypatch.setattr(detection, "get_gcn_arch", lambda: arch)
    monkeypatch.setattr(envs.SGLANG_KV_LAYOUT_HCU_FA, "get", lambda: layout)
    monkeypatch.setattr(H, "_RESOLUTION_REGISTERED", True)
    registered = []
    monkeypatch.setattr(HookRegistry, "register", lambda *args: registered.append(args))
    H.register()
    original = overrides._attention_backend_fa3_fp8_fallback
    assert attention_hook._attention_backend_fa3_fp8_fallback is original
    record = SimpleNamespace(attention_backend=backend, kv_cache_dtype=kv,
        page_size=page, dtype=dtype, _resolved_overrides=[],
        _model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[model])))
    try:
        for target, hook, kind in registered:
            if target == TARGET:
                HookRegistry._apply_target(target, [(kind, hook, None)])
        attention_hook.run_post_process_pass(record, attention_hook._attention_backend_fa3_fp8_fallback)
        assert overrides.resolved_view(record).attention_backend == expected
        assert record.attention_backend == backend
        assert record.kv_cache_dtype == kv
        assert record.page_size == page
    finally:
        wrapped = overrides._attention_backend_fa3_fp8_fallback
        _propagate_patch(wrapped, original, overrides)
        overrides._attention_backend_fa3_fp8_fallback = original


def test_fa3_gate_preserves_current_wrapped_fallback(monkeypatch):
    monkeypatch.setattr(H, "is_hcu", lambda: False)
    marker = {"attention_backend": "current-adapter"}
    view = SimpleNamespace(attention_backend="fa3", kv_cache_dtype="fp8_e5m2")
    assert H.around_qwen4_fa3_fp8_fallback(lambda value: marker, view) is marker
