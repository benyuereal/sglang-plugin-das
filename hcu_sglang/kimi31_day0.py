from inspect import signature

from hcu_sglang import configs, jit_overlay_loader, server_args
from hcu_sglang.kernel_compat import around_moe_route_radix_covered
from hcu_sglang.layers.attention.current_dsa.flashmla_backend import get_flashmla_op
from hcu_sglang.layers.quantization import registry
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _flashmla_operation(name):
    operation = get_flashmla_op(name, is_hcu=True)
    accepts_kv_format = "kv_format" in signature(operation).parameters

    def run(self, **kwargs):
        if name == "flash_mla_with_kvcache" and not accepts_kv_format:
            kv_format = kwargs.pop("kv_format", "V32")
            if kv_format != "V32":
                raise ValueError(f"HCU FlashMLA requires V32 KV format, got {kv_format}")
        return operation(**kwargs)

    return run


def _init_nccl_gather(original, self, max_tokens, *, enabled=True, skip_entry_sync=False):
    return original(self, max_tokens, enabled=False, skip_entry_sync=skip_entry_sync)


def register_hooks():
    HookRegistry.register(
        "sglang.kernels.ops.moe.moe_route_radix.covered",
        around_moe_route_radix_covered,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.triton_symm_mem_ag.MultimemAllGatherer.__init__",
        _init_nccl_gather,
        HookType.AROUND,
    )
    jit_overlay_loader.register_hooks()
    configs.register_hooks()
    server_args.register()
    registry.register_hooks()
    registry.install_quantization_method()

    backend = "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend"
    for name in ("flash_mla_sparse_fwd", "flash_mla_with_kvcache", "get_mla_metadata"):
        HookRegistry.register(f"{backend}._call_{name}", _flashmla_operation(name), HookType.REPLACE)
