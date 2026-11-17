from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace

from hcu_sglang.utils.detection import is_hcu
from sglang.srt.mem_cache.cache_init_params import CacheInitParams as NativeCacheInitParams

_SCHEDULER = ContextVar('hcu_cache_scheduler', default=None)


@dataclass
class CacheInitParams(NativeCacheInitParams):
    swa_gates_prefix_match: bool = True
    decoder_swa_bounded_replay: bool = False
    draft_swa_window: int = 0

    @property
    def swa_retain_window(self):
        if self.sliding_window_size is None:
            return None
        return max(self.sliding_window_size, self.draft_swa_window)


def swa_retain_window(self):
    if self.sliding_window_size is None:
        return None
    return max(self.sliding_window_size, self.draft_swa_window)


def around_scheduler_init(original, scheduler, *args, **kwargs):
    if not is_hcu():
        return original(scheduler, *args, **kwargs)
    token = _SCHEDULER.set(scheduler)
    try:
        return original(scheduler, *args, **kwargs)
    finally:
        _SCHEDULER.reset(token)


def around_build_kv_cache(original, **kwargs):
    if not is_hcu():
        return original(**kwargs)
    from hcu_sglang.mem_cache.cache_builder import build_kv_cache

    scheduler = _SCHEDULER.get()
    if 'draft_swa_window' not in kwargs and scheduler is not None:
        draft = scheduler.draft_worker
        kwargs['draft_swa_window'] = getattr(draft, 'draft_swa_window', 0)
    return build_kv_cache(**kwargs)


def around_unified_init(original, cache, params):
    if not is_hcu():
        return original(cache, params)
    cache.decoder_swa_bounded_replay = params.decoder_swa_bounded_replay
    cache.draft_swa_window = params.draft_swa_window
    return original(cache, replace(params, sliding_window_size=params.swa_retain_window))


def around_swa_init(original, component, cache, params):
    if not is_hcu():
        return original(component, cache, params)
    result = original(component, cache, replace(params, sliding_window_size=params.swa_retain_window))
    component.gates_prefix_match = params.swa_gates_prefix_match
    return result


def around_match_validator(original, component, *args, **kwargs):
    if not is_hcu():
        return original(component, *args, **kwargs)
    return _SOURCE_MATCH_VALIDATOR[0](component, *args, **kwargs)


_SOURCE_MATCH_VALIDATOR = []


def register_hooks():
    if not is_hcu():
        return
    from hcu_sglang.mem_cache import prefix_protocol
    prefix_protocol.register_hooks()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.unified_cache.components import swa
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    import types

    BasePrefixCache.draft_swa_window = 0
    BasePrefixCache.swa_retain_window = property(swa_retain_window)
    if not hasattr(BaseSpecWorker, 'draft_swa_window'):
        BaseSpecWorker.draft_swa_window = 0
    _SOURCE_MATCH_VALIDATOR[:] = [types.FunctionType(
        source_create_match_validator.__code__, vars(swa),
        'create_match_validator', source_create_match_validator.__defaults__)]
    HookRegistry.register('sglang.srt.mem_cache.cache_init_params.CacheInitParams', CacheInitParams, HookType.REPLACE)
    for target, wrapper in (
        ('sglang.srt.managers.scheduler.Scheduler.__init__', around_scheduler_init),
        ('sglang.srt.mem_cache.kv_cache_builder.build_kv_cache', around_build_kv_cache),
        ('sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.__init__', around_unified_init),
        ('sglang.srt.mem_cache.unified_cache.components.swa.SWAComponent.__init__', around_swa_init),
        ('sglang.srt.mem_cache.unified_cache.components.swa.SWAComponent.create_match_validator', around_match_validator),
    ):
        HookRegistry.register(target, wrapper, HookType.AROUND)


def source_create_match_validator(
    self, match_device_only: bool = False
) -> Callable[[UnifiedTreeNode], bool]:
    sliding_window_size = self.sliding_window_size
    ct = self.component_type
    state = {"len": float("inf")}
    gates_prefix_match = self.gates_prefix_match

    def validator(node: UnifiedTreeNode) -> bool:
        cd = node.component_data[ct]
        # HiCache: a host-only tombstone is a valid match boundary too
        # — load_back will restore SWA from host before use.
        if cd.value is None and (match_device_only or cd.host_value is None):
            state["len"] = 0
            if not gates_prefix_match and (node.backuped or not node.evicted):
                return True
            return False
        state["len"] += len(node.key)
        return state["len"] >= sliding_window_size

    return validator
