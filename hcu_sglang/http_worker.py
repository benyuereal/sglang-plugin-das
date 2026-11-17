# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

def around_uvicorn_run(original, app, *args, **kwargs):
    if app == "sglang.srt.entrypoints.http_server:app" and (kwargs.get("workers") or 1) > 1:
        app = "hcu_sglang.http_worker:app"
    return original(app, *args, **kwargs)


def __getattr__(name):
    if name != "app":
        raise AttributeError(name)

    from hcu_sglang.compat.aiter import bootstrap_aiter_compat

    bootstrap_aiter_compat()

    from sglang.srt.plugins import load_plugins

    load_plugins()

    from sglang.srt.entrypoints.http_server import app

    return app


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register("uvicorn.run", around_uvicorn_run, HookType.AROUND)
