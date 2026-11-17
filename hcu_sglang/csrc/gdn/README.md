# GDN HIP recurrent kernel

`gdn_recurrent.hip` is the hand-written HIP recurrent kernel for GDN. It
exports the per-layer entries (`gdn_recurrent_update`,
`gdn_recurrent_update_bplus`, `gdn_recurrent_update_bplus_state`,
`gdn_recurrent_update_nofma`), the prenorm/norm-pass entries and the one-launch
entry `gdn_recurrent_update_bplus_state_multi`. It is bit-exact with the
aiter/Triton reference (`maxULP=0` over the recorded sweep).

The Python side (`layers/attention/linear/gdn_recurrent.py`) builds the kernel
on first use with `aicc` (`hipcc` as fallback) and caches the object under
`~/.cache/sglang/gdn_recurrent/<source+flags hash>/`; the aicc flags live in
that module, next to the loader. Point `GDN_HIP_SO` at a prebuilt object to
skip the build (CI/image builds do this):

```bash
export GDN_HIP_SO=/path/to/gdn_recurrent.so
```

There is no separate build script by design; this is the same
build-on-first-use pattern as
`layers/attention/linear/kda_state_indices_hip.py`.
