# GDN HIP recurrent kernel

`gdn_recurrent.hip` is the hand-written HIP recurrent kernel for GDN. It
exports the per-layer entries (`gdn_recurrent_update`, `gdn_recurrent_update_bplus`,
`gdn_recurrent_update_bplus_state`, `gdn_recurrent_update_nofma`) and the
one-launch entry `gdn_recurrent_update_bplus_state_multi`. It is bit-exact with
the aiter/Triton reference (`maxULP=0` over the recorded sweep).

Build for gfx938 with the DTK toolchain:

```bash
bash csrc/gdn/build.sh                   # -> csrc/gdn/build/gdn_recurrent.so
# or point the runtime at a prebuilt object:
export GDN_HIP_SO=/path/to/gdn_recurrent.so
```

`gdn_hip_bplus.py` loads `GDN_HIP_SO` first, then `csrc/gdn/build/gdn_recurrent.so`.
