# GDN HIP recurrent kernel (kernel-only verify)

`gdn_recurrent_bitwise.hip` is the hand-written HIP recurrent kernel used by
`gdn_source.forward_extend` when `SGLANG_GDN_HIP_KERNEL=1`. It is bit-exact
with the aiter/Triton reference (`maxULP=0` over the recorded sweep).

Build for gfx938 with the DTK toolchain:

```bash
bash lib/build_gdn.sh            # -> lib/gdn_bw_ship.so
# or point the runtime at a prebuilt object:
export GDN_HIP_SO=/path/to/gdn_bw_ship.so
```

`gdn_hip_bplus.py` loads `GDN_HIP_SO` first, then `lib/gdn_bw_ship.so`.
The kernel-only path writes full per-draft snapshots; it does not use the
B+ commit/replay path.
