#!/bin/bash
# Build the bit-exact GDN recurrent kernel used by the kernel-only verify path.
# Requires the DTK toolchain (aicc) and a gfx938 target.
# usage: build_gdn.sh [out.so]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/gdn_bw_ship.so}"

/opt/dtk/bin/aicc -O3 --offload-arch=gfx938 -fPIC --shared \
 -mllvm=-enable-latency-hack=true -mllvm=-mmac-latency=5 -mllvm=-ds-load-store-latency=6 \
 -mllvm=-disable-machine-sink=True -mllvm=-check-valu-data-forward-hazards=0 \
 -mllvm=-disable-cluster-lds-memops=true -mllvm=-amdgpu-disable-backoff-barrier=false \
 -mllvm=-support-768-vgprs=true -mllvm=-hcu-update-wait-by-reverse-search=true \
 -mllvm=-hcu-pre-emit-load-store-opt=false \
 "$HERE/gdn_recurrent_bitwise.hip" -o "$OUT"

echo "built $OUT"
