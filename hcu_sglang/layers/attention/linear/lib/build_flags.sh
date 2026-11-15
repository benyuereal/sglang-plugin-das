#!/bin/bash
# aicc flags shared by every kernel in this directory (single source of truth).
# Source it: . "$HERE/build_flags.sh"
AICC="${AICC:-/opt/dtk/bin/aicc}"
GDN_BUILD_FLAGS=(
  -O3 --offload-arch=gfx938 -fPIC --shared
  -mllvm=-enable-latency-hack=true
  -mllvm=-mmac-latency=5
  -mllvm=-ds-load-store-latency=6
  -mllvm=-disable-machine-sink=True
  -mllvm=-check-valu-data-forward-hazards=0
  -mllvm=-disable-cluster-lds-memops=true
  -mllvm=-amdgpu-disable-backoff-barrier=false
  -mllvm=-support-768-vgprs=true
  -mllvm=-hcu-update-wait-by-reverse-search=true
  -mllvm=-hcu-pre-emit-load-store-opt=false
)
