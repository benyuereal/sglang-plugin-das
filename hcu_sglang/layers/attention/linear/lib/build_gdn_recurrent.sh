#!/bin/bash
# Build the B+ GDN kernel with the multi-layer compensate entry, the
# prenormalized-q/k verify entry and the norm-pass kernel.
# usage: build_gdn_recurrent.sh [out.so]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/gdn_recurrent_hip.so}"
. "$HERE/build_flags.sh"

"$AICC" "${GDN_BUILD_FLAGS[@]}" "$HERE/gdn_recurrent_hip.hip" -o "$OUT"
echo "built $OUT"
