#!/bin/bash
# Build the B+ GDN kernel with the multi-layer compensate entry, the
# prenormalized-q/k verify entry and the norm-pass kernel.
# usage: build.sh [out.so]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/build/gdn_recurrent.so}"
. "$HERE/build_flags.sh"

"$AICC" "${GDN_BUILD_FLAGS[@]}" "$HERE/gdn_recurrent.hip" -o "$OUT"
echo "built $OUT"
