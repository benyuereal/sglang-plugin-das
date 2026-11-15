#!/bin/bash
# Build the bit-exact GDN recurrent kernel used by the kernel-only verify path.
# usage: build_gdn.sh [out.so]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/gdn_bw_ship.so}"
. "$HERE/build_flags.sh"

"$AICC" "${GDN_BUILD_FLAGS[@]}" "$HERE/gdn_recurrent_bitwise.hip" -o "$OUT"
echo "built $OUT"
