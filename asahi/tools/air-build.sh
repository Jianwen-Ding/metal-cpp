#!/bin/sh
# Builds a .air file (Apple's AIR bitcode, Metal's kernel ABI) from OpenCL C
# or LLVM IR with the in-tree clang and the LLVM AIR target. No Metal
# Shading Language and no Apple tool is involved.
#   air-build.sh kernel.cl|kernel.ll out.air [extra clang flags]
set -e
LLVM_BUILD=${LLVM_BUILD:-/scratch/llvm-build}
B=$LLVM_BUILD/bin
src=$1
out=$2
shift 2
tmp=$(mktemp -d)
cleanup() { rm -f "$tmp/k.ll" "$tmp/air.ll"; rmdir "$tmp"; }
trap cleanup EXIT
case "$src" in
*.cl)
  # kernels/air_kernel.h stands in for OpenCL's library header.
  "$B/clang" -x cl -cl-std=CL2.0 -target spir64 -O2 -emit-llvm -S -cl-fast-relaxed-math \
      -I"$(dirname "$0")/../kernels" -include air_kernel.h "$@" "$src" -o "$tmp/k.ll"
  # AIR's address spaces are SPIR's (1 device, 2 constant, 3 threadgroup);
  # drop SPIR's layout, and keep only the kernels as entry points.
  grep -v -E '^target (datalayout|triple)' "$tmp/k.ll" |
      sed -E 's/^define (dso_local )?spir_func/define internal spir_func/' > "$tmp/air.ll"
  ;;
*.ll) cp "$src" "$tmp/air.ll" ;;
*) echo "air-build.sh: $src: expected .cl or .ll" >&2; exit 1 ;;
esac
# The OpenCL front end's inner copies of the kernels are dead once inlined.
"$B/opt" -passes=globaldce -S "$tmp/air.ll" -o "$tmp/k.ll"
# out.air: one AIR object; out.metallib: a .metallib container (one module per kernel).
kind=air
case "$out" in *.metallib) kind=metallib ;; esac
"$B/llc" -mtriple=air64-apple-macosx26.0.0 -O2 -filetype=obj -air-output=$kind -air-buffer-per-argument "$tmp/k.ll" -o "$out"
