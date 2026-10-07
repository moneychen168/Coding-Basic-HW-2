#!/usr/bin/env bash
# scripts/build.sh -- compile libggemm + test/bench binaries (owner: agent-B).
# Compile-only: safe on the login node. Never runs any binary.
#
#   scripts/build.sh                 # sm_90 + sm_120, Release (-O3), output in build/
#   ARCHS="120" scripts/build.sh     # nano4 RTX 5090 only
#   BUILD_DIR=build_dbg OPT="-O0 -g -G" scripts/build.sh
#
# Library = every src/*.cu and src/*.cpp (agent-N/K/P files), built with nvcc directly so
# this script does not depend on the state of CMakeLists.txt. Set USE_CMAKE=1 to build the
# library through CMake instead (expects a target producing libggemm.a in $BUILD_DIR).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
ARCHS="${ARCHS:-90 120}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OPT="${OPT:--O3}"
JOBS="${JOBS:-8}"

GENCODE=""
for a in ${ARCHS//;/ }; do GENCODE+=" -gencode arch=compute_${a},code=sm_${a}"; done
NVFLAGS="-std=c++17 $OPT -lineinfo $GENCODE -I$ROOT/include -I$ROOT/src -Xcompiler -Wall"

mkdir -p "$BUILD_DIR/obj"
echo "[build] nvcc=$NVCC archs=($ARCHS) out=$BUILD_DIR"
"$NVCC" --version | tail -1

if [[ "${USE_CMAKE:-0}" == 1 && -f "$ROOT/CMakeLists.txt" ]]; then
  cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${ARCHS// /;}"
  cmake --build "$BUILD_DIR" --target ggemm -j "$JOBS"
  LIB="$(find "$BUILD_DIR" -name 'libggemm.a' | head -1)"
else
  shopt -s nullglob
  SRCS=("$ROOT"/src/*.cu "$ROOT"/src/*.cpp)
  if (( ${#SRCS[@]} == 0 )); then echo "[build] ERROR: no library sources in src/" >&2; exit 1; fi
  OBJS=()
  pids=()
  for s in "${SRCS[@]}"; do
    o="$BUILD_DIR/obj/lib_$(basename "$s").o"; OBJS+=("$o")
    "$NVCC" $NVFLAGS -x cu -c "$s" -o "$o" & pids+=($!)
    if (( ${#pids[@]} >= JOBS )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
  LIB="$BUILD_DIR/libggemm.a"
  rm -f "$LIB"; ar rcs "$LIB" "${OBJS[@]}"
fi
echo "[build] library: $LIB"

# Same convention as CMakeLists.txt: tests/test_*.cu and bench/bench_*.cu each hold a main()
# -> one executable each; every other .cu/.cpp directly under tests/ and bench/ is support code.
# TARGETS="test_grouped bench_grouped" limits which executables are built (default: all).
shopt -s nullglob
SUP_OBJS=()
for s in "$ROOT"/tests/*.cu "$ROOT"/tests/*.cpp "$ROOT"/bench/*.cu "$ROOT"/bench/*.cpp; do
  b="$(basename "$s")"; [[ "$b" == test_* || "$b" == bench_* ]] && continue
  o="$BUILD_DIR/obj/sup_${b}.o"
  "$NVCC" $NVFLAGS -I"$ROOT/tests" -I"$ROOT/bench" -x cu -c "$s" -o "$o"; SUP_OBJS+=("$o")
done
fail=()
for s in "$ROOT"/tests/test_*.cu "$ROOT"/bench/bench_*.cu; do
  name="$(basename "$s" .cu)"
  if [[ -n "${TARGETS:-}" && " $TARGETS " != *" $name "* ]]; then continue; fi
  if "$NVCC" $NVFLAGS -I"$ROOT/tests" -I"$ROOT/bench" "$s" "${SUP_OBJS[@]}" "$LIB" -lcublas -lcublasLt \
       -o "$BUILD_DIR/$name"; then
    echo "[build] OK: $BUILD_DIR/$name"
  else
    fail+=("$name")
  fi
done
if (( ${#fail[@]} )); then echo "[build] FAILED: ${fail[*]}" >&2; exit 1; fi
