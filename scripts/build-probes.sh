#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/benchmarks/gpu-unlock-evaluation
SRC=/data/benchmarks/gpu-unlock-ada
CUDA=/data/venvs/minimax-h3/lib/python3.12/site-packages/nvidia/cu13
mkdir -p "$ROOT/bin"
"$CUDA/bin/nvcc" -O3 -arch=sm_89 -std=c++17 -o "$ROOT/bin/mma_bench" "$SRC/diagnostics/bench/mma_bench.cu"
"$CUDA/bin/nvcc" -O3 -arch=sm_89 -std=c++17 -L"$CUDA/lib" -l:libcublas.so.13 -Xlinker -rpath -Xlinker "$CUDA/lib" -o "$ROOT/bin/validate_gemm" "$SRC/diagnostics/bench/validate_gemm.cu"
"$CUDA/bin/nvcc" -O3 -arch=sm_89 -std=c++17 -L"$CUDA/lib" -l:libcublas.so.13 -Xlinker -rpath -Xlinker "$CUDA/lib" -o "$ROOT/bin/gemm_bench" "$SRC/docs/evidence/rtx4090-pr1-stock/gemm_bench.cu"
sha256sum "$ROOT"/bin/* > "$ROOT/bin/SHA256SUMS"
git -C "$SRC" rev-parse HEAD > "$ROOT/source-commit.txt"
