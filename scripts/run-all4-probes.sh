#!/usr/bin/env bash
set -euo pipefail
root=/data/benchmarks/gpu-unlock-evaluation/all4-unlocked
bin=/data/benchmarks/gpu-unlock-evaluation/bin
mkdir -p "$root"
sha256sum -c /data/benchmarks/gpu-unlock-evaluation/bin/SHA256SUMS
uuids=(
  GPU-REDACTED-0
  GPU-REDACTED-1
  GPU-REDACTED-2
  GPU-REDACTED-3
)
for index in 0 1 2 3; do
  out="$root/gpu$index"
  mkdir -p "$out"
  export CUDA_VISIBLE_DEVICES="${uuids[$index]}"
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit --format=csv > "$out/gpu.csv"
  "$bin/validate_gemm" | tee "$out/numerical.txt"
  "$bin/mma_bench" 20000 5 | tee "$out/mma.txt"
  "$bin/gemm_bench" | tee "$out/gemm.jsonl"
done
journalctl -k -b --no-pager > "$root/kernel-journal.txt"
