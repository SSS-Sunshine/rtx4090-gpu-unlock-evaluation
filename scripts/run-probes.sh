#!/usr/bin/env bash
set -euo pipefail
phase=${1:?stock or unlocked required}
case "$phase" in stock|unlocked|gpu1-stock) ;; *) exit 2 ;; esac
ROOT=/data/benchmarks/gpu-unlock-evaluation
export CUDA_VISIBLE_DEVICES=GPU-REDACTED-0
if [[ "$phase" == gpu1-stock ]]; then
    export CUDA_VISIBLE_DEVICES=GPU-REDACTED-1
fi
mkdir -p "$ROOT/$phase"
sha256sum -c "$ROOT/bin/SHA256SUMS"
if [[ "$phase" == unlocked ]]; then
    grep -q 'gpu_unlock.targets=0000:01:00.0' /proc/cmdline || { echo 'Not in test boot'; exit 1; }
    journalctl -k -b --no-pager > "$ROOT/$phase/kernel-journal.txt"
    grep -q 'gpu_unlock: unlocked' "$ROOT/$phase/kernel-journal.txt" || { echo 'Unlock not proven'; exit 1; }
fi
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu --format=csv -lms 1000 > "$ROOT/$phase/probe-telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
"$ROOT/bin/validate_gemm" | tee "$ROOT/$phase/numerical.txt"
"$ROOT/bin/mma_bench" 20000 5 | tee "$ROOT/$phase/mma.txt"
"$ROOT/bin/gemm_bench" | tee "$ROOT/$phase/gemm.jsonl"
