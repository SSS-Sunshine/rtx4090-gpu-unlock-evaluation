#!/usr/bin/env bash
set -euo pipefail
out=/data/benchmarks/gpu-unlock-evaluation/gpu3-full
export CUDA_VISIBLE_DEVICES=GPU-REDACTED-3
nvidia-smi -q -i 3 > "$out/nvidia-before.txt"
nvidia-smi --query-gpu=timestamp,index,pstate,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.gr,clocks.sm,clocks.mem --format=csv -i 3 -lms 500 > "$out/stress-telemetry.csv" &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT
"$out/gpu3-stress" 60 | tee "$out/stress-result.json"
cleanup
trap - EXIT
nvidia-smi -q -i 3 > "$out/nvidia-after.txt"
journalctl -k -b --no-pager > "$out/kernel-journal-after.txt"
