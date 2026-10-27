# Unified results

## Evaluated stack

- Four modified RTX 4090 48GB cards, 450W maximum each.
- Two A30 cards present but excluded from every 4090 P2P test.
- Linux `7.0.0-38-generic`.
- NVIDIA userspace and open kernel module `595.99.02`.
- `gpu-unlock-ada` 1.1.1 for Ada Tensor issue-rate unlock.
- Dynamic, per-allocation BAR1 P2P ported to the matching 595.99.02 source.
- Physical BAR1 32GB; framebuffer 48GB.

## Compute and communication are independent gains

Re-running the latest issue-rate binary with the P2P driver loaded showed that
the two mechanisms coexist:

| GPU | FP16/FP32 | BF16/FP32 | FP8 E4M3 |
|---:|---:|---:|---:|
| 0 | 331.5 | 331.3 | 660.4 |
| 1 | 334.4 | 334.2 | 666.1 |
| 2 | 333.1 | 332.5 | 660.1 |
| 3 | 328.0 | 331.0 | 663.6 |

Units are TFLOPS-like instruction issue rate from the benchmark. Stock
FP16/FP32 and BF16/FP32 were about 165, so the roughly 2x Tensor Unlock remains.
FP8 does not receive another 2x multiplier.

## Dynamic BAR1 evidence chain

1. All 12 ordered pairs advertise peer access.
2. 180 matrix cases checked writes, reads, owner readback, and guards.
3. A 34GiB resident allocation was kept intact while a 64MiB VMM allocation
   was mapped into both GPUs; both directions returned zero mismatches.
4. Allocations returned to baseline and the filtered kernel-error file stayed empty.

This demonstrates access to allocations placed above the 32GB BAR1 boundary.
It does not establish support for every legacy whole-device peer API, MIG,
vGPU, mixed static/dynamic GPU pairs, or allocations that collectively exceed
the available BAR1-window budget.

## NCCL transport A/B

The enabled run logged 22 P2P channels and zero SHM channels. The disabled run
logged 22 SHM channels and zero P2P channels. Every rank verified the reduction.

Large-message performance scaled with payload, reaching 10.64GB/s bus bandwidth
and 2.70x speedup at 256MiB. The result is lower than reports from homogeneous
24GB cards with full-framebuffer static BAR1 because this host has a slow
target-side direction and uses dynamic BAR1 windows.

## Qwen interpretation

The Qwen Flash 4.93x total improvement decomposes into:

- removing serialized scheduling (`max-num-seqs=1`),
- capturing batch sizes 1/2/4/8,
- selecting expert parallelism for the sparse model,
- enabling NCCL P2P at `NCCL_P2P_LEVEL=SYS`.

Only the transport contribution should be attributed to P2P: +7.36% under TP4
and +8.19% under TEP4. The Qwen27 negative result shows why a global “P2P is
always faster” claim would be wrong.

## Comparison limitations

- Historical MiniMax-H3 and stock/unlocked Qwen27 rows are comparable evidence,
  not current-software bitwise A/B.
- TEP4 changes the parallel decomposition and did not reproduce TP4's exact
  output hash. A bounded generated regression suite passed 4/4, but that is not
  a full intelligence or quality benchmark.
- NVML utilization counters are activity percentages, not measured GB/s.
- No long-duration lifetime claim is made.
