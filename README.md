# RTX 4090 48GB Ada Tensor Unlock Evaluation

Reproducible measurements from a four-GPU RTX 4090 48GB server using
[`JustAResearcher/gpu-unlock-ada`](https://github.com/JustAResearcher/gpu-unlock-ada).
The repository covers instruction issue rate, numerical correctness, cuBLAS,
PyTorch workloads, VRAM and PCIe bandwidth, NCCL collectives, ComfyUI
MiniMax-H3 generation, Qwen3.8-27B BF16, Qwen3.8-Flash-Next FP8, MTP, and
runtime bottleneck telemetry.

The published files are sanitized copies. API credentials, passwords, public
addresses, host names, root filesystem UUIDs, and full GPU UUIDs are excluded.

## Main results

| Workload | Stock | Unlocked | Change |
|---|---:|---:|---:|
| RTX 4090 FP16 input, FP32 accumulate issue rate | ~165 TFLOPS | 331–334 TFLOPS | ~2.0x |
| RTX 4090 BF16 input, FP32 accumulate issue rate | ~165 TFLOPS | 331–334 TFLOPS | ~2.0x |
| Unlocked FP8 E4M3 issue rate | — | 663.1 TFLOPS | reaches nominal full rate |
| ComfyUI MiniMax-H3 complete 5.17 s video | 92.24 s | 73.43 s | 25.6% faster |
| MiniMax-H3 denoising sampler | 77.75 s | 60.07 s | 29.4% faster |
| Qwen3.8-27B BF16 TP2, no MTP | 32.64 tok/s | 49.78 tok/s | 52.5% faster* |
| Qwen3.8-27B BF16 TP2, unlocked + MTP | — | 74.81 tok/s | 50.3% over unlocked/no-MTP |
| Qwen3.8-Flash-Next FP8 TP4 decode | ~98 tok/s | ~98 tok/s | no measurable gain |

\* The archived stock and unlocked Qwen27 requests have matching prompt and
completion token counts, but different output hashes/reasoning partition. A
current-software stock reboot is required for a bitwise strict A/B.

## GPU3 health and bandwidth

| Metric | Result |
|---|---:|
| VRAM tested | 45 GiB, three patterns |
| VRAM errors | 0 |
| Sequential VRAM write | 850–854 GB/s |
| Sequential VRAM verify/read | 956–957 GB/s |
| D2D payload bandwidth | 459.6 GB/s |
| D2D read + write traffic | 919.2 GB/s |
| PCIe H2D | 23.38 GB/s |
| PCIe D2H | 21.97 GB/s |
| 66 s BF16/FP32 GEMM average | 188.2 TFLOPS |
| Stress peak power / temperature | 395.9 W / 56 C |
| Xid, PCIe replay, numerical mismatches | 0 |

## Qwen FP8 bottleneck finding

The FP8 hardware path is unlocked, but single-stream Qwen decode does not
consume the additional tensor throughput:

- 48 layers, 512 routed experts, top-10 experts per token, intermediate size 640.
- The estimated active path is only about 32% FP8 by weight elements; most
  dense hot-path modules remain BF16.
- A forced 1,024-token decode sustained ~99.6 tokens/s while memory-controller
  utilization averaged only ~48% and power averaged 178–196 W per GPU.
- RTX 4090 peer read/write is unsupported in this topology. TP4 uses NCCL over
  host PCIe with P2P and custom all-reduce disabled.
- The service is configured for `max-num-seqs=1`, so batch-1 expert GEMMs are
  too small to approach the 663 TFLOPS FP8 issue-rate ceiling.

The limiting factors are small-matrix/kernel latency and TP4 collective
latency, followed by mixed BF16/non-GEMM work and memory access. Power and FP8
issue rate are not the limiting resources.

## Repository layout

- `results/` — sanitized JSON, CSV, and text outputs.
- `scripts/` — CUDA, Python, shell, and systemd test helpers.
- `docs/METHODOLOGY.md` — environment, matching rules, caveats, and reproduction notes.
- `manifests/SHA256SUMS` — hashes of every published result and script.

## Safety and scope

The unlock executes firmware before the NVIDIA driver binds the GPU and
performs a Function Level Reset. Keep a stock boot entry available. A successful
module load is not proof of success; verify the exact card's
`gpu_unlock: unlocked` log line, healthy `nvidia-smi`, numerical checks, and the
absence of Xid errors.

These measurements cover one server, modified 48GB RTX 4090 boards, the listed
driver/kernel combinations, and bounded test durations. They do not establish
lifetime reliability or predict every model and kernel.
