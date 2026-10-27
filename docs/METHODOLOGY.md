# Methodology

## Hardware and software

- 4 x modified NVIDIA GeForce RTX 4090 48GB, 450 W limits.
- 2 x NVIDIA A30 present but excluded from the Ada unlock target list.
- Ubuntu 26.04, Linux 7.0.0-38-generic.
- NVIDIA driver 595.99.02 for the extended local test suite.
- Dynamic BAR1 P2P module port `srcversion=ED321805D08C46B02C20421`.
- CUDA 13 toolchains bundled with the model environments.
- `gpu-unlock-ada` v1.1.1. The initial hardware run used commit
  `4e7fefcc4923f9e083d5324dc75a7d2f906af77f`; the persistent DKMS package and
  latest FP8 benchmark were built from
  `d28c485418c3d9d034cd4bac19200bc33dd3453b`.

## Boot validation

Each selected RTX 4090 had to emit its own `gpu_unlock: unlocked` kernel line
before the NVIDIA driver initialized. The stock NVIDIA driver then had to
enumerate all GPUs without an Xid. The persistent installation used exact PCI
addresses rather than `targets=all`, DKMS, fixed-hash firmware assets, and an
init-top script that loads `gpu_unlock` before udev coldplug. A separate P2P
DKMS package provides the five matching NVIDIA open kernel modules and rebuilds
them for future kernels.

## Compute correctness

- Six seeded FP16/BF16 cuBLAS cases per GPU compared every output cell against
  CPU reference values.
- Large square cuBLAS GEMMs checked every output cell against a cheap exact
  reference.
- Tensor-mode probes checked every output cell where supported.
- The latest `mma_bench` FP8 mode is an issue-rate probe, not a complete model
  or application correctness test.

## Application A/B

MiniMax-H3 used the same ComfyUI graph, prompt, 864x480 resolution, 124 frames,
20 denoising steps, and seeds. One warmup was excluded and three measured runs
were summarized by their median. Decoded video and audio hashes matched for all
three stock/unlocked seed pairs.

Qwen benchmarks used streamed OpenAI-compatible requests and counted usage
tokens from the final stream event. Decode throughput used completion tokens
over the time from first generated token through stream completion. Thinking
tokens were included. Generated regression tests were executed in an isolated
harness against the real extracted project function.

## VRAM and PCIe

The VRAM test allocated 45 GiB and wrote/verified three deterministic 64-bit
patterns. D2D copy reports both payload bandwidth and read-plus-write traffic.
PCIe H2D/D2H used CUDA pinned host memory. The stress pass ran an 8192-cubed
BF16-input, FP32-accumulate cuBLAS GEMM for approximately 66 seconds while NVML
telemetry sampled power, clocks, temperature, utilization, and memory.

## Dynamic BAR1 P2P and NCCL

The P2P matrix covered all 12 ordered pairs, three rounds, and sizes from 4KiB
through 64MiB. Each case verified remote writes, remote reads, owner-side
readback, and guards. A CUDA VMM test separately kept 34GiB resident while
mapping a later 64MiB allocation into both peers.

Four-rank NCCL AllReduce compared `NCCL_P2P_DISABLE=1` against P2P enabled with
`NCCL_P2P_LEVEL=SYS`; all other benchmark settings were fixed. Logs were parsed
to require SHM channels in the disabled case and P2P channels in the enabled
case. Every rank verified the reduced values. `NCCL_LOCAL_REGISTER=0` and
`NCCL_GRAPH_REGISTER=0` bounded BAR1 registration during qualification.

A separate two-rank sweep measured 4KiB through 1MiB to identify the crossover
between synchronization latency and P2P bandwidth.

## Qwen transport A/B

Qwen Flash used identical eight-request, 1,500-input-token and
2,000-output-token workloads for SHM/P2P comparisons. TP4 and TEP4 were treated
as separate modes; two measured runs per transport were averaged. The original
serialized service was retained as a configuration baseline, not a P2P
baseline.

Qwen27 BF16 TP2 + MTP1 used the same prompt, 256-token output, software, and
output hash for its P2P/SHM A/B. It is reported as a negative P2P result.

## Privacy transformations

Published copies replace GPU UUIDs, local user names, server addresses, and
root UUIDs. Full model chain-of-thought records are excluded; only performance
metrics, output hashes, and test pass/fail summaries are published.

## Important comparison limits

- The Qwen27 archived stock and unlocked tests match prompt/completion token
  counts but not output hashes.
- FP8 issue-rate improvement does not imply single-stream decode improvement.
- P2P bandwidth improvement does not imply a gain for small collectives.
- The 4.93x Qwen Flash number combines scheduler, TEP4, graph-capture, and P2P changes.
- NVML memory utilization is memory-controller busy time, not a direct GB/s
  measurement.
- Short stress tests do not establish long-term board reliability.
