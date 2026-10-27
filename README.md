# RTX 4090 48GB Unified Tensor Unlock + Dynamic BAR1 P2P Evaluation

Reproducible measurements from one four-GPU modified RTX 4090 48GB server.
This repository now evaluates the complete stack:

1. Ada Tensor issue-rate unlock with
   [`gpu-unlock-ada`](https://github.com/JustAResearcher/gpu-unlock-ada).
2. PCIe peer access for 48GB framebuffer / 32GB BAR1 cards using a dynamic,
   per-allocation BAR1 driver path derived from community open-kernel-module work.
3. NCCL transport, real Qwen workloads, correctness boundaries, boot recovery,
   and a DKMS-based kernel-update lifecycle.

The published files are sanitized. API credentials, passwords, server addresses,
host names, filesystem UUIDs, full GPU UUIDs, machine serial numbers, model
weights, firmware binaries, and private prompts are excluded.

## Result at a glance

| Layer | Baseline | Optimized | Result |
|---|---:|---:|---:|
| FP16 input / FP32 accumulate issue rate | 164.9 TFLOPS | 328.0–334.4 TFLOPS | about 2.0x |
| BF16 input / FP32 accumulate issue rate | 164.8 TFLOPS | 331.0–334.2 TFLOPS | about 2.0x |
| FP8 E4M3 issue rate with both unlocks active | — | 660.1–666.1 TFLOPS | no second 2x multiplier |
| Four-rank NCCL AllReduce, 256 MiB | 102.25 ms | 37.85 ms | 2.70x |
| Qwen Flash TP4, concurrency 8 | 331.50 tok/s SHM | 355.91 tok/s P2P | +7.36% |
| Qwen Flash TEP4, concurrency 8 | 432.23 tok/s SHM | 467.61 tok/s P2P | +8.19% |
| Qwen Flash old serialized config → final | 94.81 tok/s | 467.61 tok/s | 4.93x total |
| Qwen3.8-27B BF16 TP2 + MTP1, concurrency 1 | 75.12 tok/s SHM | 73.62 tok/s P2P | −1.99% |

The 4.93x figure is a **configuration-stack** gain, not a pure P2P claim. Most
of it comes from removing `max-num-seqs=1` serialization and enabling TEP4;
strict P2P-only A/B gains are 7.36% and 8.19% on the two Qwen Flash modes.

## Dynamic BAR1 correctness

The cards expose 48GB framebuffer memory but only 32GB BAR1. The tested driver
maps individual peer allocations into bounded BAR1 windows instead of assuming
that BAR1 identity-maps the whole framebuffer.

- Four RTX 4090s: 12 ordered GPU pairs.
- 3 rounds × 5 allocation sizes per pair: **180/180 passed**.
- Every case checked peer writes, peer reads, owner-side readback, and guards.
- A destination kept 34GiB resident while a later 64MiB CUDA VMM allocation was
  mapped into both GPUs: **both directions passed with zero mismatches**.
- Test allocations were released and no Xid, invalid PTE, IOMMU fault, or
  Method3 error was added to the kernel log.

See [Dynamic BAR1 P2P](docs/DYNAMIC_BAR1_P2P.md) and the raw sanitized logs in
[`results/v2`](results/v2).

## NCCL: bandwidth helps only above the latency floor

`NCCL_P2P_LEVEL=SYS` was required because the four GPUs sit behind separate
PCIe host bridges. Without it, NCCL reported CUDA P2P capability but still
selected `SHM/direct`.

| Payload | SHM latency | P2P latency | Speedup |
|---:|---:|---:|---:|
| 1 MiB | 0.372 ms | 0.328 ms | 1.14x |
| 16 MiB | 6.440 ms | 2.834 ms | 2.27x |
| 64 MiB | 25.706 ms | 10.030 ms | 2.56x |
| 256 MiB | 102.250 ms | 37.847 ms | 2.70x |

The small-message TP2 sweep explains the Qwen27 counterexample. Around the
model's approximate 10KiB batch-1 hidden-state payload, P2P and SHM are close;
8KiB P2P was slower. At 1MiB, P2P was about 4.6x faster.

## Real-model findings

### Qwen3.8-Flash-Next FP8, TP4 / TEP4

The same eight concurrent, 1,500-input-token / 2,000-output-token real-project
requests were replayed twice per transport and parallelism mode.

| Mode | SHM runs | P2P runs | Mean change |
|---|---:|---:|---:|
| TP4 | 331.31 / 331.69 | 354.41 / 357.42 | +7.36% |
| TEP4 | 430.28 / 434.17 | 465.80 / 469.42 | +8.19% |

The original `max-num-seqs=1` configuration queued seven of eight requests and
delivered only 94.81 aggregate tok/s. TEP4 + concurrency 8 + P2P averaged
467.61 tok/s. A bounded generated-test check passed 4/4 cases. TEP4 and TP4 are
not bitwise-identical decompositions, so this is not presented as a complete
model-quality evaluation.

### Qwen3.8-27B BF16, TP2 + MTP1

P2P was 1.99% slower than SHM in a strict same-software A/B, while output hashes
matched. This is an important negative result: enabling P2P globally is not
always optimal. The retained operational policy is workload-specific:

- Qwen Flash TEP4/concurrency 8: P2P enabled.
- Qwen27 TP2/concurrency 1: P2P disabled.

## Directional topology limitation

Most 256MiB peer-copy directions reached about 26.1–26.3GB/s. Traffic targeting
one GPU was much slower (about 10.7–11.0GB/s in the matrix; repeated SM-driven
remote accesses also showed the same directionality). The reverse direction
remained full speed. Local VRAM, Gen4x16, AER, power, and thermal tests passed,
so this is reported as a target-side BAR1 / host-bridge path limitation, not a
failed GPU.

## Update and recovery model

The evaluated host uses two DKMS modules:

- `gpu-unlock/1.1.1`
- `nvidia-open-p2p/595.99.02-p2p3`

The local `nvidia-open-p2p-dkms` package vendors the audited source, sets
`AUTOINSTALL=yes`, rebuilds for new kernels, and refreshes initramfs after the
header DKMS hook. All former NVIDIA `apt-mark hold` entries were removed.

Driver updates remain **paired**, not blindly automatic: the P2P package depends
on the exact matching userspace version. A new NVIDIA release is installed only
together with a matching P2P source/package. A version-pinned validated P2P boot
entry and an initramfs-embedded stock NVIDIA rescue entry remain available.

See [Update and rollback](docs/UPDATE_AND_ROLLBACK.md).

## Earlier retained application and health results

The v1 evidence remains in `results/`. These rows are preserved with their
original comparison boundary rather than rewritten as new strict A/B results.

| Workload | Earlier baseline | Unlocked result | Classification |
|---|---:|---:|---|
| ComfyUI MiniMax-H3 complete 5.17s video | 92.24s | 73.43s | comparable; decoded video/audio hashes matched |
| MiniMax-H3 denoising sampler | 77.75s | 60.07s | comparable |
| Qwen3.8-27B BF16 TP2, no MTP | 32.64 tok/s | 49.78 tok/s | historical comparable, not bitwise A/B |
| Qwen3.8-27B unlocked, MTP1 vs no MTP | 49.78 tok/s | 74.81 tok/s | same-boot current-software A/B |

GPU3 also passed a 45GiB three-pattern VRAM test with zero errors, approximately
459.6GB/s D2D payload bandwidth, 23.38GB/s H2D, 21.97GB/s D2H, and a bounded
BF16/FP32 stress pass without Xid, PCIe replay, thermal, or numerical failures.

## Evidence classes

- **Strict A/B:** same boot/software/workload with one controlled transport or
  service-setting change. Current P2P/NCCL and Qwen transport comparisons use
  this class.
- **Comparable, not strict:** archived stock vs later unlocked measurements
  where software state or output hash differs. Historical Qwen27 and MiniMax-H3
  claims retain this qualifier.
- **Microbenchmark:** issue rate, peer bandwidth, and NCCL throughput establish
  mechanism capacity, not application quality or universal speedup.

## Repository layout

- `results/v2/` — sanitized dynamic-P2P, NCCL, Qwen, and lifecycle evidence.
- `results/v2/summary.json` — generated unified machine-readable summary.
- `patches/` — matching-version tracked-source driver patch; no binaries.
- `scripts/` — reproducible compute, communication, model, and analysis tools.
- `docs/UNIFIED_RESULTS.md` — detailed result interpretation.
- `docs/DYNAMIC_BAR1_P2P.md` — design, correctness, and limitations.
- `docs/UPDATE_AND_ROLLBACK.md` — DKMS, paired updates, boot entries, and recovery.
- `docs/METHODOLOGY.md` — test definitions and comparison boundaries.
- `manifests/SHA256SUMS` — hashes for every published result, script, patch, and document.

## Safety and scope

These are experimental driver changes on one server. A successful build,
`nvidia-smi`, or an `OK` topology matrix alone is not sufficient proof. Retain a
stock rescue path and validate owner-side data, allocation-above-BAR1 behavior,
NCCL correctness, module provenance, and kernel errors. Results do not establish
lifetime reliability or transfer automatically to another board, BIOS, kernel,
driver, or topology.
