# v2.0.0 release notes

This release expands the original Ada Tensor Unlock evaluation into a unified
compute, dynamic-BAR1 P2P, NCCL, real-model, and update-lifecycle study.

## Added

- NVIDIA 595.99.02 tracked-source dynamic BAR1 P2P port.
- 180-case, four-GPU ordered-pair correctness matrix.
- Bidirectional 34GiB-resident / 64MiB-window CUDA VMM boundary test.
- Strict NCCL SHM/P2P A/B from 1MiB through 256MiB.
- Two-rank 4KiB–1MiB small-collective latency sweep.
- Strict Qwen Flash TP4 and TEP4 transport A/B at concurrency 8.
- Qwen27 BF16 TP2 + MTP1 negative P2P result.
- Directional SM-driven peer read/write probe.
- DKMS kernel-update design, exact userspace pairing, early unlock ordering,
  persistent validated boot, and embedded-stock rescue image.
- Sanitized raw evidence, third-party test-source provenance, and generated
  machine-readable v2 summary.

## Corrected

- Replaced the obsolete statement that RTX 4090 peer access was unsupported.
- Separated pure P2P gains from scheduler, TEP4, graph-capture, and MTP gains.
- Retained historical MiniMax-H3 and Qwen27 claims as comparable rather than
  silently promoting them to strict current-software A/B results.
- Documented the target-side GPU direction bottleneck and the small-message
  crossover where SHM can beat P2P.

## Principal numbers

- FP16/BF16 issue rate: about 2x, retained with P2P active.
- NCCL 256MiB AllReduce: 2.70x.
- Qwen Flash pure P2P gain: +7.36% TP4; +8.19% TEP4.
- Qwen Flash final stack vs old serialized config: 4.93x aggregate throughput.
- Qwen27 TP2/MTP1 P2P change: −1.99%.

No compiled NVIDIA modules, DKMS packages, model weights, firmware binaries,
credentials, host addresses, or private prompts are included.
