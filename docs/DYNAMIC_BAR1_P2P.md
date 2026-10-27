# Dynamic BAR1 P2P on a 48GB framebuffer / 32GB BAR1 card

## Why static BAR1 is insufficient

Static BAR1 P2P assumes the peer framebuffer can be represented by one global
BAR1 mapping. A modified RTX 4090 in this evaluation exposes 48GB framebuffer
memory but only 32GB BAR1, so an allocation may live above the static aperture.

The evaluated Method3 path maps each peer allocation into a bounded BAR1 window
on demand. The peer GPU's GMMU maps that BAR1 virtual range to the allocation's
actual framebuffer pages. Modern CUDA VMM / cuMem and NCCL paths can then share
communication buffers without identity-mapping all 48GB.

## Evaluated source

- Base: NVIDIA open GPU kernel modules 595.99.02.
- Patch: [`patches/p2p48g-595.99.02-port.patch`](../patches/p2p48g-595.99.02-port.patch).
- Primary community design lineage:
  [`mochgolf/open-gpu-kernel-modules`](https://github.com/mochgolf/open-gpu-kernel-modules)
  and the 48GB dynamic-BAR1 work published by
  [`xiaoyanzi191/open-gpu-kernel-modules-p2p`](https://github.com/xiaoyanzi191/open-gpu-kernel-modules-p2p).

The port compiled cleanly at the module/link stage. Objtool emitted the same
large family of `naked return` warnings seen during the audited direct build;
absence of a compiler error was not treated as runtime qualification.

## Required runtime checks

1. Match kernel module and userspace driver versions.
2. Confirm the running module `srcversion`, not only `modinfo` on a disk file.
3. Confirm all intended cards unlocked before NVIDIA bound them.
4. Validate owner-side data and guards on every ordered pair.
5. Test an allocation beyond BAR1 capacity.
6. Verify NCCL's selected transport in logs.
7. Compare kernel logs before and after the workload.

## Directional limitation observed

Most 256MiB copy directions reached 26.1–26.3GB/s. Directions targeting one GPU
were about 10.7–11.0GB/s in the matrix. SM-driven peer loads/stores reproduced
the asymmetry, which rules out a `cudaMemcpyPeer` copy-engine-only explanation.

The affected GPU passed local VRAM, Gen4x16, AER, power, and thermal checks, and
its reverse direction was full speed. The remaining suspect is the target-side
BAR1 / AMD host-bridge path. This limits four-rank ring bandwidth and is kept as
an unresolved platform finding.

## Non-goals

- No support claim for mixed 4090/A30 peer mappings.
- No GPUDirect RDMA claim for the RTX 4090.
- No claim that legacy whole-device peer APIs expose all 48GB.
- No claim that a patch from one NVIDIA release applies to another release.
