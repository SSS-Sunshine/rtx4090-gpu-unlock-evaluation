# Reproducing the measurements

The scripts are sanitized snapshots of the local harness. Before running them,
replace the following placeholders and paths for your machine:

- `GPU-REDACTED-0` through `GPU-REDACTED-5`
- `/home/USER`
- `/data/USER`
- `/data/benchmarks/gpu-unlock-evaluation`
- the example PCI addresses in boot scripts

Never copy the example GRUB or initramfs commands blindly. Use `lspci -Dnn` to
identify exact supported GPU functions and keep a known-good stock boot entry.

Recommended order:

1. Record a stock boot: kernel, command line, driver, UUID, clocks, power limit,
   and binary hashes.
2. Build the benchmark binaries once and preserve their SHA256 hashes.
3. Run numerical checks, issue-rate probes, GEMMs, application workloads, and
   telemetry in the stock state.
4. Follow the upstream project's current quick start using an isolated boot
   entry and one exact card for the first attempt.
5. Confirm the exact card's `gpu_unlock: unlocked` line and healthy NVIDIA
   driver before running workloads.
6. Repeat the same binaries and commands without changing clocks, power limits,
   software, prompts, or workload settings.
7. Check the kernel journal for Xid and PCIe errors after the suite.

For dynamic BAR1 qualification, additionally:

8. Confirm the running module `srcversion`; checking only a disk module is insufficient.
9. Validate every ordered pair with owner-side checks and guards.
10. Keep more than 32GiB resident and verify a later CUDA VMM peer allocation.
11. Set `NCCL_P2P_LEVEL=SYS` only for the controlled P2P run and confirm NCCL's
    logged channel transport instead of inferring it from the topology matrix.
12. Preserve a stock initramfs with matching NVIDIA modules embedded inside it.

Kernel updates may use DKMS, but a new NVIDIA userspace release requires a
matching P2P source port and package. Do not disable version-mismatch checks.

The included systemd drop-ins reference local virtual environments and model
paths. They are documentation examples, not portable service files.
