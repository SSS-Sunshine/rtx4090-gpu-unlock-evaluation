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

The included systemd drop-ins reference local virtual environments and model
paths. They are documentation examples, not portable service files.
