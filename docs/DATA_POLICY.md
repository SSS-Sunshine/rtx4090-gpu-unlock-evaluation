# Published and excluded data

Published:

- Timing, throughput, power, temperature, clock, utilization, and bandwidth.
- Numerical mismatch counts and generated-test pass/fail results.
- Sanitized benchmark scripts and service configuration examples.
- Source and binary commit/hash provenance.

Excluded:

- API keys, passwords, `.env` files, and bearer tokens.
- Public and private server addresses and host names.
- Root filesystem UUIDs and full GPU UUIDs.
- Full local-model chain-of-thought and private project source prompts.
- Model weights, NVIDIA firmware archives, generated firmware binaries, and
  compiled benchmark executables.

The upstream project documents the NVIDIA firmware provenance and licensing.
This repository contains hashes and measurements, not redistributable firmware.
