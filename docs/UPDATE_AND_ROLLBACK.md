# Kernel updates, driver pairing, and rollback

## Problem

`gpu_unlock.ko` is an independent DKMS module, but dynamic BAR1 P2P changes the
NVIDIA kernel module source itself. Kernel updates and NVIDIA driver updates
therefore have different compatibility boundaries.

## Adopted policy

### Kernel updates

Both module families use DKMS with `AUTOINSTALL=yes`:

- `gpu-unlock/1.1.1`
- `nvidia-open-p2p/595.99.02-p2p3`

After the ordinary header DKMS hook builds and installs modules for a new
kernel, a later header-postinst hook refreshes that kernel's initramfs. The
initramfs includes an init-top script that loads `gpu_unlock` before udev can
bind the NVIDIA driver.

### NVIDIA driver updates

The P2P package depends on the exact userspace driver components it was built
against. This replaces a collection of manual `apt-mark hold` entries with an
explicit package relationship:

```text
matching userspace + matching P2P DKMS source = installable pair
```

An arbitrary new NVIDIA userspace version is not mixed with the older kernel
module. A new driver becomes eligible only when a matching source port and
replacement P2P DKMS package exist. This is controlled upgrading, not universal
cross-version compatibility.

## Boot entries

- Validated default: version-pinned P2P initramfs and exact unlock targets.
- Standard generated image: early Tensor Unlock, then P2P DKMS modules.
- Stock rescue: a separate initramfs with matching Ubuntu stock NVIDIA modules
  embedded inside it, so rootfs DKMS precedence cannot replace them.

The GRUB menu remains visible. Do not delete the last known-good kernel merely
because a newer kernel was installed successfully.

## Lightweight acceptance for a new kernel

1. Confirm both DKMS modules report `installed` for the new kernel.
2. Inspect the generated initramfs for the early unlock script and module.
3. Boot the new kernel once.
4. Confirm exact unlock logs, running P2P `srcversion`, healthy inventory, and no Xid.
5. Run one owner-verified ordered-pair sweep, one >32GiB resident test, one
   256MiB NCCL correctness test, and one fixed Qwen request.

Run the full matrix again after a driver-source port, BIOS/topology change,
hardware move, or any lightweight-test failure.

## Rollback

Choose the stock rescue entry in the GRUB menu. From the rescue environment,
remove or repair the P2P DKMS package, rebuild the affected initramfs, and retain
the verified rescue image until the repaired candidate passes acceptance.

Never disable the NVIDIA kernel/userspace version check to hide a mismatch.
