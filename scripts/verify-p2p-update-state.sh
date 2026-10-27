#!/usr/bin/env bash
set -euo pipefail

expected_driver="${EXPECTED_DRIVER:-595.99.02}"
expected_srcversion="${EXPECTED_SRCVERSION:-ED321805D08C46B02C20421}"
kernel="${KERNEL_VERSION:-$(uname -r)}"

echo "kernel=$kernel"
dkms status -k "$kernel"

resolved="$(modinfo -k "$kernel" -n nvidia)"
version="$(modinfo -F version "$resolved")"
srcversion="$(modinfo -F srcversion "$resolved")"
printf 'resolved_nvidia=%s\nversion=%s\nsrcversion=%s\n' "$resolved" "$version" "$srcversion"

case "$resolved" in
  /lib/modules/"$kernel"/updates/dkms/nvidia.ko*) ;;
  *) echo 'nvidia does not resolve to the P2P DKMS module' >&2; exit 1;;
esac
test "$version" = "$expected_driver"
test "$srcversion" = "$expected_srcversion"

if [ "$kernel" = "$(uname -r)" ]; then
  test "$(cat /sys/module/nvidia/version)" = "$expected_driver"
  test "$(cat /sys/module/nvidia/srcversion)" = "$expected_srcversion"
fi

image="/boot/initrd.img-$kernel"
test -r "$image"
initramfs_contents="$(lsinitramfs "$image")"
grep -q 'gpu_unlock\.ko' <<<"$initramfs_contents"
grep -q 'scripts/init-top/00-gpu-unlock-p2p' <<<"$initramfs_contents"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if journalctl -k -b --no-pager | grep -Eiq \
  'NVRM: Xid|METHOD3.*(error|fail)|IOMMU.*(fault|error)|invalid PTE|NVRM.*(assert|error)'; then
  echo 'kernel error marker found' >&2
  exit 1
fi

echo 'P2P update-state verification passed.'
