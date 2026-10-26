#!/bin/sh
set -eu
backup=/home/USER/service-backups/20261026-gpu-unlock-persistent
candidate=/data/benchmarks/grub.gpu-unlock-persistent
targets='gpu_unlock.targets=0000:01:00.0,0000:22:00.0,0000:41:00.0,0000:61:00.0'
cmp -s /etc/default/grub "$backup/grub.before" || {
  echo '/etc/default/grub changed since backup; refusing overwrite' >&2
  exit 1
}
grep -Fq "$targets" "$candidate"
install -m 0644 -o root -g root "$candidate" /etc/default/grub
update-grub
grub-script-check /boot/grub/grub.cfg
grep -Fq "$targets" /boot/grub/grub.cfg
inspect_dir=$(mktemp -d /tmp/gpu-unlock-initrd.XXXXXX)
trap 'rm -rf "$inspect_dir"' EXIT
unmkinitramfs /boot/initrd.img-7.0.0-38-generic "$inspect_dir"
find "$inspect_dir" -name 'gpu_unlock.ko.zst' -type f | grep -q .
find "$inspect_dir" -name 'v67-ad102-610.43.03-four-plm.bin' -type f | grep -q .
asset_count=$(find "$inspect_dir" -path '*/usr/lib/firmware/gpu-unlock/610.43.03/*' -type f | wc -l)
[ "$asset_count" -eq 9 ]
grep -qx 'gpu_unlock ' "$inspect_dir/main/conf/modules"
rm -rf "$inspect_dir"
trap - EXIT
cp -a /etc/default/grub "$backup/grub.after"
cp -a /boot/grub/grub.cfg "$backup/grub.cfg.after"
sha256sum /boot/initrd.img-* > "$backup/initrd-sha256.after"
cp -a /data/benchmarks/gpu-unlock-evaluation/latest-debbuild/deb/gpu-unlock-dkms_1.1.1_all.deb "$backup/"
cp -a /data/benchmarks/gpu-unlock-evaluation/latest-debbuild/deb/gpu-unlock-firmware_1.1.1_all.deb "$backup/"
sha256sum "$backup"/*.deb > "$backup/deb-SHA256SUMS"
dpkg-query -W gpu-unlock-dkms gpu-unlock-firmware dkms > "$backup/installed-packages.txt"
chown -R ise:ise "$backup"
echo 'Persistent four-GPU GRUB target installed and validated.'
