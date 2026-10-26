#!/bin/sh
set -eu
backup=/home/USER/service-backups/20261026-gpu-unlock-persistent
install -d -m 0700 -o ise -g ise "$backup"
cp -a /etc/default/grub "$backup/grub.before"
cp -a /boot/grub/grub.cfg "$backup/grub.cfg.before"
cp -a /boot/grub/grubenv "$backup/grubenv.before"
sha256sum /boot/initrd.img-* > "$backup/initrd-sha256.before"
dpkg-query -W > "$backup/dpkg-query.before.txt"
cp -a /data/benchmarks/gpu-unlock-evaluation/test-grub-4gpu-entry.txt "$backup/"
chown -R ise:ise "$backup"
apt-get install -y dkms dh-sequence-dkms
