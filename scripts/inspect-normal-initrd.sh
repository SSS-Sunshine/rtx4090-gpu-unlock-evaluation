#!/bin/sh
set -eu
modprobe --show-depends gpu_unlock
grep -n gpu_unlock /lib/modules/7.0.0-38-generic/modules.dep /lib/modules/7.0.0-38-generic/modules.alias /lib/modules/7.0.0-38-generic/modules.softdep 2>/dev/null || true
inspect_dir=$(mktemp -d /tmp/initrd-check.XXXXXX)
trap 'rm -rf "$inspect_dir"' EXIT
unmkinitramfs /boot/initrd.img-7.0.0-38-generic "$inspect_dir"
find "$inspect_dir" \( -path '*conf/modules' -o -iname '*gpu*unlock*' -o -name 'v67-ad102*' \) -print
find "$inspect_dir" -path '*conf/modules' -type f -exec sh -c 'echo FILE:$1; cat "$1"' sh {} \;
