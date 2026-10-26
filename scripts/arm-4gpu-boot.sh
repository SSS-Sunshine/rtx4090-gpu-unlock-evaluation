#!/bin/sh
set -eu
entry='ada-4gpu-test-7.0.0-38'
targets='gpu_unlock.targets=0000:01:00.0,0000:22:00.0,0000:41:00.0,0000:61:00.0'
grep -n 'Four RTX4090 Ada unlock TEST' /boot/grub/grub.cfg
grep -n "$targets" /boot/grub/grub.cfg
grep -n 'initrd.img-7.0.0-38-generic-ada-4gpu-test' /boot/grub/grub.cfg
grub-script-check /boot/grub/grub.cfg
grub-reboot "$entry"
grub-editenv /boot/grub/grubenv list
