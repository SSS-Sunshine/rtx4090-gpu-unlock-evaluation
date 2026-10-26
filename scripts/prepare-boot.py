"""Prepare a separate initramfs/GRUB entry. No module loading, no reboot."""
import os
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
SRC = pathlib.Path('/data/benchmarks/gpu-unlock-ada')
KERNEL = '7.0.0-38-generic'
ENTRY_ID = 'ada-gpu0-test-7.0.0-38'
if os.geteuid() != 0:
    raise SystemExit('Run through sudo')
if subprocess.check_output(['uname', '-r'], text=True).strip() != KERNEL:
    raise SystemExit('Kernel changed; stop and repeat stock baseline')
backup = ROOT / 'boot-backup'
backup.mkdir(mode=0o700, exist_ok=True)
for source in ['/etc/default/grub', '/boot/grub/grub.cfg', '/boot/grub/grubenv']:
    dest = backup / pathlib.Path(source).name
    if not dest.exists():
        shutil.copy2(source, dest)
stock_initrd = pathlib.Path('/boot') / f'initrd.img-{KERNEL}'
import hashlib
initial_hash = hashlib.sha256(stock_initrd.read_bytes()).hexdigest()
(backup / 'stock-initrd.sha256').write_text(initial_hash + '\n')
module_dest = pathlib.Path('/lib/modules') / KERNEL / 'updates/ada-evaluation/gpu_unlock.ko'
module_dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(SRC / 'kernel/gpu_unlock/gpu_unlock.ko', module_dest)
firmware_dest = pathlib.Path('/usr/lib/firmware/gpu-unlock/610.43.03')
firmware_dest.mkdir(parents=True, exist_ok=True)
for p in (ROOT / 'firmware-cache/assets').iterdir():
    if p.is_file():
        shutil.copy2(p, firmware_dest / p.name)
subprocess.run(['depmod', '-a', KERNEL], check=True)
conf = ROOT / 'initramfs-config'
if conf.exists():
    raise SystemExit('Custom initramfs config exists; inspect previous attempt first')
shutil.copytree('/etc/initramfs-tools', conf)
hook = conf / 'hooks/ada-evaluation'
hook.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / 'initramfs-hook', hook)
hook.chmod(0o755)
early = conf / 'scripts/init-top/00-ada-evaluation'
early.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / 'init-top-unlock', early)
early.chmod(0o755)
custom_initrd = pathlib.Path('/boot') / f'initrd.img-{KERNEL}-ada-gpu0-test'
subprocess.run(['mkinitramfs', '-d', str(conf), '-o', str(custom_initrd), KERNEL], check=True)
listing = subprocess.check_output(['lsinitramfs', str(custom_initrd)], text=True)
for required in ['scripts/init-top/00-ada-evaluation', 'gpu_unlock.ko', 'booter_load_ad102.bin', 'v67-ad102-610.43.03-four-plm.bin']:
    if required not in listing:
        raise SystemExit(f'Missing initramfs item: {required}')
(ROOT / 'test-initramfs-files.txt').write_text(listing)
if hashlib.sha256(stock_initrd.read_bytes()).hexdigest() != initial_hash:
    raise SystemExit('Stock initrd changed unexpectedly')
grub = pathlib.Path('/boot/grub/grub.cfg').read_text()
match = re.search(r"^menuentry 'Ubuntu'[^\n]*\{\n.*?^\}", grub, re.M | re.S)
if not match:
    raise SystemExit('Cannot find exact stock Ubuntu entry')
block = match.group(0)
if f'vmlinuz-{KERNEL}' not in block or f'initrd.img-{KERNEL}' not in block:
    raise SystemExit('Stock entry kernel differs from measured kernel')
lines = block.splitlines()
lines[0] = f"menuentry 'GPU0 Ada unlock TEST - {KERNEL}' --id '{ENTRY_ID}' {{"
for i, line in enumerate(lines):
    if re.match(r'\s*linux\s', line):
        lines[i] = line + ' gpu_unlock.targets=0000:01:00.0'
    elif re.match(r'\s*initrd\s', line):
        lines[i] = line.replace(f'initrd.img-{KERNEL}', custom_initrd.name)
block = '\n'.join(lines) + '\n'
script = pathlib.Path('/etc/grub.d/42_ada_gpu0_test')
if script.exists():
    raise SystemExit('Custom GRUB script already exists; inspect it first')
script.write_text("#!/bin/sh\nexec tail -n +3 $0\n" + block)
script.chmod(0o755)
subprocess.run(['update-grub'], check=True)
subprocess.run(['grub-script-check', '/boot/grub/grub.cfg'], check=True)
(ROOT / 'test-grub-entry.txt').write_text(block)
print('Prepared separate entry:', ENTRY_ID)
print('Stock initrd SHA256 unchanged:', initial_hash)
print('Default boot unchanged. One-shot is NOT armed. No firmware executed.')
