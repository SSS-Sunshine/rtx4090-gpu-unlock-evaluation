"""Prepare a separate four-RTX4090 one-shot-capable GRUB entry."""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
KERNEL = '7.0.0-38-generic'
TARGETS = '0000:01:00.0,0000:22:00.0,0000:41:00.0,0000:61:00.0'
ENTRY_ID = 'ada-4gpu-test-7.0.0-38'
if os.geteuid() != 0:
    raise SystemExit('Run through sudo')
if subprocess.check_output(['uname', '-r'], text=True).strip() != KERNEL:
    raise SystemExit('Kernel changed; rebuild and repeat stock baseline')
actual = []
for line in subprocess.check_output(['lspci', '-Dnn'], text=True).splitlines():
    if '10de:2684' in line and ('VGA compatible controller' in line or '3D controller' in line):
        actual.append(line.split()[0].lower())
expected = TARGETS.split(',')
if actual != expected:
    raise SystemExit(f'RTX4090 PCI inventory differs: actual={actual}, expected={expected}')
stock_initrd = pathlib.Path('/boot') / f'initrd.img-{KERNEL}'
stock_hash = hashlib.sha256(stock_initrd.read_bytes()).hexdigest()
saved_hash = (ROOT / 'boot-backup/stock-initrd.sha256').read_text().strip()
if stock_hash != saved_hash:
    raise SystemExit('Stock initrd differs from the previously measured image')
conf = ROOT / 'initramfs-config-4gpu'
if conf.exists():
    raise SystemExit('Four-GPU initramfs config already exists; inspect it first')
shutil.copytree('/etc/initramfs-tools', conf)
hook = conf / 'hooks/ada-evaluation'
hook.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / 'initramfs-hook', hook)
hook.chmod(0o755)
early = conf / 'scripts/init-top/00-ada-evaluation'
early.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / 'init-top-unlock-4gpu', early)
early.chmod(0o755)
custom_initrd = pathlib.Path('/boot') / f'initrd.img-{KERNEL}-ada-4gpu-test'
if custom_initrd.exists():
    raise SystemExit('Four-GPU test initrd already exists; inspect it first')
subprocess.run(['mkinitramfs', '-d', str(conf), '-o', str(custom_initrd), KERNEL], check=True)
listing = subprocess.check_output(['lsinitramfs', str(custom_initrd)], text=True)
for required in ['scripts/init-top/00-ada-evaluation', 'gpu_unlock.ko',
                 'booter_load_ad102.bin', 'v67-ad102-610.43.03-four-plm.bin']:
    if required not in listing:
        raise SystemExit(f'Missing initramfs item: {required}')
(ROOT / 'test-initramfs-4gpu-files.txt').write_text(listing)
if hashlib.sha256(stock_initrd.read_bytes()).hexdigest() != stock_hash:
    raise SystemExit('Stock initrd changed unexpectedly')
grub = pathlib.Path('/boot/grub/grub.cfg').read_text()
match = re.search(r"^menuentry 'Ubuntu'[^\n]*\{\n.*?^\}", grub, re.M | re.S)
if not match:
    raise SystemExit('Cannot find exact stock Ubuntu entry')
block = match.group(0)
if f'vmlinuz-{KERNEL}' not in block or f'initrd.img-{KERNEL}' not in block:
    raise SystemExit('Stock entry kernel differs from measured kernel')
lines = block.splitlines()
lines[0] = f"menuentry 'Four RTX4090 Ada unlock TEST - {KERNEL}' --id '{ENTRY_ID}' {{"
for index, line in enumerate(lines):
    if re.match(r'\s*linux\s', line):
        lines[index] = line + f' gpu_unlock.targets={TARGETS}'
    elif re.match(r'\s*initrd\s', line):
        lines[index] = line.replace(f'initrd.img-{KERNEL}', custom_initrd.name)
block = '\n'.join(lines) + '\n'
script = pathlib.Path('/etc/grub.d/43_ada_4gpu_test')
if script.exists():
    raise SystemExit('Four-GPU GRUB script already exists; inspect it first')
script.write_text("#!/bin/sh\nexec tail -n +3 $0\n" + block)
script.chmod(0o755)
subprocess.run(['update-grub'], check=True)
subprocess.run(['grub-script-check', '/boot/grub/grub.cfg'], check=True)
(ROOT / 'test-grub-4gpu-entry.txt').write_text(block)
print('Prepared separate entry:', ENTRY_ID)
print('Targets:', TARGETS)
print('Stock initrd SHA256 unchanged:', stock_hash)
print('Default boot unchanged. One-shot is NOT armed. No firmware executed.')
