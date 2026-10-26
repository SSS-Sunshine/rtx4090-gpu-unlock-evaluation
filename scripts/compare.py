"""Compare matched local workload runs and decoded video/audio outputs."""
import csv
import hashlib
import json
import pathlib
import re
import statistics
import subprocess

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
VIDEOS = pathlib.Path('/data/apps/ComfyUI/output/video')


def summarize(phase):
    directory = ROOT / phase
    data = json.loads((directory / 'results.json').read_text())
    measured = [x for x in data if not x['warmup']]
    if len(measured) != 3:
        raise RuntimeError(f'{phase} does not contain 3 completed measured runs')
    out = {
        'total_s': [r['total_s'] for r in measured],
        'sampler_s': [r['sampler_s'] for r in measured],
        'total_median_s': statistics.median(r['total_s'] for r in measured),
        'sampler_median_s': statistics.median(r['sampler_s'] for r in measured),
    }
    rows = list(csv.DictReader((directory / 'telemetry.csv').open(), skipinitialspace=True))
    gpu0 = [r for r in rows if r.get('index') == '0']
    for field in ['power.draw [W]', 'memory.used [MiB]', 'temperature.gpu']:
        values = []
        for r in gpu0:
            try:
                values.append(float(r[field].split()[0]))
            except (KeyError, ValueError):
                pass
        if values:
            out['peak_' + field] = max(values)
    out['gemm'] = [json.loads(line) for line in (directory / 'gemm.jsonl').read_text().splitlines()]
    out['mma_log'] = (directory / 'mma.txt').read_text()
    return out


def decoded_hash(path, audio=False):
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path)]
    if audio:
        cmd += ['-vn', '-acodec', 'pcm_s16le', '-f', 's16le', '-']
    else:
        cmd += ['-an', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    h = hashlib.sha256()
    while chunk := p.stdout.read(1024 * 1024):
        h.update(chunk)
    if p.wait() != 0:
        raise RuntimeError('ffmpeg decode failed')
    return h.hexdigest()


stock, unlocked = summarize('stock'), summarize('unlocked')
result = {'stock': stock, 'unlocked': unlocked}
for metric in ['total', 'sampler']:
    a, b = stock[metric + '_median_s'], unlocked[metric + '_median_s']
    result[metric + '_speedup'] = a / b
    result[metric + '_time_reduction_pct'] = (1 - b / a) * 100
result['video_comparison'] = []
for seed in [101, 102, 103]:
    a = VIDEOS / f'ada_evaluation_stock_{seed}_00001_.mp4'
    b = VIDEOS / f'ada_evaluation_unlocked_{seed}_00001_.mp4'
    video_equal = decoded_hash(a) == decoded_hash(b)
    audio_equal = decoded_hash(a, True) == decoded_hash(b, True)
    item = {'seed': seed, 'decoded_video_identical': video_equal, 'decoded_audio_identical': audio_equal}
    if not video_equal:
        p = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(a), '-i', str(b),
                            '-lavfi', '[0:v][1:v]ssim', '-an', '-f', 'null', '-'],
                           text=True, capture_output=True, check=True)
        lines = [line for line in p.stderr.splitlines() if 'SSIM ' in line]
        item['ssim_log'] = lines[-1] if lines else 'not reported'
        match = re.search(r'All:([0-9.]+)', item['ssim_log'])
        if match:
            item['ssim'] = float(match.group(1))
    result['video_comparison'].append(item)
(ROOT / 'comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
