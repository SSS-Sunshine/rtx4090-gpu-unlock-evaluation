import json
import pathlib
import statistics
import subprocess

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
VIDEOS = pathlib.Path('/data/apps/ComfyUI/output/video')
summary = {}
for phase in ['stock', 'unlocked', 'gpu1-stock']:
    runs = [r for r in json.loads((ROOT / phase / 'results.json').read_text()) if not r['warmup']]
    summary[phase] = {
        'total_s': [r['total_s'] for r in runs],
        'sampler_s': [r['sampler_s'] for r in runs],
        'total_median_s': statistics.median(r['total_s'] for r in runs),
        'sampler_median_s': statistics.median(r['sampler_s'] for r in runs),
    }
summary['gpu0_vs_gpu1_speedup'] = summary['gpu1-stock']['total_median_s'] / summary['unlocked']['total_median_s']
summary['gpu0_vs_gpu1_sampler_speedup'] = summary['gpu1-stock']['sampler_median_s'] / summary['unlocked']['sampler_median_s']


def hash_media(path, stream):
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-map', f'0:{stream}:0']
    cmd += ['-c:v', 'rawvideo'] if stream == 'v' else ['-c:a', 'pcm_s16le']
    return subprocess.check_output(cmd + ['-f', 'hash', '-'], text=True).strip()


summary['video_comparison'] = []
for seed in [101, 102, 103]:
    a = VIDEOS / f'ada_evaluation_unlocked_{seed}_00001_.mp4'
    b = VIDEOS / f'ada_evaluation_gpu1-stock_{seed}_00001_.mp4'
    summary['video_comparison'].append({
        'seed': seed,
        'decoded_video_identical': hash_media(a, 'v') == hash_media(b, 'v'),
        'decoded_audio_identical': hash_media(a, 'a') == hash_media(b, 'a'),
    })
framework = {}
for gpu in [0, 1]:
    framework[gpu] = json.loads((ROOT / 'framework' / f'gpu{gpu}.json').read_text())
summary['framework'] = []
for name, unlocked in framework[0]['tests'].items():
    stock = framework[1]['tests'][name]
    summary['framework'].append({
        'test': name, 'gpu1_stock': stock, 'gpu0_unlocked': unlocked,
        'wall_speedup': stock['median_wall_ms'] / unlocked['median_wall_ms'],
    })
summary['gemm'] = []
stock = [json.loads(line) for line in (ROOT / 'gpu1-stock/gemm.jsonl').read_text().splitlines()]
unlocked = [json.loads(line) for line in (ROOT / 'unlocked/gemm.jsonl').read_text().splitlines()]
for a, b in zip(stock, unlocked):
    assert (a['mode'], a['n']) == (b['mode'], b['n'])
    summary['gemm'].append({
        'mode': a['mode'], 'n': a['n'], 'gpu1_TFLOPS': a['median_TF_s'],
        'gpu0_TFLOPS': b['median_TF_s'], 'speedup': b['median_TF_s'] / a['median_TF_s'],
        'gpu1_failures': a['failures'], 'gpu0_failures': b['failures'],
    })
(ROOT / 'gpu1-comparison.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
