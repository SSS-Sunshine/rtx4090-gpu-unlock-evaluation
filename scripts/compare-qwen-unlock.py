import json
import pathlib

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
a = json.loads((ROOT / 'qwen-flash-gpu0-unlocked/summary.json').read_text())
b = json.loads((ROOT / 'qwen-flash-all4-unlocked/summary.json').read_text())
out = {'baseline': 'only GPU0 unlocked', 'candidate': 'GPU0-3 unlocked', 'tasks': []}
for old, new in zip(a['tasks'], b['tasks']):
    assert old['task'] == new['task']
    row = {'task': old['task']}
    for metric in ['decode_tokens_per_s_approx_median', 'ttft_any_s_median',
                   'ttft_answer_s_median', 'total_s_median']:
        row['gpu0_only_' + metric] = old[metric]
        row['all4_' + metric] = new[metric]
    row['decode_speed_change_pct'] = (
        new['decode_tokens_per_s_approx_median'] /
        old['decode_tokens_per_s_approx_median'] - 1) * 100
    out['tasks'].append(row)
(ROOT / 'qwen-unlock-comparison.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
