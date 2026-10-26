import json
import pathlib

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
old = json.loads((ROOT / 'qwen-flash-all4-unlocked/summary.json').read_text())
new = json.loads((ROOT / 'qwen-flash-all4-persistent/summary.json').read_text())
out = {'baseline': 'four GPUs unlocked with one-shot boot entry',
       'candidate': 'same boot after permanent DKMS/GRUB installation', 'tasks': []}
for a,b in zip(old['tasks'],new['tasks']):
    assert a['task']==b['task']
    old_speed=a['decode_tokens_per_s_approx_median'];new_speed=b['decode_tokens_per_s_approx_median']
    out['tasks'].append({'task':a['task'],'before_tokens_per_second':old_speed,
                         'after_tokens_per_second':new_speed,
                         'change_pct':(new_speed/old_speed-1)*100,
                         'before_ttft_s':a['ttft_any_s_median'],
                         'after_ttft_s':b['ttft_any_s_median']})
(ROOT/'qwen-fp8-persistent-comparison.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
