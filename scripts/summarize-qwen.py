import json
import os
import pathlib
import statistics

phase = os.environ.get('QWEN_BENCH_PHASE', 'qwen-flash-gpu0-unlocked')
if phase not in {'qwen-flash-gpu0-unlocked', 'qwen-flash-all4-unlocked', 'qwen-flash-all4-persistent'}:
    raise ValueError('Unsupported phase')
ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation') / phase
results = json.loads((ROOT / 'results.json').read_text())
summary = {'model': 'Qwen3.8-Flash-Next', 'precision': 'FP8', 'TP': 4,
           'unlocked_gpu_indices': [0, 1, 2, 3] if phase in {'qwen-flash-all4-unlocked', 'qwen-flash-all4-persistent'} else [0], 'thinking': True,
           'reasoning_effort': 'xhigh (model template default)',
           'max_output_tokens_for_test': '16384 for short tasks; no explicit cap for full config task',
           'gpu_memory_utilization_for_this_test': 0.97, 'tasks': []}
for name in dict.fromkeys(r['task'] for r in results):
    runs = [r for r in results if r['task'] == name]
    task = {'task': name, 'repeats': len(runs), 'finish_reasons': [r['finish_reason'] for r in runs],
            'prompt_tokens': [r['usage'].get('prompt_tokens') for r in runs],
            'completion_tokens': [r['usage'].get('completion_tokens') for r in runs],
            'reasoning_details': [r['usage'].get('completion_tokens_details') for r in runs]}
    for metric in ['ttft_any_s', 'ttft_answer_s', 'total_s', 'decode_tokens_per_s_approx', 'completion_tokens_per_total_s']:
        values = [r[metric] for r in runs if r[metric] is not None]
        task[metric + '_median'] = statistics.median(values) if values else None
    summary['tasks'].append(task)
validation = ROOT / 'generated-test-validation.json'
if validation.exists():
    summary['generated_test_validation'] = json.loads(validation.read_text())
summary['all_requests_completed'] = len(results) == 9 and all(r['finish_reason'] == 'stop' for r in results)
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
