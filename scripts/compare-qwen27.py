import json
import pathlib
import statistics

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
current_path = ROOT / 'qwen27-bf16-all4-unlocked-thinking/summary.json'
current = json.loads(current_path.read_text())
archive = json.loads(pathlib.Path('/home/USER/qwen38-ab/20260823-005636/baseline.json').read_text())
adjusted = []
for result in current['results']:
    adjusted.append((result['completion_tokens'] - 1) /
                    (result['total_seconds'] - result['ttft_seconds']))
stock = archive['throughput_summary']['mean_tokens_per_second']
mean = statistics.mean(adjusted)
out = {
    'model': 'Qwen3.8-27B-BF16', 'tensor_parallel_size': 2,
    'mtp': False, 'prefix_caching': False,
    'archived_stock_tokens_per_second': [x['generation_tokens_per_second'] for x in archive['throughput']],
    'archived_stock_mean_tokens_per_second': stock,
    'all4_unlocked_tokens_per_second_adjusted': adjusted,
    'all4_unlocked_mean_tokens_per_second_adjusted': mean,
    'speed_change_pct': (mean / stock - 1) * 100,
    'prompt_tokens_match': all(x['prompt_tokens'] == archive['throughput'][0]['prompt_tokens'] for x in current['results']),
    'completion_tokens_match': all(x['completion_tokens'] == archive['throughput'][0]['completion_tokens'] for x in current['results']),
    'stock_output_sha256': [x['output_sha256'] for x in archive['throughput']],
    'unlocked_answer_sha256': [x['answer_sha256'] for x in current['results']],
    'output_hashes_match': [x['output_sha256'] == y['answer_sha256'] for x, y in zip(archive['throughput'], current['results'])],
    'comparison_limit': 'Prompt and completion token counts match, but output hashes and reasoning/content partition differ from the archived stock run; a current-software stock reboot is required for a strict bitwise A/B.',
}
(ROOT / 'qwen27-bf16-unlock-comparison.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
