import json
import pathlib
import re

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
summary = {'boot_targets': ['0000:01:00.0', '0000:22:00.0', '0000:41:00.0', '0000:61:00.0'],
           'gpus': [], 'nccl': {}, 'qwen_comparison': {}}
for index in range(4):
    directory = ROOT / f'all4-unlocked/gpu{index}'
    mma = (directory / 'mma.txt').read_text()
    numerical = (directory / 'numerical.txt').read_text()
    def rate(mode):
        match = re.search(rf'^{re.escape(mode)}\s+([0-9.]+)', mma, re.M)
        return float(match.group(1))
    summary['gpus'].append({
        'index': index,
        'fp16_fp32_instruction_TFLOPS': rate('f16/f32'),
        'bf16_fp32_instruction_TFLOPS': rate('bf16/f32'),
        'numerical_cases': numerical.count('failures='),
        'numerical_failures': sum(int(x) for x in re.findall(r'failures=(\d+)', numerical)),
        'gemm': [json.loads(line) for line in (directory / 'gemm.jsonl').read_text().splitlines()],
    })
nccl = json.loads((ROOT / 'all4-unlocked/nccl-allreduce.json').read_text())
summary['nccl'] = {
    'world_size': nccl['world_size'],
    'all_correct': all(x['correct'] for rank in nccl['rank_results'] for x in rank),
    'rank0': nccl['rank_results'][0],
}
summary['qwen_comparison'] = json.loads((ROOT / 'qwen-unlock-comparison.json').read_text())
(ROOT / 'all4-unlocked/summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
