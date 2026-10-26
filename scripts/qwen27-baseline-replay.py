"""Replay the archived Qwen3.8-27B BF16 no-MTP throughput case."""
import hashlib
import json
import pathlib
import shlex
import statistics
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation/qwen27-bf16-all4-unlocked-mtp')
PROMPT = '请连续写一段关于人工智能推理优化的中文说明，不要使用标题。'
URL = 'http://127.0.0.1:8000/v1/chat/completions'


def key():
    for line in pathlib.Path('/home/USER/qwen36-vllm/api.env').read_text().splitlines():
        if line.startswith('VLLM_API_KEY='):
            return shlex.split(line.split('=', 1)[1])[0]
    raise RuntimeError('VLLM_API_KEY missing')


def request(api_key, label):
    body = {
        'model': 'Qwen3.8-27B',
        'messages': [{'role': 'user', 'content': PROMPT}],
        'temperature': 0.0,
        'max_tokens': 256,
        'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': True},
    }
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode(),
                                 {'Authorization': 'Bearer ' + api_key,
                                  'Content-Type': 'application/json; charset=utf-8'})
    start, first = time.perf_counter(), None
    answer, reasoning, usage, finish = [], [], {}, None
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            data = raw[6:].strip()
            if data == b'[DONE]':
                break
            item = json.loads(data)
            if item.get('usage'):
                usage = item['usage']
            for choice in item.get('choices', []):
                delta = choice.get('delta') or {}
                content = delta.get('content') or ''
                thought = delta.get('reasoning_content') or ''
                if content or thought:
                    first = first or time.perf_counter()
                answer.append(content)
                reasoning.append(thought)
                finish = choice.get('finish_reason') or finish
    end = time.perf_counter()
    output = ''.join(answer)
    all_output = ''.join(reasoning) + output
    tokens = usage.get('completion_tokens', 0)
    result = {
        'run': label, 'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': tokens, 'ttft_seconds': first - start,
        'total_seconds': end - start,
        # Match the archived harness: the first streamed token starts the decode clock.
        'generation_tokens_per_second': max(tokens - 1, 1) / (end - first),
        'finish_reason': finish,
        'answer_sha256': hashlib.sha256(output.encode()).hexdigest(),
        'combined_output_sha256': hashlib.sha256(all_output.encode()).hexdigest(),
        'reasoning_characters': len(''.join(reasoning)),
        'content_characters': len(output), 'reasoning': ''.join(reasoning), 'content': output,
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ('reasoning', 'content')}, ensure_ascii=False), flush=True)
    return result


ROOT.mkdir(parents=True, exist_ok=True)
deadline = time.monotonic() + 900
while True:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        pass
    if time.monotonic() > deadline:
        raise RuntimeError('Qwen3.8-27B did not become healthy within 15 minutes')
    print('Waiting for Qwen3.8-27B loading...', flush=True)
    time.sleep(15)
archive = json.loads(pathlib.Path('/home/USER/qwen38-ab/20260823-005636/baseline.json').read_text())
(ROOT / 'archived-baseline.json').write_text(json.dumps(archive, ensure_ascii=False, indent=2))
(ROOT / 'gpu.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,driver_version,power.limit', '--format=csv'], text=True))
api_key = key()
request(api_key, 'warmup')
results = [request(api_key, index) for index in [1, 2, 3]]
summary = {
    'archived_stock_mean_tokens_per_second': archive['throughput_summary']['mean_tokens_per_second'],
    'unlocked_tokens_per_second': [x['generation_tokens_per_second'] for x in results],
    'unlocked_mean_tokens_per_second': statistics.mean(x['generation_tokens_per_second'] for x in results),
    'unlocked_median_tokens_per_second': statistics.median(x['generation_tokens_per_second'] for x in results),
    'stock_output_sha256': [x['output_sha256'] for x in archive['throughput']],
    'unlocked_answer_sha256': [x['answer_sha256'] for x in results],
    'unlocked_combined_output_sha256': [x['combined_output_sha256'] for x in results],
    'finish_reasons': [x['finish_reason'] for x in results],
    'speed_change_pct_vs_archived_mean': (statistics.mean(x['generation_tokens_per_second'] for x in results) /
                                           archive['throughput_summary']['mean_tokens_per_second'] - 1) * 100,
    'results': results,
}
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, ensure_ascii=False, indent=2))
