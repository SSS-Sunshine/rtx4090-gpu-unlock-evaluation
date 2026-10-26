"""Profile fixed decode and prefill requests with NVML telemetry."""
import csv
import json
import pathlib
import shlex
import threading
import time
import urllib.request

import pynvml

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation/qwen-fp8-bottleneck')
ROOT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:8000/v1/chat/completions'
state = {'phase': 'idle', 'stop': False}
rows = []


def load_key():
    for line in pathlib.Path('/home/USER/qwen36-vllm/api.env').read_text().splitlines():
        if line.startswith('VLLM_API_KEY='):
            return shlex.split(line.split('=', 1)[1])[0]
    raise RuntimeError('VLLM_API_KEY missing')


def sample_nvml():
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
    while not state['stop']:
        now = time.time()
        for index, handle in enumerate(handles):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            def safe(fn, default=None):
                try: return fn()
                except Exception: return default
            rows.append({
                'timestamp': now, 'phase': state['phase'], 'gpu': index,
                'gpu_util_pct': util.gpu, 'memory_controller_util_pct': util.memory,
                'power_W': safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000),
                'sm_clock_MHz': safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                'memory_clock_MHz': safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
                'pcie_tx_KBps': safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)),
                'pcie_rx_KBps': safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)),
            })
        time.sleep(0.1)
    pynvml.nvmlShutdown()


def request(api_key, label, prompt, max_tokens):
    body = {
        'model': 'Qwen3.8-Flash-Next',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.0, 'max_tokens': max_tokens,
        'ignore_eos': True, 'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {'Authorization': 'Bearer ' + api_key,
                                  'Content-Type': 'application/json'})
    state['phase'] = label
    start = time.perf_counter(); first = None; usage = {}; finish = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b'data: '): continue
            data = raw[6:].strip()
            if data == b'[DONE]': break
            item = json.loads(data)
            if item.get('usage'): usage = item['usage']
            for choice in item.get('choices', []):
                delta = choice.get('delta') or {}
                if first is None and (delta.get('content') or delta.get('reasoning_content')):
                    first = time.perf_counter()
                finish = choice.get('finish_reason') or finish
    end = time.perf_counter(); state['phase'] = 'idle'
    result = {'label': label, 'prompt_tokens': usage.get('prompt_tokens'),
              'completion_tokens': usage.get('completion_tokens'),
              'ttft_s': first-start if first else None, 'total_s': end-start,
              'decode_tokens_s': (usage.get('completion_tokens', 1)-1)/(end-first) if first else None,
              'finish_reason': finish}
    print(json.dumps(result), flush=True)
    return result


thread = threading.Thread(target=sample_nvml, daemon=True); thread.start()
key = load_key(); results = []
try:
    results.append(request(key, 'decode_1024', '请持续输出关于GPU推理优化的技术说明。', 1024))
    time.sleep(1)
    long_prompt = ('hello ' * 6000) + '请只回答收到。'
    results.append(request(key, 'prefill_long', long_prompt, 64))
finally:
    state['stop'] = True; thread.join(timeout=5)

with (ROOT / 'telemetry.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summary = {'requests': results, 'telemetry': {}}
for phase in ['decode_1024', 'prefill_long']:
    phase_rows = [r for r in rows if r['phase'] == phase]
    summary['telemetry'][phase] = []
    for gpu in range(4):
        group = [r for r in phase_rows if r['gpu'] == gpu]
        item = {'gpu': gpu, 'samples': len(group)}
        for field in ['gpu_util_pct','memory_controller_util_pct','power_W','sm_clock_MHz','memory_clock_MHz','pcie_tx_KBps','pcie_rx_KBps']:
            values = [r[field] for r in group if r[field] is not None]
            item[field] = {'mean': sum(values)/len(values), 'max': max(values), 'min': min(values)} if values else None
        summary['telemetry'][phase].append(item)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
