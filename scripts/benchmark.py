"""Matched ComfyUI MiniMax-H3 runs. Does not modify firmware or services."""
import argparse
import asyncio
import hashlib
import json
import pathlib
import subprocess
import time

import aiohttp

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
PROMPT = ('A cinematic continuous shot of a small red robot walking through a sunlit '
          'greenhouse. The camera tracks slowly to the right. Leaves sway gently, '
          'consistent robot identity and smooth natural movement. Soft footsteps '
          'and birdsong, no speech, no subtitles.')


def graph(seed, prefix):
    def n(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    return {
        '1': n('UNETLoader', unet_name='minimax_h3_fl2va_pruned_int8_convrot.safetensors', weight_dtype='default'),
        '2': n('CLIPLoader', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type='minimax', device='default'),
        '3': n('VAELoader', vae_name='minimax_h3_video_vae_fp16.safetensors'),
        '4': n('VAELoader', vae_name='minimax_h3_audio_vae_fp32.safetensors'),
        '5': n('MiniMaxH3ImageToVideo', clip=['2', 0], vae=['3', 0], prompt=PROMPT,
               width=864, height=480, length=124),
        '6': n('RandomNoise', noise_seed=seed),
        '7': n('BasicGuider', model=['1', 0], conditioning=['5', 0]),
        '8': n('KSamplerSelect', sampler_name='res_multistep'),
        '9': n('BasicScheduler', model=['1', 0], scheduler='simple', steps=20, denoise=1.0),
        '10': n('SamplerCustomAdvanced', noise=['6', 0], guider=['7', 0], sampler=['8', 0],
                sigmas=['9', 0], latent_image=['5', 1]),
        '11': n('VAEDecode', samples=['10', 0], vae=['3', 0]),
        '12': n('VAEDecodeAudio', samples=['10', 0], vae=['4', 0]),
        '13': n('CreateVideo', images=['11', 0], audio=['12', 0], fps=24.0),
        '14': n('SaveVideo', video=['13', 0], filename_prefix=prefix, format='auto', codec='auto'),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['stock', 'unlocked', 'gpu1-stock'])
    args = parser.parse_args()
    out = ROOT / args.phase
    out.mkdir(parents=True, exist_ok=True)
    baseline = graph(101, 'video/ada_benchmark')
    (out / 'workflow.json').write_text(json.dumps(baseline, indent=2))
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit', '--format=csv'], text=True)
    (out / 'gpu.csv').write_text(identity)
    (out / 'kernel.txt').write_text(subprocess.check_output(['uname', '-r'], text=True))
    (out / 'cmdline.txt').write_text(pathlib.Path('/proc/cmdline').read_text())
    (out / 'comfy-commit.txt').write_text(subprocess.check_output(['git', '-C', '/data/apps/ComfyUI', 'rev-parse', 'HEAD'], text=True))
    telemetry_file = (out / 'telemetry.csv').open('w')
    telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.gr,clocks.mem', '--format=csv', '-lms', '1000'], stdout=telemetry_file)
    results = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get('http://127.0.0.1:8189/queue') as r:
                queue = await r.json()
                if queue['queue_running'] or queue['queue_pending']:
                    raise RuntimeError('ComfyUI queue is not empty; refusing to interfere')
            for seed in [100, 101, 102, 103]:
                client = f'ada-evaluation-{args.phase}-{seed}'
                prefix = f'video/ada_evaluation_{args.phase}_{seed}'
                async with session.ws_connect(f'http://127.0.0.1:8189/ws?clientId={client}', heartbeat=30) as ws:
                    g = graph(seed, prefix)
                    started = time.perf_counter()
                    async with session.post('http://127.0.0.1:8189/prompt', json={'prompt': g, 'client_id': client}) as r:
                        body = await r.json()
                        if r.status != 200:
                            raise RuntimeError(f'Workflow rejected: {body}')
                    pid = body['prompt_id']
                    events = []
                    current_node = None
                    current_start = None
                    node_seconds = {}
                    print(json.dumps({'phase': args.phase, 'seed': seed, 'prompt_id': pid, 'started': True}), flush=True)
                    while True:
                        msg = await ws.receive(timeout=1800)
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(msg.data)
                        typ, data = event.get('type'), event.get('data', {})
                        if data.get('prompt_id') != pid:
                            continue
                        elapsed = time.perf_counter() - started
                        if typ != 'progress':
                            events.append({'elapsed_s': elapsed, **event})
                        if typ == 'executing':
                            if current_node is not None:
                                node_seconds[current_node] = elapsed - current_start
                            current_node, current_start = data.get('node'), elapsed
                        if typ in ('execution_error', 'execution_interrupted'):
                            raise RuntimeError(json.dumps(event))
                        if typ == 'execution_success' or (typ == 'executing' and data.get('node') is None):
                            if current_node is not None:
                                node_seconds[current_node] = elapsed - current_start
                            break
                    async with session.get(f'http://127.0.0.1:8189/history/{pid}') as r:
                        history = await r.json()
                    (out / f'history-{seed}.json').write_text(json.dumps(history, indent=2))
                    (out / f'events-{seed}.json').write_text(json.dumps(events, indent=2))
                    result = {'seed': seed, 'warmup': seed == 100, 'total_s': elapsed,
                              'node_seconds': node_seconds,
                              'sampler_s': node_seconds.get('10'), 'prompt_id': pid}
                    results.append(result)
                    (out / 'results.json').write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()


if __name__ == '__main__':
    asyncio.run(main())
