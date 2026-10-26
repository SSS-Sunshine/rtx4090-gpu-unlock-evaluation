import asyncio
import importlib.util
import json
import pathlib
import shlex

import aiohttp


async def main():
    spec = importlib.util.spec_from_file_location('qwen_bench', '/data/benchmarks/qwen-real-project.py')
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    env = {}
    for line in pathlib.Path('/home/USER/qwen36-vllm/api.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            env[k.strip()] = shlex.split(v)[0]
    cases = json.loads((bench.ROOT / 'cases.json').read_text())
    task, prompt = next(c for c in cases if c[0] == 'robot_training_config')
    results = json.loads((bench.ROOT / 'results.json').read_text())
    capped = [r for r in results if r['task'] == task]
    (bench.ROOT / 'pilot16384-config-results.json').write_text(json.dumps(capped, ensure_ascii=False, indent=2))
    results = [r for r in results if r['task'] != task]
    first = json.loads((bench.ROOT / 'robot_training_config-native-context.json').read_text())
    results.append(first)
    (bench.ROOT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        for rep in [2, 3]:
            result = await bench.request(session, env['VLLM_API_KEY'], task, prompt, f'native-context-{rep}', token_limit=None)
            results.append(result)
            (bench.ROOT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))


asyncio.run(main())
