"""Generic streamed OpenAI-compatible fixed-output concurrency benchmark."""
import asyncio
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

import aiohttp

URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ["MODEL"]
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))
PROMPT = pathlib.Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else (
    "Review this configuration and identify correctness, concurrency, and performance risks."
)


async def request(session, index, start_event):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    await start_event.wait()
    started, first, last = time.perf_counter(), None, None
    usage, chunks = {}, []
    async with session.post(
        f"{URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
    ) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            item = json.loads(line[6:])
            usage = item.get("usage") or usage
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    now = time.perf_counter()
                    first, last = first or now, now
                    chunks.append(text)
    ended = time.perf_counter()
    tokens = usage.get("completion_tokens", 0)
    return {
        "index": index,
        "completion_tokens": tokens,
        "ttft_s": first - started if first else None,
        "total_s": ended - started,
        "decode_tokens_s": (tokens - 1) / (last - first) if first and last and last > first else None,
        "output_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest(),
    }


async def main():
    timeout = aiohttp.ClientTimeout(total=1800)
    start_event = asyncio.Event()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(request(session, i, start_event)) for i in range(CONCURRENCY)]
        started = time.perf_counter()
        start_event.set()
        runs = await asyncio.gather(*tasks)
        wall = time.perf_counter() - started
    total = sum(run["completion_tokens"] for run in runs)
    print(json.dumps({
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "max_tokens_each": MAX_TOKENS,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "wall_s": wall,
        "total_completion_tokens": total,
        "aggregate_completion_tokens_s": total / wall,
        "median_request_tokens_s": statistics.median(run["decode_tokens_s"] for run in runs),
        "median_ttft_s": statistics.median(run["ttft_s"] for run in runs),
        "runs": runs,
    }, indent=2))


asyncio.run(main())
