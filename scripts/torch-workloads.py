"""Same-boot GPU0/GPU1 framework benchmarks, no downloads or firmware writes.

GPT-2 and ResNet-50 use seeded synthetic inputs and random initial weights.
They measure architecture training performance, not model accuracy.
"""
import argparse
import gc
import json
import pathlib
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation/framework')
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def time_steps(fn, iterations, warmups=5):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        wall = time.perf_counter()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append({'cuda_ms_per_step': start.elapsed_time(end) / iterations,
                      'wall_ms_per_step': (time.perf_counter() - wall) * 1000 / iterations})
    return {'repetitions': times,
            'median_cuda_ms': statistics.median(t['cuda_ms_per_step'] for t in times),
            'median_wall_ms': statistics.median(t['wall_ms_per_step'] for t in times)}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run(gpu):
    torch.cuda.set_device(gpu)
    torch.manual_seed(20261025)
    cleanup()
    result = {'gpu_index': gpu, 'gpu_name': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'tests': {}}
    ROOT.mkdir(parents=True, exist_ok=True)

    def save_progress(name, test):
        result['tests'][name] = test
        (ROOT / f'gpu{gpu}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'gpu': gpu, 'test': name, **test}), flush=True)
    for dtype in [torch.float16, torch.bfloat16]:
        a = torch.randn(4096, 4096, device='cuda', dtype=dtype)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        test = time_steps(lambda: torch.mm(a, b, out=c), 40)
        test['finite'] = bool(torch.isfinite(c).all())
        test['TFLOPS'] = 2 * 4096**3 / (test['median_cuda_ms'] * 1e9)
        save_progress('torch_mm_' + str(dtype), test)
        del a, b, c
    cleanup()
    q = torch.randn(2, 16, 4096, 64, device='cuda', dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    test = time_steps(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), 30)
    test['finite'] = bool(torch.isfinite(F.scaled_dot_product_attention(q, k, v, is_causal=True)).all())
    save_progress('sdpa_bf16_b2_h16_s4096_d64', test)
    del q, k, v
    cleanup()
    torch.manual_seed(20261025)
    model = resnet50(weights=None).cuda().train()
    x = torch.randn(32, 3, 224, 224, device='cuda')
    labels = torch.randint(0, 1000, (32,), device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    losses = []

    def resnet_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

    test = time_steps(resnet_step, 10)
    test['finite_loss'] = bool(torch.isfinite(torch.stack(losses)).all())
    test['first_loss'] = float(losses[0])
    test['last_loss'] = float(losses[-1])
    test['images_per_second'] = 32 * 1000 / test['median_wall_ms']
    test['peak_allocated_MiB'] = torch.cuda.max_memory_allocated() / 2**20
    save_progress('resnet50_bf16_train_b32_224', test)
    del model, optimizer, x, labels, losses, resnet_step
    cleanup()
    torch.manual_seed(20261025)
    config = GPT2Config(vocab_size=50257, n_positions=512, n_ctx=512,
                        n_embd=1024, n_layer=24, n_head=16,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                        use_cache=False)
    config._attn_implementation = 'sdpa'
    model = GPT2LMHeadModel(config).cuda().train()
    inputs = torch.randint(0, config.vocab_size, (4, 512), device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    losses = []

    def gpt_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=inputs, labels=inputs).loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

    test = time_steps(gpt_step, 5)
    test['finite_loss'] = bool(torch.isfinite(torch.stack(losses)).all())
    test['first_loss'] = float(losses[0])
    test['last_loss'] = float(losses[-1])
    test['tokens_per_second'] = 4 * 512 * 1000 / test['median_wall_ms']
    test['peak_allocated_MiB'] = torch.cuda.max_memory_allocated() / 2**20
    test['parameters'] = sum(p.numel() for p in model.parameters())
    save_progress('gpt2_medium_bf16_train_b4_s512', test)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f'gpu{gpu}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int, choices=[0, 1, 2, 3])
    run(parser.parse_args().gpu)
