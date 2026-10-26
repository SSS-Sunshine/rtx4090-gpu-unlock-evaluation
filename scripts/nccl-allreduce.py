"""Four-GPU NCCL all-reduce health and throughput check."""
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
results = []
for mib in [1, 16, 64, 256]:
    count = mib * 1024 * 1024 // 4
    tensor = torch.full((count,), float(rank + 1), device='cuda', dtype=torch.float32)
    for _ in range(10):
        dist.all_reduce(tensor)
        tensor.fill_(float(rank + 1))
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        tensor.fill_(float(rank + 1))
        dist.barrier()
        start = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(tensor)
            tensor.fill_(float(rank + 1))
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) / 20)
    tensor.fill_(float(rank + 1))
    dist.all_reduce(tensor)
    expected = world * (world + 1) / 2
    correct = bool(torch.all(tensor == expected).item())
    sec = statistics.median(samples)
    algorithm_gbps = (mib * 1024 * 1024) / sec / 1e9
    bus_gbps = algorithm_gbps * 2 * (world - 1) / world
    results.append({'size_MiB': mib, 'median_ms': sec * 1000,
                    'algorithm_GBps': algorithm_gbps, 'bus_GBps': bus_gbps,
                    'correct': correct})
gathered = [None] * world if rank == 0 else None
dist.gather_object(results, gathered, dst=0)
if rank == 0:
    print(json.dumps({'world_size': world, 'rank_results': gathered}, indent=2))
dist.destroy_process_group()
