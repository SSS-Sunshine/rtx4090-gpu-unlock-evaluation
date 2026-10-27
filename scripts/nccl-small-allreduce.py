"""Two-or-more-rank NCCL AllReduce latency sweep for 4 KiB through 1 MiB."""
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
results = []

for kib in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    tensor = torch.ones(kib * 1024 // 4, device="cuda", dtype=torch.float32)
    for _ in range(50):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / 200)
    tensor.fill_(float(rank + 1))
    dist.all_reduce(tensor)
    correct = bool(torch.all(tensor == world * (world + 1) / 2).item())
    seconds = statistics.median(samples)
    algorithm_gbps = (kib * 1024) / seconds / 1e9
    results.append({
        "size_KiB": kib,
        "median_us": seconds * 1e6,
        "algorithm_GBps": algorithm_gbps,
        "bus_GBps": algorithm_gbps * 2 * (world - 1) / world,
        "correct": correct,
    })

gathered = [None] * world if rank == 0 else None
dist.gather_object(results, gathered, dst=0)
if rank == 0:
    print(json.dumps({"world_size": world, "rank_results": gathered}, indent=2))
dist.destroy_process_group()
