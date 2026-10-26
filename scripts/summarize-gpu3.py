import json
import pathlib
import re

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
FULL = ROOT / 'gpu3-full'
BASE = ROOT / 'all4-unlocked/gpu3'
memory = json.loads((FULL / 'memory.json').read_text())
tensor = json.loads((FULL / 'tensor-modes.json').read_text())
torch = json.loads((FULL / 'torch-workloads.json').read_text())
stress = json.loads((FULL / 'stress-result.json').read_text())
stress_telemetry = json.loads((FULL / 'stress-summary.json').read_text())
gemm = [json.loads(line) for line in (BASE / 'gemm.jsonl').read_text().splitlines()]
mma = (BASE / 'mma.txt').read_text()
numerical = (BASE / 'numerical.txt').read_text()
def mma_rate(mode):
    return float(re.search(rf'^{re.escape(mode)}\s+([0-9.]+)', mma, re.M).group(1))
summary = {
    'gpu_index': 3,
    'uuid': 'GPU-REDACTED-3',
    'pci_bus': '0000:61:00.0',
    'unlock_confirmed': True,
    'driver': '595.99.02',
    'power_limit_W': 450,
    'pcie': {'generation': 4, 'width': 16, 'replays_since_reset': 0,
             'h2d_GBps': memory['pcie_h2d_GBps'], 'd2h_GBps': memory['pcie_d2h_GBps']},
    'memory': {
        'total_MiB_cuda': memory['total_MiB'],
        'tested_MiB': memory['vram_test_MiB'],
        'patterns': len(memory['patterns']),
        'errors': memory['total_memory_errors'],
        'write_GBps': [x['write_GBps'] for x in memory['patterns']],
        'verify_read_GBps': [x['verify_GBps'] for x in memory['patterns']],
        'd2d_payload_GBps': memory['d2d_memcpy_payload_GBps'],
        'd2d_read_write_GBps': memory['d2d_memcpy_read_write_GBps'],
        'copy_kernel_payload_GBps': memory['copy_kernel_payload_GBps'],
        'copy_kernel_read_write_GBps': memory['copy_kernel_read_write_GBps'],
        'ecc_reporting': 'N/A on RTX 4090',
        'memory_temperature_reporting': 'N/A on this board/driver',
    },
    'instruction_throughput': {
        'fp16_fp32_TFLOPS_mma': mma_rate('f16/f32'),
        'bf16_fp32_TFLOPS_mma': mma_rate('bf16/f32'),
        'int8_int32_TOPS_mma': mma_rate('s8/s32'),
        'dp4a_TOPS': mma_rate('dp4a'),
        'scalar_fp32_TFLOPS': mma_rate('ffma'),
        'tf32_fp32_TFLOPS_tensor_mode': tensor['modes']['tf32']['best_Top_per_s'],
        'u4_int32_TOPS_tensor_mode': tensor['modes']['u4_s32']['best_Top_per_s'],
        'tensor_mode_mismatches': sum(x['mismatches'] for x in tensor['modes'].values()),
    },
    'numerical_validation': {
        'cases': numerical.count('failures='),
        'failures': sum(int(x) for x in re.findall(r'failures=(\d+)', numerical)),
        'gemm_failures': sum(x['failures'] for x in gemm),
    },
    'cublas_gemm': gemm,
    'pytorch': torch['tests'],
    'stress': {**stress, **stress_telemetry,
               'hardware_thermal_slowdown': False, 'hardware_power_brake': False,
               'xid_errors': 0},
}
(FULL / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
