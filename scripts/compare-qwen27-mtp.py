import json
import pathlib
import statistics

ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation')
no_mtp = json.loads((ROOT / 'qwen27-bf16-unlock-comparison.json').read_text())
mtp = json.loads((ROOT / 'qwen27-bf16-all4-unlocked-mtp/summary.json').read_text())
mtp_speeds = mtp['unlocked_tokens_per_second']
mtp_mean = statistics.mean(mtp_speeds)
no_mtp_mean = no_mtp['all4_unlocked_mean_tokens_per_second_adjusted']
stock_no_mtp = no_mtp['archived_stock_mean_tokens_per_second']
stock_mtp = json.loads(pathlib.Path('/home/USER/qwen38-ab/20260823-005636/final_mtp.json').read_text())['throughput_summary']['mean_tokens_per_second']
accepted, drafted = 432, 588
out = {
    'model': 'Qwen3.8-27B-BF16', 'tensor_parallel_size': 2,
    'mtp_num_speculative_tokens': 1, 'prefix_caching': False,
    'current_all4_unlocked_no_mtp_tokens_per_second': no_mtp_mean,
    'current_all4_unlocked_mtp_tokens_per_second': mtp_speeds,
    'current_all4_unlocked_mtp_mean_tokens_per_second': mtp_mean,
    'mtp_gain_pct_same_boot_current_software': (mtp_mean / no_mtp_mean - 1) * 100,
    'archived_stock_no_mtp_tokens_per_second': stock_no_mtp,
    'combined_gain_pct_vs_archived_stock_no_mtp': (mtp_mean / stock_no_mtp - 1) * 100,
    'archived_stock_mtp_tokens_per_second': stock_mtp,
    'gain_pct_vs_archived_stock_mtp': (mtp_mean / stock_mtp - 1) * 100,
    'mtp_accepted_tokens': accepted, 'mtp_drafted_tokens': drafted,
    'mtp_acceptance_pct': accepted / drafted * 100,
    'current_no_mtp_output_sha256': no_mtp['unlocked_answer_sha256'],
    'current_mtp_output_sha256': mtp['unlocked_answer_sha256'],
    'current_output_hashes_match': [a == b for a, b in zip(no_mtp['unlocked_answer_sha256'], mtp['unlocked_answer_sha256'])],
}
(ROOT / 'qwen27-bf16-mtp-unlock-comparison.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
