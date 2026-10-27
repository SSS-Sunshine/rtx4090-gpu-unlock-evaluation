"""Build the public v2 summary from sanitized evidence files."""
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v2"


def load(name):
    return json.loads((DATA / name).read_text())


def mean_runs(prefix):
    rows = [load(f"{prefix}-run1.json"), load(f"{prefix}-run2.json")]
    values = [row["aggregate_completion_tokens_s"] for row in rows]
    return {"runs": values, "mean": statistics.mean(values)}


tp4_shm = mean_runs("qwen-flash-tp4-shm-conc8")
tp4_p2p = mean_runs("qwen-flash-tp4-p2p-conc8")
tep4_shm = mean_runs("qwen-flash-tep4-shm-conc8")
tep4_p2p = mean_runs("qwen-flash-tep4-p2p-conc8")
seq1 = load("qwen-flash-tp4-seq1.json")
nccl = load("nccl-allreduce-p2p-ab.json")

summary = {
    "schema": 1,
    "scope": {
        "gpus": "4x modified RTX 4090 48GB",
        "other_gpus_excluded": "2x A30",
        "kernel": "7.0.0-38-generic",
        "driver": "595.99.02 open kernel module",
        "tensor_unlock": "gpu-unlock-ada 1.1.1",
        "p2p": "dynamic per-allocation BAR1, 32GB BAR1 / 48GB framebuffer",
    },
    "tensor_issue_rate": {
        "stock_f16_input_f32_accumulate_TFps": 164.9,
        "unlocked_four_gpu_range_TFps": [328.0, 334.4],
        "stock_bf16_input_f32_accumulate_TFps": 164.8,
        "unlocked_bf16_four_gpu_range_TFps": [331.0, 334.2],
        "unlocked_fp8_e4m3_four_gpu_range_TFps": [660.1, 666.1],
        "note": "FP16/BF16 retain about 2x issue rate with P2P active; FP8 does not receive another 2x multiplier.",
    },
    "p2p_correctness": {
        "directed_pairs": 12,
        "rounds": 3,
        "sizes": 5,
        "cases": 180,
        "all_passed": True,
        "resident_GiB": 34,
        "window_MiB": 64,
        "resident_boundary_bidirectional_pass": True,
    },
    "nccl_allreduce": {
        "rows": nccl["comparison"],
        "p2p_enabled_transport": nccl["cases"]["p2p-enabled"]["transport_evidence"],
        "p2p_disabled_transport": nccl["cases"]["p2p-disabled"]["transport_evidence"],
    },
    "qwen_flash_real_project_concurrency_8": {
        "tp4_shm": tp4_shm,
        "tp4_p2p": tp4_p2p,
        "tp4_p2p_gain_percent": (tp4_p2p["mean"] / tp4_shm["mean"] - 1) * 100,
        "tep4_shm": tep4_shm,
        "tep4_p2p": tep4_p2p,
        "tep4_p2p_gain_percent": (tep4_p2p["mean"] / tep4_shm["mean"] - 1) * 100,
        "original_max_num_seqs_1": seq1["aggregate_completion_tokens_s"],
        "final_vs_original_speedup_x": tep4_p2p["mean"] / seq1["aggregate_completion_tokens_s"],
    },
    "qwen27_bf16_tp2_mtp1": {
        "p2p_tokens_per_second": 73.62307082292386,
        "shm_tokens_per_second": 75.11593863729937,
        "p2p_change_percent": (73.62307082292386 / 75.11593863729937 - 1) * 100,
        "output_hash_equal": True,
        "note": "Batch-1 hidden-state collectives are around 10KiB; P2P bandwidth does not offset small-message synchronization cost.",
    },
    "update_lifecycle": {
        "dkms_package": "nvidia-open-p2p-dkms 595.99.02-3ubuntu24.04p2p",
        "dkms_version": "nvidia-open-p2p/595.99.02-p2p3",
        "autoinstall": True,
        "exact_userspace_pairing": True,
        "apt_mark_holds": 0,
        "stock_rescue_entry": "nvidia-stock-rescue-7.0.0-38",
        "validated_default_entry": "nvidia-p2p-48g-test-7.0.0-38",
    },
}

(DATA / "summary.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode())
print(json.dumps(summary, indent=2))
