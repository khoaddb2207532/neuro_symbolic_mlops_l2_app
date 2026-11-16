"""Chạy tiếp song song 2 seeds (mỗi seed một GPU) từ kết quả lần chạy trước.

Cấu trúc giống ``run_dual_gpu_cbm_experiment`` để notebook Kaggle dùng cùng cách:
  - Seed A ghi vào ``<output_root>/outputs_seed_A/`` trên GPU 0, seed B trên GPU 1.
  - Mỗi tiến trình con chạy ``pipelines.run_concept_rules_pipeline`` với
    ``CUDA_VISIBLE_DEVICES`` riêng, log ghi vào ``execution_concept_rules_seed_<s>.log``.

Chạy:
  python -m pipelines.run_concept_rules_dual_gpu --config params.yaml --seeds 42 44
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _run_worker(
    gpu_id: int,
    seed: int,
    base_config: str,
    output_root: str,
    modes: list[str],
    results: dict[int, dict[str, object]],
) -> None:
    seed_dir = Path(output_root) / f"outputs_seed_{seed}"
    if not seed_dir.is_dir():
        results[seed] = {"gpu": gpu_id, "success": False, "elapsed_min": 0.0,
                         "output_dir": str(seed_dir), "error": "thiếu thư mục kết quả lần chạy trước"}
        return

    with open(base_config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["seed"] = seed
    params["output_dir"] = str(seed_dir)
    params["num_workers"] = 2
    params["batch_size"] = min(int(params.get("batch_size", 64)), 64)
    seed_config = seed_dir / f"params_concept_rules_seed_{seed}.yaml"
    with open(seed_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    cmd = [sys.executable, "-m", "pipelines.run_concept_rules_pipeline",
           "--config", str(seed_config), "--modes", *modes]
    log_path = seed_dir / f"execution_concept_rules_gpu_{gpu_id}_seed_{seed}.log"

    print(f"[GPU {gpu_id}] Seed {seed} -> log {log_path}", flush=True)
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if any(k in line for k in ("STAGE", ">>>", "Epoch", "Acc", "Error", "HOÀN", "Chọn")):
                print(f"[GPU {gpu_id} | Seed {seed}] {line.strip()}", flush=True)
        proc.wait()

    elapsed = (time.time() - start) / 60
    results[seed] = {"gpu": gpu_id, "success": proc.returncode == 0,
                     "elapsed_min": elapsed, "output_dir": str(seed_dir)}
    status = "HOÀN THÀNH" if proc.returncode == 0 else f"LỖI (exit {proc.returncode})"
    print(f"[GPU {gpu_id} | Seed {seed}] {status} sau {elapsed:.2f} phút", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--seeds", nargs=2, type=int, default=[42, 44])
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--modes", nargs="+", default=["penalty", "head", "both"])
    args = parser.parse_args()
    seed_a, seed_b = args.seeds

    import torch

    n_gpus = torch.cuda.device_count()
    gpu_b = 1 if n_gpus >= 2 else 0
    if n_gpus < 2:
        print("CẢNH BÁO: dưới 2 GPU, seed B sẽ chạy chung GPU 0.", flush=True)

    results: dict[int, dict[str, object]] = {}
    threads = [
        threading.Thread(target=_run_worker, args=(0, seed_a, args.config, args.output_root, args.modes, results)),
        threading.Thread(target=_run_worker, args=(gpu_b, seed_b, args.config, args.output_root, args.modes, results)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("=" * 70)
    for seed in (seed_a, seed_b):
        info = results.get(seed, {"success": False})
        status = "THÀNH CÔNG" if info.get("success") else "THẤT BẠI"
        extra = f" ({info['error']})" if info.get("error") else ""
        print(f"  Seed {seed}: {status}{extra}")
    print("=" * 70)

    failed = [s for s in (seed_a, seed_b) if not results.get(s, {}).get("success")]
    if failed:
        print(f"LỖI: seed {failed} không hoàn thành.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
