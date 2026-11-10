"""Chạy song song 2 seeds trên 2 GPU (Kaggle GPU T4 x2, 15GB VRAM mỗi GPU).

Kiến trúc chống OOM (Out Of Memory):
  1. Cô lập GPU:
     - Tiến trình 1 chạy trên GPU 0 bằng `CUDA_VISIBLE_DEVICES="0"`
     - Tiến trình 2 chạy trên GPU 1 bằng `CUDA_VISIBLE_DEVICES="1"`
     Mỗi tiến trình chỉ nhận diện duy nhất 1 GPU, không chia sẻ VRAM.
  2. Kiểm soát CPU RAM (Giới hạn 30GB của Kaggle):
     - Mỗi tiến trình đặt `num_workers = 2` (tổng 4 workers, RAM chiếm < 6GB).
     - `batch_size = 64` (MobileNetV3 chỉ chiếm ~2.5GB / 15GB VRAM).
     - Garbage collection & giải phóng cache giữa các stage.
  3. Thư mục độc lập:
     - Seed 1 ghi vào `outputs_seed_42/`
     - Seed 2 ghi vào `outputs_seed_44/`

Chạy:
  python -m pipelines.run_dual_gpu_cbm_experiment --config params.yaml --seeds 42 44
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
    base_config_path: str,
    output_root: str,
    skip_baseline: bool,
    results: dict,
) -> None:
    seed_output_dir = os.path.join(output_root, f"outputs_seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    # 1. Tạo file config riêng cho seed này
    with open(base_config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    params["seed"] = seed
    params["output_dir"] = seed_output_dir
    params["num_workers"] = 2  # Chống RAM OOM trên Kaggle (2 workers per process)
    params["batch_size"] = min(params.get("batch_size", 64), 64)  # An toàn cho 15GB VRAM

    seed_config_path = os.path.join(seed_output_dir, f"params_seed_{seed}.yaml")
    with open(seed_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    # 2. Thiết lập biến môi trường cô lập GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["DISABLE_DVCLIVE"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["DVCLIVE_LOGLEVEL"] = "ERROR"
    env["DVC_NO_ANALYTICS"] = "1"

    cmd = [
        sys.executable, "-m", "pipelines.run_cbm_neuro_symbolic_pipeline",
        "--config", seed_config_path,
    ]
    if skip_baseline:
        cmd.append("--skip-baseline")

    log_path = os.path.join(seed_output_dir, f"execution_gpu_{gpu_id}_seed_{seed}.log")
    print(f"[GPU {gpu_id}] Khởi động huấn luyện Seed {seed} -> Ghi log tại {log_path}", flush=True)

    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if any(key in line for key in ["STAGE", "Epoch", "Acc", "Loss", "HOÀN THÀNH", "Error"]):
                print(f"[GPU {gpu_id} | Seed {seed}] {line.strip()}", flush=True)

        proc.wait()

    elapsed = (time.time() - start_time) / 60
    success = proc.returncode == 0
    results[seed] = {
        "gpu": gpu_id,
        "success": success,
        "elapsed_min": elapsed,
        "output_dir": seed_output_dir,
    }
    if success:
        print(f"[GPU {gpu_id} | Seed {seed}] HOÀN THÀNH XUẤT SẮC sau {elapsed:.2f} phút!", flush=True)
    else:
        print(f"[GPU {gpu_id} | Seed {seed}] GẶP LỖI (Exit code: {proc.returncode}). Xem log: {log_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Chạy song song 2 seeds CBM trên 2 GPU.")
    parser.add_argument("--config", default="params.yaml", help="Path to base params.yaml")
    parser.add_argument("--seeds", nargs=2, type=int, default=[42, 44], help="2 seeds cần chạy (vd: 42 44)")
    parser.add_argument("--output-root", default=".", help="Thư mục gốc chứa outputs")
    parser.add_argument("--skip-baseline", action="store_true", help="Bỏ qua Stage 1 nếu đã có baseline")

    args = parser.parse_args()
    seed_a, seed_b = args.seeds[0], args.seeds[1]

    # Kiểm tra GPU có đủ 2 thiết bị
    import torch
    n_gpus = torch.cuda.device_count()
    print("=" * 70)
    print(f"DUAL-GPU CBM RUNNER: Khởi chạy song song 2 Seeds ({seed_a} và {seed_b})")
    print(f"Số GPU phát hiện được: {n_gpus}")
    if n_gpus < 2:
        print("CẢNH BÁO: Hệ thống chỉ có 1 GPU. Khuyến nghị chọn Kaggle Accelerator: GPU T4 x2.")
        print("Tiến trình 2 sẽ chạy cùng trên GPU 0 (có thể tăng tải VRAM).")
        gpu_b = 0
    else:
        print(f"  GPU 0: {torch.cuda.get_device_name(0)} (gán cho Seed {seed_a})")
        print(f"  GPU 1: {torch.cuda.get_device_name(1)} (gán cho Seed {seed_b})")
        gpu_b = 1
    print("=" * 70)

    total_start = time.time()
    results = {}

    t0 = threading.Thread(
        target=_run_worker,
        args=(0, seed_a, args.config, args.output_root, args.skip_baseline, results),
        name=f"Worker-GPU0-Seed{seed_a}",
    )
    t1 = threading.Thread(
        target=_run_worker,
        args=(gpu_b, seed_b, args.config, args.output_root, args.skip_baseline, results),
        name=f"Worker-GPU1-Seed{seed_b}",
    )

    t0.start()
    t1.start()

    t0.join()
    t1.join()

    total_min = (time.time() - total_start) / 60
    print("\n" + "=" * 70)
    print(f"TỔNG KẾT DUAL-GPU RUN SAU {total_min:.2f} PHÚT:")
    for s, info in results.items():
        status = "THÀNH CÔNG" if info["success"] else "THẤT BẠI"
        print(f"  - Seed {s} [GPU {info['gpu']}]: {status} trong {info['elapsed_min']:.2f} phút -> {info['output_dir']}")
    print("=" * 70)

    # Tổng hợp báo cáo phân loại chi tiết của tất cả các models và seeds
    try:
        import pandas as pd
        all_seed_dfs = []
        for s in [seed_a, seed_b]:
            sum_csv = Path(args.output_root) / f"outputs_seed_{s}" / "seed_classification_summary.csv"
            if sum_csv.exists():
                all_seed_dfs.append(pd.read_csv(sum_csv))
        if all_seed_dfs:
            combined_df = pd.concat(all_seed_dfs, ignore_index=True)
            combined_csv = Path(args.output_root) / "dual_gpu_all_seeds_classification_summary.csv"
            combined_xlsx = Path(args.output_root) / "dual_gpu_all_seeds_classification_summary.xlsx"
            combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
            try:
                combined_df.to_excel(combined_xlsx, index=False)
            except Exception:
                pass
            print(f"\n>>> ĐÃ TỔNG HỢP TOÀN BỘ CLASSIFICATION REPORT CHO MỌI MÔ HÌNH VÀ SEED:")
            print(f"    CSV:  {combined_csv}")
            print(f"    Excel: {combined_xlsx}")
    except Exception as e:
        print(f"Lưu ý: Không thể xuất file tổng hợp: {e}")


if __name__ == "__main__":
    main()
