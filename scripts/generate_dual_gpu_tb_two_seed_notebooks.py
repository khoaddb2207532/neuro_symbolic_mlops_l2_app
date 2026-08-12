"""Generate GFlowNet-TB notebooks that run exactly two seeds on two GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.generate_dual_gpu_tb_multiseed_notebooks import (
        DEFAULT_BACKBONES,
        SUPPORTED_BACKBONES,
        generate,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<this-file>.py
    from generate_dual_gpu_tb_multiseed_notebooks import (  # type: ignore[no-redef]
        DEFAULT_BACKBONES,
        SUPPORTED_BACKBONES,
        generate,
    )


def validate_seeds(seeds: list[int] | tuple[int, ...]) -> tuple[int, int]:
    """Return the GPU-ordered seed pair, rejecting ambiguous workloads."""
    if len(seeds) != 2:
        raise ValueError("--seeds phải chứa đúng 2 seed: seed_GPU0 seed_GPU1.")
    first, second = map(int, seeds)
    if first == second:
        raise ValueError("Hai seed phải khác nhau.")
    return first, second


def generate_two_seed_notebooks(
    registry_path: Path,
    template_path: Path,
    output_dir: Path,
    *,
    git_ref: str,
    seeds: list[int] | tuple[int, ...],
    backbones: list[str] | tuple[str, ...] = DEFAULT_BACKBONES,
    datasets: list[str] | tuple[str, ...] | None = None,
    resume: bool = False,
) -> list[Path]:
    """Generate notebooks where GPU 0 owns seed[0] and GPU 1 owns seed[1]."""
    gpu0_seed, gpu1_seed = validate_seeds(seeds)
    prefix = f"dual_gpu_tb_two_seed_{gpu0_seed}_{gpu1_seed}"
    paths = generate(
        registry_path,
        template_path,
        output_dir,
        git_ref=git_ref,
        backbones=backbones,
        seeds=(gpu0_seed, gpu1_seed),
        datasets=datasets,
        filename_prefix=prefix,
        resume=resume,
    )

    # Make the fixed GPU ownership explicit inside every generated notebook.
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"][0]["source"] = [
            "# GFlowNet-TB: 2 seed song song trên 2 GPU\n",
            "\n",
            f"- GPU 0 chỉ chạy seed `{gpu0_seed}`.\n",
            f"- GPU 1 chỉ chạy seed `{gpu1_seed}`.\n",
            "- Mỗi GPU chạy tuần tự từng dataset, không tạo thêm worker đồng thời.\n",
            f"- Resume output cũ: `{'bật' if resume else 'tắt'}`.\n",
        ]
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sinh notebook chạy đúng hai seed, mỗi seed trên một GPU."
    )
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument("--template", default="dual-gpu-tb-multiseed-template.ipynb")
    parser.add_argument(
        "--output-dir", default="generated_dual_gpu_tb_two_seed_notebooks"
    )
    parser.add_argument("--git-ref", default="main")
    parser.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        required=True,
        metavar=("SEED_GPU0", "SEED_GPU1"),
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=sorted(SUPPORTED_BACKBONES),
        default=list(DEFAULT_BACKBONES),
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Cho phép restore output cũ; mặc định hai seed chạy mới hoàn toàn.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    generate_two_seed_notebooks(
        Path(arguments.registry),
        Path(arguments.template),
        Path(arguments.output_dir),
        git_ref=arguments.git_ref,
        seeds=arguments.seeds,
        backbones=arguments.backbones,
        datasets=arguments.datasets,
        resume=arguments.resume,
    )
