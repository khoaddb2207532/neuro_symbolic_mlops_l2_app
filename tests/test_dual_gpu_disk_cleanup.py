from pathlib import Path

from pipelines.run_dual_gpu_elite_seed_experiment import (
    _cleanup_completed_run,
    _remove_legacy_seed_archive,
    _remove_stale_tmp_files,
    _worker,
)


def test_completed_run_keeps_best_and_removes_resume_files(tmp_path: Path) -> None:
    run_root = tmp_path / "dual_elite_seed_42"
    stage_dir = run_root / "tb" / "05b_rules_model_bayesian_elite"
    stage_dir.mkdir(parents=True)
    best = stage_dir / "rule_regularized_best.pth"
    resume = stage_dir / "training_last.pth"
    final = stage_dir / "final_model_weights.pth"
    interrupted = stage_dir / "training_last.pth.tmp"
    best.write_bytes(b"best")
    resume.write_bytes(b"resume")
    final.write_bytes(b"final")
    interrupted.write_bytes(b"partial")

    result = _cleanup_completed_run(run_root)

    assert best.read_bytes() == b"best"
    assert not resume.exists()
    assert not final.exists()
    assert not interrupted.exists()
    assert result["removed_bytes"] == len(b"resume") + len(b"final") + len(b"partial")
    assert result["kept_checkpoint"] == "rule_regularized_best.pth"


def test_startup_cleanup_removes_tmp_and_only_matching_legacy_archive(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "dual_elite_seed_44"
    run_root.mkdir()
    stale = run_root / "training_last.pth.tmp"
    stale.write_bytes(b"partial")
    matching = tmp_path / "seed_44_dual_elite_artifacts.tar.gz"
    other_seed = tmp_path / "seed_42_dual_elite_artifacts.tar.gz"
    matching.write_bytes(b"old archive")
    other_seed.write_bytes(b"keep")

    tmp_files, tmp_bytes = _remove_stale_tmp_files(run_root)
    archives, archive_bytes = _remove_legacy_seed_archive(run_root, 44)

    assert tmp_files == ["training_last.pth.tmp"]
    assert tmp_bytes == len(b"partial")
    assert archives == [str(matching)]
    assert archive_bytes == len(b"old archive")
    assert not stale.exists()
    assert not matching.exists()
    assert other_seed.read_bytes() == b"keep"


def test_tb_worker_trains_its_own_fixed_prior(tmp_path: Path, monkeypatch) -> None:
    branch_dir = tmp_path / "tb"
    filtered = branch_dir / "04_filtered_rules"
    filtered.mkdir(parents=True)
    for filename in (
        "gflownet_best_elite.pth",
        "gflownet_best_diverse.pth",
        "gflownet_rule_order.pkl",
        "selected_rules.pkl",
    ):
        (filtered / filename).write_bytes(b"ready")
    config_path = branch_dir / "params_tb.yaml"
    config_path.write_text("gflownet:\n  loss_type: tb\n", encoding="utf-8")
    called_modules = []

    def fake_run(command, **_kwargs):
        called_modules.append(command[2])

    monkeypatch.setattr(
        "pipelines.run_dual_gpu_elite_seed_experiment.subprocess.run",
        fake_run,
    )

    _worker(tmp_path, config_path, branch_dir)

    assert "pipelines.stage5_train_rule_regularized" in called_modules
    assert "pipelines.stage5_train_rule_bayesian_tb" in called_modules


def test_worker_skips_completed_fixed_prior(tmp_path: Path, monkeypatch) -> None:
    branch_dir = tmp_path / "db"
    filtered = branch_dir / "04_filtered_rules"
    filtered.mkdir(parents=True)
    for filename in (
        "gflownet_best_elite.pth",
        "gflownet_best_diverse.pth",
        "gflownet_rule_order.pkl",
        "selected_rules.pkl",
    ):
        (filtered / filename).write_bytes(b"ready")
    fixed_dir = branch_dir / "05_rules_model"
    fixed_dir.mkdir()
    (fixed_dir / "rule_regularized_best.pth").write_bytes(b"complete")
    config_path = branch_dir / "params_db.yaml"
    config_path.write_text("gflownet:\n  loss_type: db\n", encoding="utf-8")
    called_modules = []

    def fake_run(command, **_kwargs):
        called_modules.append(command[2])

    monkeypatch.setattr(
        "pipelines.run_dual_gpu_elite_seed_experiment.subprocess.run",
        fake_run,
    )

    _worker(tmp_path, config_path, branch_dir)

    assert "pipelines.stage5_train_rule_regularized" not in called_modules
    assert "pipelines.stage5_train_rule_bayesian" in called_modules
