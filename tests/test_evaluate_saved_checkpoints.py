from pathlib import Path

from pipelines.evaluate_saved_checkpoints import _method_metrics_path


def test_method_metrics_are_written_below_current_output() -> None:
    output = Path("current_run")

    result = _method_metrics_path(output, "cnn_baseline")

    assert result == (
        output / "exact_test_metrics_by_method" / "cnn_baseline.json"
    )
