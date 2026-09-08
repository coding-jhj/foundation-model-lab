from pathlib import Path


def test_training_script_exposes_a_config_argument() -> None:
    script_path = Path("scripts/train_mini_gpt.py")

    assert script_path.exists()
    help_text = script_path.read_text(encoding="utf-8")

    assert "--config" in help_text
    assert "baseline.yaml" in help_text


def test_training_script_reports_trainable_parameter_count() -> None:
    script_path = Path("scripts/train_mini_gpt.py")
    script_text = script_path.read_text(encoding="utf-8")

    assert "count_parameters" in script_text
    assert "trainable parameters" in script_text
