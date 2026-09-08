from pathlib import Path


def test_sampling_script_exposes_reproducible_generation_arguments() -> None:
    script_path = Path("scripts/sample_mini_gpt.py")

    assert script_path.exists()
    help_text = script_path.read_text(encoding="utf-8")

    assert "--config" in help_text
    assert "--checkpoint" in help_text
    assert "--prompt" in help_text
    assert "--max-new-tokens" in help_text
    assert "--temperature" in help_text
    assert "--top-k" in help_text
