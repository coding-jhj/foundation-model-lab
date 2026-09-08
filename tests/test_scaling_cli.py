from pathlib import Path


def test_scaling_sweep_declares_three_model_variants() -> None:
    script_path = Path("scripts/run_scaling_sweep.py")

    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")

    assert "wikitext2_scaling_small.yaml" in script_text
    assert "wikitext2_scaling_base.yaml" in script_text
    assert "wikitext2_scaling_large.yaml" in script_text
