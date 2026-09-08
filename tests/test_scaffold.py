import yaml
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_directories_exist() -> None:
    expected = [
        "papers",
        "papers/reading_notes",
        "papers/reproduction",
        "src/model",
        "src/data",
        "src/training",
        "src/evaluation",
        "configs",
        "experiments",
        "reports",
        "tests",
        "docs",
    ]

    missing = [path for path in expected if not (ROOT / path).is_dir()]
    assert not missing, f"Missing scaffold directories: {missing}"


def test_baseline_config_is_valid() -> None:
    config_path = ROOT / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["seed"] == 42
    assert config["model"]["context_length"] == 256
    assert config["model"]["n_layers"] == 6
    assert config["model"]["n_heads"] == 4
