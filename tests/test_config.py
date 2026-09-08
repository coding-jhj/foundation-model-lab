import pytest


def _config_module():
    try:
        from training.config import load_config
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected training.config to exist: {error}") from error

    return load_config


def test_load_config_reads_yaml_and_validates_required_sections(tmp_path) -> None:
    load_config = _config_module()
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
project:
  name: test-run
data:
  train_path: train.txt
  validation_path: validation.txt
model:
  vocab_size: 16
training:
  max_steps: 2
evaluation:
  eval_batches: 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["project"]["name"] == "test-run"
    assert config["training"]["max_steps"] == 2


def test_load_config_rejects_missing_required_sections(tmp_path) -> None:
    load_config = _config_module()
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text("project: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required sections"):
        load_config(config_path)
