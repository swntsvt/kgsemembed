from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_loads() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    assert cfg.model.name
    assert cfg.dataset.name
    assert cfg.verbalisation.strategy


def test_group_override_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["model=all-minilm-l6-v2", "verbalisation=v1"],
        )
    assert cfg.model.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.verbalisation.strategy == "v1"


def test_value_override_applies() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["model.batch_size=16"])
    assert cfg.model.batch_size == 16


def test_invalid_override_fails_fast() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        with pytest.raises(Exception):
            compose(config_name="config", overrides=["not_a_key=1"])
