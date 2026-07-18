"""Tests for the embedding model registry and SentenceTransformer factory."""

from unittest.mock import MagicMock

import pytest
from sentence_transformers import SentenceTransformer

from kgsemembed.embeddings import (
    MODEL_REGISTRY,
    ModelConfig,
    get_model_config,
    load_sentence_transformer,
)
from kgsemembed.embeddings import models
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS

EXPECTED_KEYS = {"M1", "M2", "M3", "M4", "M5"}

EXPECTED_MODEL_IDS = {
    "M1": "sentence-transformers/all-MiniLM-L6-v2",
    "M2": "BAAI/bge-large-en-v1.5",
    "M3": "Alibaba-NLP/gte-large-en-v1.5",
    "M4": "FremyCompany/BioLORD-2023",
    "M5": "dunzhang/stella_en_1.5B_v5",
}

EXPECTED_MAX_TOKENS = {"M1": 256, "M2": 512, "M3": 8192, "M4": 512, "M5": 512}


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------
def test_registry_keys_match_supported_models() -> None:
    assert set(MODEL_REGISTRY) == EXPECTED_KEYS


def test_registry_values_are_model_configs() -> None:
    assert all(isinstance(cfg, ModelConfig) for cfg in MODEL_REGISTRY.values())


def test_registry_entry_model_key_matches_dict_key() -> None:
    assert all(key == cfg.model_key for key, cfg in MODEL_REGISTRY.items())


@pytest.mark.parametrize("key, model_id", EXPECTED_MODEL_IDS.items())
def test_registry_exposes_expected_model_ids(key: str, model_id: str) -> None:
    assert MODEL_REGISTRY[key].model_id == model_id


@pytest.mark.parametrize("key, max_tokens", EXPECTED_MAX_TOKENS.items())
def test_registry_exposes_expected_token_limits(
    key: str, max_tokens: int
) -> None:
    assert MODEL_REGISTRY[key].max_tokens == max_tokens


def test_registry_metadata_is_well_formed() -> None:
    for cfg in MODEL_REGISTRY.values():
        assert cfg.device_hint in {"mps", "cuda", "cpu"}
        assert isinstance(cfg.batch_size, int) and cfg.batch_size > 0


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------
def test_get_model_config_returns_registry_entry() -> None:
    assert get_model_config("M1") is MODEL_REGISTRY["M1"]


def test_m2_ppas_budget() -> None:
    assert get_model_config("M2").ppas_budget == 420


def test_m3_ppas_budget_is_none() -> None:
    assert get_model_config("M3").ppas_budget is None


def test_m5_query_prefix_is_instruction() -> None:
    assert get_model_config("M5").query_prefix.startswith("Instruct:")


def test_m2_query_prefix_is_none() -> None:
    assert get_model_config("M2").query_prefix is None


def test_unknown_model_key_raises_descriptive_keyerror() -> None:
    with pytest.raises(KeyError) as exc_info:
        get_model_config("M6")
    message = str(exc_info.value)
    assert "M6" in message
    for key in EXPECTED_KEYS:
        assert key in message


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mps_available, cuda_available, expected",
    [
        (True, True, "mps"),
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_device_selection_order(
    monkeypatch: pytest.MonkeyPatch,
    mps_available: bool,
    cuda_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(
        models.torch.backends.mps, "is_available", lambda: mps_available
    )
    monkeypatch.setattr(
        models.torch.cuda, "is_available", lambda: cuda_available
    )
    assert models._select_device() == expected


# ---------------------------------------------------------------------------
# Factory wiring and lazy loading (mocked)
# ---------------------------------------------------------------------------
def _patch_cpu_and_mock_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    fake = MagicMock(name="SentenceTransformer")
    monkeypatch.setattr(models, "SentenceTransformer", fake)
    return fake


def test_model_constructed_only_when_factory_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_cpu_and_mock_transformer(monkeypatch)
    assert fake.call_count == 0
    result = load_sentence_transformer("M1")
    fake.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    assert result is fake.return_value


def test_factory_reports_device_and_model_before_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_cpu_and_mock_transformer(monkeypatch)
    load_sentence_transformer("M1")
    out = capsys.readouterr().out
    assert "cpu" in out
    assert "sentence-transformers/all-MiniLM-L6-v2" in out


def test_repeated_factory_calls_do_not_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_cpu_and_mock_transformer(monkeypatch)
    load_sentence_transformer("M1")
    load_sentence_transformer("M1")
    assert fake.call_count == 2


def test_factory_selects_mps_over_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: True)
    fake = MagicMock(name="SentenceTransformer")
    monkeypatch.setattr(models, "SentenceTransformer", fake)
    load_sentence_transformer("M2")
    fake.assert_called_once_with("BAAI/bge-large-en-v1.5", device="mps")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def test_package_exports() -> None:
    import kgsemembed.embeddings as embeddings

    assert set(embeddings.__all__) == {
        "MODEL_REGISTRY",
        "ModelConfig",
        "get_model_config",
        "load_sentence_transformer",
    }
    for name in embeddings.__all__:
        assert hasattr(embeddings, name)


def test_registry_budgets_match_ppas_budgets() -> None:
    for key, cfg in MODEL_REGISTRY.items():
        assert cfg.ppas_budget == PPAS_BUDGETS[key]


# ---------------------------------------------------------------------------
# Slow integration
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_load_sentence_transformer_returns_instance() -> None:
    model = load_sentence_transformer("M1")
    assert isinstance(model, SentenceTransformer)
