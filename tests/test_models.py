"""Tests for the embedding model registry and SentenceTransformer factory."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
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
from kgsemembed.utils.errors import ConfigurationError
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS

EXPECTED_KEYS = {"M1", "M2", "M2_uncapped", "M3", "M4", "M5"}

EXPECTED_MODEL_IDS = {
    "M1": "sentence-transformers/all-MiniLM-L6-v2",
    "M2": "BAAI/bge-large-en-v1.5",
    "M2_uncapped": "BAAI/bge-large-en-v1.5",
    "M3": "BAAI/bge-m3",
    "M4": "FremyCompany/BioLORD-2023",
    "M5": "dunzhang/stella_en_1.5B_v5",
}

EXPECTED_MAX_TOKENS = {
    "M1": 256,
    "M2": 512,
    "M2_uncapped": 512,
    "M3": 8192,
    "M4": 512,
    "M5": 512,
}


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
        assert isinstance(cfg.batch_size, int) and cfg.batch_size > 0


def test_model_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        MODEL_REGISTRY["M1"].batch_size = 1


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------
def test_get_model_config_returns_registry_entry() -> None:
    assert get_model_config("M1") is MODEL_REGISTRY["M1"]


def test_m2_ppas_budget() -> None:
    assert get_model_config("M2").ppas_budget == 420


def test_m3_ppas_budget_is_none() -> None:
    assert get_model_config("M3").ppas_budget is None


def test_m2_uncapped_ppas_budget_is_none() -> None:
    assert get_model_config("M2_uncapped").ppas_budget is None


def test_m2_uncapped_shares_the_m2_model_id() -> None:
    assert get_model_config("M2_uncapped").model_id == "BAAI/bge-large-en-v1.5"
    assert get_model_config("M2_uncapped").model_id == get_model_config("M2").model_id


def test_m2_uncapped_batch_size() -> None:
    assert get_model_config("M2_uncapped").batch_size == 64


def test_m2_uncapped_differs_from_m2_only_in_budget_and_batch_size() -> None:
    """The twin must not drift from M2 on any field that changes embeddings."""
    m2 = get_model_config("M2")
    uncapped = get_model_config("M2_uncapped")
    for field in ("model_id", "max_tokens", "query_prefix", "doc_prefix"):
        assert getattr(m2, field) == getattr(uncapped, field)
    assert m2.ppas_budget == 420
    assert uncapped.ppas_budget is None


def test_m2_uncapped_query_prefix_is_none() -> None:
    assert get_model_config("M2_uncapped").query_prefix is None


def test_m2_uncapped_documents_are_encoded_without_prefix() -> None:
    assert get_model_config("M2_uncapped").doc_prefix is None


def test_m5_query_prefix_is_instruction() -> None:
    assert get_model_config("M5").query_prefix.startswith("Instruct:")


def test_m2_query_prefix_is_none() -> None:
    assert get_model_config("M2").query_prefix is None


def test_m5_documents_are_encoded_without_prefix() -> None:
    assert get_model_config("M5").doc_prefix is None


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
    fake.return_value.__getitem__.return_value.auto_model.config = _position_config(8192)
    monkeypatch.setattr(models, "SentenceTransformer", fake)
    return fake


def test_model_constructed_only_when_factory_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_cpu_and_mock_transformer(monkeypatch)
    assert fake.call_count == 0
    model, _ = load_sentence_transformer("M1")
    fake.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        revision=MODEL_REGISTRY["M1"].revision,
    )
    assert model is fake.return_value


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
    fake.return_value.__getitem__.return_value.auto_model.config = _position_config(512)
    monkeypatch.setattr(models, "SentenceTransformer", fake)
    load_sentence_transformer("M2")
    fake.assert_called_once_with(
        "BAAI/bge-large-en-v1.5", device="mps", revision=MODEL_REGISTRY["M2"].revision
    )


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
def _position_config(capacity: int) -> SimpleNamespace:
    return SimpleNamespace(model_type="bert", max_position_embeddings=capacity, pad_token_id=0)


class _FakeModel:
    """Stand-in exposing only the attributes the revision lookup inspects."""

    def __init__(self, module=None, tokenizer=None):
        self._module = module
        if tokenizer is not None:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self._module is None:
            raise AttributeError("no transformer module")
        return self._module


def _model_with_commit_hash(commit_hash: str) -> _FakeModel:
    config = SimpleNamespace(_commit_hash=commit_hash, **vars(_position_config(512)))
    return _FakeModel(module=SimpleNamespace(auto_model=SimpleNamespace(config=config)))


def _load_with_model(monkeypatch: pytest.MonkeyPatch, model: _FakeModel):
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models, "SentenceTransformer", lambda *_a, **_k: model)
    return load_sentence_transformer("M1")


def test_loader_returns_model_and_metadata_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _model_with_commit_hash("abc123")
    model, model_info = _load_with_model(monkeypatch, fake_model)
    assert model is fake_model
    assert model_info == {
        "model_key": "M1",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "hf_revision": "abc123",
        "hf_revision_pinned": MODEL_REGISTRY["M1"].revision,
        "max_seq_length": 256,
    }


def test_metadata_holds_exactly_the_expected_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model_info = _load_with_model(monkeypatch, _model_with_commit_hash("abc123"))
    assert set(model_info) == {
        "model_key", "model_id", "device", "hf_revision", "hf_revision_pinned", "max_seq_length"
    }


def test_loader_sets_the_registered_input_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    model, model_info = _load_with_model(monkeypatch, _model_with_commit_hash("abc123"))
    assert model.max_seq_length == MODEL_REGISTRY["M1"].max_tokens == model_info["max_seq_length"]


def test_input_limit_beyond_the_position_capacity_is_refused() -> None:
    model = _FakeModel(module=SimpleNamespace(auto_model=SimpleNamespace(config=_position_config(256))))
    with pytest.raises(ConfigurationError, match="exceeds the model's position capacity 256"):
        models.set_input_limit(model, MODEL_REGISTRY["M2"])


@pytest.mark.parametrize("model_type, positions, pad, capacity", [
    ("bert", 512, 0, 512), ("mpnet", 514, 1, 512), ("xlm-roberta", 8194, 1, 8192), ("qwen2", 131072, None, 131072),
])
def test_position_capacity_accounts_for_offset_positions(model_type, positions, pad, capacity) -> None:
    config = SimpleNamespace(model_type=model_type, max_position_embeddings=positions, pad_token_id=pad)
    model = _FakeModel(module=SimpleNamespace(auto_model=SimpleNamespace(config=config)))
    assert models.position_capacity(model) == capacity


def test_metadata_records_selected_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        models, "SentenceTransformer", lambda *_a, **_k: _model_with_commit_hash("d4")
    )
    _, model_info = load_sentence_transformer("M2")
    assert model_info["device"] == "mps"
    assert model_info["model_key"] == "M2"
    assert model_info["model_id"] == "BAAI/bge-large-en-v1.5"


# The revision every stored C1-C20 and ablation result records (#141).
EXPECTED_REVISIONS = {
    "M1": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "M2": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    "M2_uncapped": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    "M3": "5617a9f61b028005a4858fdac845db406aefb181",
    "M4": "167aab527b238a50ca65224e6319215d2ff4fc9f",
    "M5": "7817065102fd9e1b031fe874e910c01f40b2f001",
}


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_every_model_pins_the_revision_of_the_stored_results(key: str) -> None:
    assert MODEL_REGISTRY[key].revision == EXPECTED_REVISIONS[key]


def test_loader_refuses_a_commit_other_than_the_pinned_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ConfigurationError, match="pinned"):
        _load_with_model(monkeypatch, _model_with_commit_hash("0" * 40))


def test_loader_accepts_the_pinned_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = MODEL_REGISTRY["M1"].revision
    _, model_info = _load_with_model(monkeypatch, _model_with_commit_hash(pinned))
    assert model_info["hf_revision"] == model_info["hf_revision_pinned"] == pinned


def test_require_device_accepts_mps() -> None:
    assert models.require_device({"device": "mps"}) == {
        "device": "mps", "device_override": False
    }


def test_require_device_refuses_another_device_by_default() -> None:
    with pytest.raises(ConfigurationError, match="allow_device"):
        models.require_device({"device": "cpu"})


def test_require_device_records_an_explicit_override() -> None:
    assert models.require_device({"device": "cpu"}, allowed_device="cpu") == {
        "device": "cpu", "device_override": True
    }


def test_require_device_override_names_one_device_only() -> None:
    with pytest.raises(ConfigurationError):
        models.require_device({"device": "cuda"}, allowed_device="cpu")


def test_revision_prefers_commit_hash() -> None:
    model = _model_with_commit_hash("9f1c2d")
    assert models._resolve_hf_revision(model) == "9f1c2d"


def test_revision_falls_back_to_tokenizer_path_on_attribute_error() -> None:
    model = _FakeModel(
        module=SimpleNamespace(),
        tokenizer=SimpleNamespace(name_or_path="/cache/all-MiniLM-L6-v2"),
    )
    assert models._resolve_hf_revision(model) == "/cache/all-MiniLM-L6-v2"


def test_revision_is_unknown_when_both_lookups_fail() -> None:
    assert models._resolve_hf_revision(_FakeModel()) == "unknown"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def test_package_exports() -> None:
    import kgsemembed.embeddings as embeddings

    assert set(embeddings.__all__) == {
        "EmbeddingEncoder",
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
    model, model_info = load_sentence_transformer("M1")
    assert isinstance(model, SentenceTransformer)
    assert set(model_info) == {
        "model_key", "model_id", "device", "hf_revision", "hf_revision_pinned", "max_seq_length"
    }
    assert model_info["hf_revision"] == MODEL_REGISTRY["M1"].revision


def test_factory_buckets_sequence_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cpu_and_mock_transformer(monkeypatch)
    model, _ = load_sentence_transformer("M1")
    assert model[0].processing_kwargs == {
        "text": {"pad_to_multiple_of": models.PAD_TO_MULTIPLE_OF}
    }


def test_only_m5_needs_bidirectional_attention() -> None:
    flagged = {key for key, config in MODEL_REGISTRY.items() if config.bidirectional_attention}
    assert flagged == {"M5"}


@pytest.mark.parametrize("key,expected", [("M1", 0), ("M5", 1)])
def test_factory_enables_bidirectional_attention_only_when_flagged(
    monkeypatch: pytest.MonkeyPatch, key: str, expected: int
) -> None:
    _patch_cpu_and_mock_transformer(monkeypatch)
    enable = MagicMock(name="enable_bidirectional_attention")
    monkeypatch.setattr(models, "enable_bidirectional_attention", enable)
    model, _ = load_sentence_transformer(key)
    assert enable.call_count == expected
    if expected:
        enable.assert_called_once_with(model)
