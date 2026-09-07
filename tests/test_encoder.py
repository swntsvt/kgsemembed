"""Tests for the model-aware EmbeddingEncoder.

All tests use mocked ``SentenceTransformer`` instances; no real models are
loaded. The mock returns deterministic vectors so encoding behaviour, prefix
handling, batch encoding, cosine similarity, and data types can be inspected.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from kgsemembed.embeddings import EmbeddingEncoder, get_model_config
from kgsemembed.embeddings import encoder as encoder_module

ALL_KEYS = ["M1", "M2", "M3", "M4", "M5"]
PREFIX_FREE_KEYS = ["M1", "M2", "M3", "M4"]
M5_QUERY_PREFIX = get_model_config("M5").query_prefix
M5_INSTRUCTION = "Instruct: Retrieve semantically similar text.\nQuery: "


def _make_encoder(model_key: str, dim: int = 4) -> tuple[EmbeddingEncoder, MagicMock]:
    """Build an encoder wrapping a mock model that returns unit vectors."""
    model = MagicMock(name="SentenceTransformer")
    model.encode.return_value = np.ones(dim, dtype=np.float64)
    return EmbeddingEncoder(model_key, model), model


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ALL_KEYS)
def test_construction_retrieves_registry_config(model_key: str) -> None:
    encoder, model = _make_encoder(model_key)
    assert encoder.config is get_model_config(model_key)
    assert encoder.model is model


def test_construction_does_not_load_a_model() -> None:
    model = MagicMock(name="SentenceTransformer")
    EmbeddingEncoder("M1", model)
    model.encode.assert_not_called()


def test_unknown_model_key_raises() -> None:
    with pytest.raises(KeyError):
        EmbeddingEncoder("M6", MagicMock())


# ---------------------------------------------------------------------------
# Prefix handling
# ---------------------------------------------------------------------------
def test_m5_source_applies_instruction_prefix() -> None:
    encoder, _ = _make_encoder("M5")
    result = encoder._apply_prefix("hello", "source")
    assert result == M5_INSTRUCTION + "hello"
    assert result.startswith("Instruct:")
    assert result.endswith("hello")


def test_m5_query_prefix_has_single_interpolation_placeholder() -> None:
    assert get_model_config("M5").query_prefix.count("{}") == 1


def test_m5_candidate_receives_no_prefix() -> None:
    encoder, _ = _make_encoder("M5")
    assert encoder._apply_prefix("hello", "candidate") == "hello"


@pytest.mark.parametrize("model_key", PREFIX_FREE_KEYS)
def test_prefix_free_models_leave_source_unchanged(model_key: str) -> None:
    encoder, _ = _make_encoder(model_key)
    assert encoder._apply_prefix("hello", "source") == "hello"


@pytest.mark.parametrize("model_key", ALL_KEYS)
def test_all_models_leave_candidate_unchanged(model_key: str) -> None:
    encoder, _ = _make_encoder(model_key)
    assert encoder._apply_prefix("hello", "candidate") == "hello"


def test_prefix_behaviour_matches_registry_config() -> None:
    for model_key in ALL_KEYS:
        encoder, _ = _make_encoder(model_key)
        config = get_model_config(model_key)
        expected_source = (
            "text" if config.query_prefix is None
            else config.query_prefix.format("text")
        )
        expected_candidate = (
            "text" if config.doc_prefix is None
            else config.doc_prefix.format("text")
        )
        assert encoder._apply_prefix("text", "source") == expected_source
        assert encoder._apply_prefix("text", "candidate") == expected_candidate


def test_invalid_role_raises_descriptive_value_error() -> None:
    encoder, _ = _make_encoder("M1")
    with pytest.raises(ValueError) as exc_info:
        encoder._apply_prefix("hello", "invalid")
    message = str(exc_info.value)
    assert "invalid" in message
    assert "source" in message
    assert "candidate" in message


# ---------------------------------------------------------------------------
# Source encoding
# ---------------------------------------------------------------------------
def test_encode_source_applies_prefix_and_options() -> None:
    encoder, model = _make_encoder("M5")
    result = encoder.encode_source("hello")
    args, kwargs = model.encode.call_args
    assert args[0] == M5_QUERY_PREFIX.format("hello")
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["show_progress_bar"] is False
    assert result.dtype == np.float32
    assert result.ndim == 1


def test_encode_source_is_prefix_free_for_symmetric_models() -> None:
    encoder, model = _make_encoder("M1")
    encoder.encode_source("hello")
    assert model.encode.call_args.args[0] == "hello"


# ---------------------------------------------------------------------------
# Candidate encoding
# ---------------------------------------------------------------------------
def test_encode_candidate_applies_no_prefix_and_options() -> None:
    encoder, model = _make_encoder("M5")
    result = encoder.encode_candidate("hello")
    args, kwargs = model.encode.call_args
    assert args[0] == "hello"
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["show_progress_bar"] is False
    assert result.dtype == np.float32
    assert result.ndim == 1


def test_source_and_candidate_are_separate_operations() -> None:
    encoder, model = _make_encoder("M5")
    encoder.encode_source("a")
    encoder.encode_candidate("a")
    first_text = model.encode.call_args_list[0].args[0]
    second_text = model.encode.call_args_list[1].args[0]
    assert first_text == M5_QUERY_PREFIX.format("a")
    assert second_text == "a"


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------
def test_encode_batch_applies_prefix_to_every_element() -> None:
    encoder, model = _make_encoder("M5", dim=3)
    model.encode.return_value = np.ones((2, 3), dtype=np.float64)
    encoder.encode_batch(["alpha", "beta"], role="source", show_progress=False)
    prefixed = model.encode.call_args.args[0]
    assert prefixed == [M5_INSTRUCTION + "alpha", M5_INSTRUCTION + "beta"]
    assert prefixed[0].endswith("alpha") and prefixed[1].endswith("beta")


def test_encode_batch_uses_configured_options() -> None:
    encoder, model = _make_encoder("M2", dim=3)
    model.encode.return_value = np.ones((2, 3), dtype=np.float64)
    result = encoder.encode_batch(["a", "b"], show_progress=False)
    kwargs = model.encode.call_args.kwargs
    assert kwargs["batch_size"] == get_model_config("M2").batch_size
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["show_progress_bar"] is False
    assert result.dtype == np.float32
    assert result.shape == (2, 3)


def test_encode_batch_defaults_to_candidate_role() -> None:
    encoder, model = _make_encoder("M5", dim=3)
    model.encode.return_value = np.ones((2, 3), dtype=np.float64)
    encoder.encode_batch(["a", "b"], show_progress=False)
    assert model.encode.call_args.args[0] == ["a", "b"]


def test_encode_batch_forwards_show_progress_argument() -> None:
    encoder, model = _make_encoder("M1", dim=3)
    model.encode.return_value = np.ones((1, 3), dtype=np.float64)
    encoder.encode_batch(["a"], show_progress=True)
    assert model.encode.call_args.kwargs["show_progress_bar"] is True


def test_encode_empty_batch_does_not_raise() -> None:
    encoder, model = _make_encoder("M1", dim=3)
    model.encode.return_value = np.empty((0, 3), dtype=np.float64)
    result = encoder.encode_batch([], show_progress=False)
    assert model.encode.call_args.args[0] == []
    assert result.dtype == np.float32
    assert result.shape == (0, 3)


# ---------------------------------------------------------------------------
# Dynamic batch sizing for long texts
# ---------------------------------------------------------------------------
def _batch_size_used(model_key: str, texts: list[str]) -> int:
    """Return the batch size ``model.encode`` received for ``texts``."""
    encoder, model = _make_encoder(model_key, dim=3)
    model.encode.return_value = np.ones((len(texts), 3), dtype=np.float64)
    encoder.encode_batch(texts, show_progress=False)
    return model.encode.call_args.kwargs["batch_size"]


def test_short_texts_keep_the_configured_batch_size() -> None:
    assert _batch_size_used("M3", ["short text", "another"]) == 16


def test_texts_below_threshold_keep_the_configured_batch_size() -> None:
    assert _batch_size_used("M3", ["c" * 2047]) == 16


def test_long_texts_reduce_the_batch_size() -> None:
    assert _batch_size_used("M3", ["c" * 4096]) == 8


def test_very_long_texts_reduce_the_batch_size_further() -> None:
    assert _batch_size_used("M3", ["c" * 8192]) == 4


def test_batch_size_never_falls_below_one() -> None:
    assert _batch_size_used("M3", ["c" * 1_000_000]) == 1


def test_longest_text_drives_the_reduction() -> None:
    assert _batch_size_used("M3", ["short", "c" * 4096, "short"]) == 8


def test_empty_batch_keeps_the_configured_batch_size() -> None:
    assert _batch_size_used("M3", []) == 16


@pytest.mark.parametrize("model_key", ["M1", "M2", "M4", "M5"])
def test_short_context_models_never_reduce_the_batch_size(model_key: str) -> None:
    """A model truncating at or below the threshold cannot exceed the budget."""
    assert get_model_config(model_key).max_tokens <= 512
    expected = get_model_config(model_key).batch_size
    assert _batch_size_used(model_key, ["c" * 1_000_000]) == expected


def test_estimate_is_capped_at_the_model_token_limit() -> None:
    beyond_limit = encoder_module._estimate_batch_tokens(["c" * 1_000_000], 8192)
    assert beyond_limit == 8192


def test_reduction_saturates_at_the_token_limit() -> None:
    assert _batch_size_used("M3", ["c" * 32_768]) == _batch_size_used(
        "M3", ["c" * 1_000_000]
    )


def test_prefix_length_counts_towards_the_estimate() -> None:
    encoder, _ = _make_encoder("M5")
    text = "c" * 4090
    raw = encoder_module._estimate_batch_tokens([text], 8192)
    prefixed = encoder_module._estimate_batch_tokens(
        [encoder._apply_prefix(text, "source")], 8192
    )
    assert prefixed > raw


def _debug_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the encoder module logger with a mock and return it."""
    logger = MagicMock(name="logger")
    monkeypatch.setattr(encoder_module, "_LOGGER", logger)
    return logger


def test_reduction_emits_a_debug_message(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, _ = _make_encoder("M3")
    logger = _debug_logger(monkeypatch)
    encoder._effective_batch_size(["c" * 4096])
    template, *arguments = logger.debug.call_args.args
    assert template % tuple(arguments) == (
        "Reducing batch size from 16 to 8 for 1 texts with estimated avg 1024 tokens"
    )


def test_no_debug_message_without_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, _ = _make_encoder("M3")
    logger = _debug_logger(monkeypatch)
    encoder._effective_batch_size(["short text"])
    logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def test_cosine_similarity_shape_and_dtype() -> None:
    encoder, _ = _make_encoder("M1")
    source = np.random.rand(3, 8).astype(np.float32)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    candidate = np.random.rand(5, 8).astype(np.float32)
    candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
    matrix = encoder.cosine_similarity_matrix(source, candidate)
    assert matrix.shape == (3, 5)
    assert matrix.dtype == np.float32


def test_cosine_similarity_numerical_correctness() -> None:
    encoder, _ = _make_encoder("M1")
    source = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrix = encoder.cosine_similarity_matrix(source, candidate)
    expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(matrix, expected)


def test_cosine_similarity_does_not_renormalise() -> None:
    encoder, _ = _make_encoder("M1")
    source = np.array([[2.0, 0.0]], dtype=np.float32)
    candidate = np.array([[3.0, 0.0]], dtype=np.float32)
    matrix = encoder.cosine_similarity_matrix(source, candidate)
    np.testing.assert_allclose(matrix, np.array([[6.0]], dtype=np.float32))


# ---------------------------------------------------------------------------
# Edge cases and regression
# ---------------------------------------------------------------------------
def test_empty_string_encoding_does_not_raise() -> None:
    encoder, model = _make_encoder("M5")
    encoder.encode_source("")
    assert model.encode.call_args.args[0] == M5_QUERY_PREFIX.format("")


def test_repeated_encoding_calls_are_independent() -> None:
    encoder, model = _make_encoder("M1")
    encoder.encode_source("a")
    encoder.encode_candidate("b")
    encoder.encode_source("c")
    assert model.encode.call_count == 3


@pytest.mark.parametrize("model_key", ALL_KEYS)
def test_all_models_encode_without_error(model_key: str) -> None:
    encoder, _ = _make_encoder(model_key)
    result = encoder.encode_candidate("entity")
    assert result.dtype == np.float32
    assert result.ndim == 1


def test_encoder_is_exported_from_package() -> None:
    import kgsemembed.embeddings as embeddings

    assert "EmbeddingEncoder" in embeddings.__all__
    assert embeddings.EmbeddingEncoder is EmbeddingEncoder
