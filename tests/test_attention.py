"""Tests for restoring bidirectional attention on decoder embedding models."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from kgsemembed.embeddings.attention import enable_bidirectional_attention


def _tiny_qwen2() -> Qwen2Model:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    return Qwen2Model(config).eval()


def _wrapped(decoder: Qwen2Model) -> list:
    return [SimpleNamespace(auto_model=decoder)]


def _first_token_state(decoder: Qwen2Model, ids: list, mask: list) -> torch.Tensor:
    with torch.no_grad():
        output = decoder.forward(
            input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask])
        )
    return output.last_hidden_state[0, 0]


def _first_token_moves_with_last_token(decoder: Qwen2Model) -> bool:
    before = _first_token_state(decoder, [5, 6, 7, 8], [1, 1, 1, 1])
    after = _first_token_state(decoder, [5, 6, 7, 9], [1, 1, 1, 1])
    return not torch.allclose(before, after)


def test_stock_decoder_is_causal() -> None:
    assert not _first_token_moves_with_last_token(_tiny_qwen2())


def test_enabled_decoder_attends_to_later_tokens() -> None:
    decoder = _tiny_qwen2()
    enable_bidirectional_attention(_wrapped(decoder))
    assert _first_token_moves_with_last_token(decoder)


def test_padding_is_still_masked() -> None:
    decoder = _tiny_qwen2()
    enable_bidirectional_attention(_wrapped(decoder))
    unpadded = _first_token_state(decoder, [5, 6, 7], [1, 1, 1])
    padded = _first_token_state(decoder, [5, 6, 7, 0, 0], [1, 1, 1, 0, 0])
    np.testing.assert_allclose(unpadded.numpy(), padded.numpy(), atol=1e-5)


def test_every_attention_layer_is_marked_non_causal() -> None:
    decoder = _tiny_qwen2()
    enable_bidirectional_attention(_wrapped(decoder))
    assert all(not layer.self_attn.is_causal for layer in decoder.layers)


def test_forward_keeps_its_signature_for_sentence_transformers() -> None:
    decoder = _tiny_qwen2()
    enable_bidirectional_attention(_wrapped(decoder))
    assert "attention_mask" in inspect.signature(decoder.forward).parameters


def test_non_decoder_backbone_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a decoder backbone"):
        enable_bidirectional_attention(_wrapped(torch.nn.Linear(2, 2)))


def test_sliding_window_backbone_is_rejected() -> None:
    decoder = _tiny_qwen2()
    decoder.has_sliding_layers = True
    with pytest.raises(TypeError, match="Sliding-window"):
        enable_bidirectional_attention(_wrapped(decoder))


_CARD_QUERIES = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
_CARD_DOCS = [
    "There are many effective ways to reduce stress. Some common techniques "
    "include deep breathing, meditation, and physical activity. Engaging in "
    "hobbies, spending time in nature, and connecting with loved ones can also "
    "help alleviate stress. Additionally, setting boundaries, practicing "
    "self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential "
    "health benefits. It contains antioxidants that may help protect the body "
    "against damage caused by free radicals. Regular consumption of green tea "
    "has been associated with improved heart health, enhanced cognitive "
    "function, and a reduced risk of certain types of cancer. The polyphenols "
    "in green tea may also have anti-inflammatory and weight loss properties.",
]
_CARD_SIMILARITIES = np.array([[0.8179, 0.2958], [0.3194, 0.7854]])


@pytest.mark.slow
def test_m5_reproduces_the_model_card_similarities() -> None:
    from kgsemembed.embeddings.models import load_sentence_transformer

    model, _ = load_sentence_transformer("M5")
    queries = model.encode(_CARD_QUERIES, prompt_name="s2p_query", normalize_embeddings=True)
    docs = model.encode(_CARD_DOCS, normalize_embeddings=True)
    np.testing.assert_allclose(queries @ docs.T, _CARD_SIMILARITIES, atol=1e-3)
