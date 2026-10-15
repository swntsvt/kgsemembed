"""Bidirectional attention for decoder-architecture embedding models.

Some embedding models are fine-tuned from decoder LLMs with the causal mask
removed, and publish custom modelling code that restores bidirectional
attention at load time (``auto_map`` in the model's ``config.json``).  Without
that code ``transformers`` loads the stock decoder, which applies a causal
mask: each token attends only to earlier tokens, so mean-pooled embeddings no
longer match the ones the model was trained to produce.

:func:`enable_bidirectional_attention` restores the trained behaviour on the
stock architecture without executing repository code: every attention layer
is marked non-causal and the padding mask is expanded into a bidirectional 4D
mask before the decoder would build its causal one.
"""

import functools
from typing import Any, Callable

import torch
from sentence_transformers import SentenceTransformer
from transformers.masking_utils import create_bidirectional_mask

_FULL_ATTENTION = "full_attention"


def _decoder(model: SentenceTransformer) -> torch.nn.Module:
    """Return the decoder backbone wrapped by the model's first module."""
    decoder = model[0].auto_model
    if not hasattr(decoder, "layers") or not hasattr(decoder, "embed_tokens"):
        raise TypeError(
            f"{type(decoder).__name__} is not a decoder backbone with "
            "attention layers; bidirectional attention cannot be enabled."
        )
    return decoder


def _bidirectional_mask(decoder: torch.nn.Module, kwargs: dict) -> dict:
    """Build the per-layer-type mask mapping the decoder accepts as prepared."""
    embeddings = kwargs.get("inputs_embeds")
    if embeddings is None:
        embeddings = decoder.embed_tokens(kwargs["input_ids"])
    mask = create_bidirectional_mask(
        config=decoder.config,
        inputs_embeds=embeddings,
        attention_mask=kwargs.get("attention_mask"),
    )
    return {_FULL_ATTENTION: mask}


def _with_bidirectional_mask(decoder: torch.nn.Module) -> Callable[..., Any]:
    """Wrap ``decoder.forward`` so it receives a bidirectional mask mapping."""
    original = decoder.forward

    @functools.wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        if not isinstance(kwargs.get("attention_mask"), dict):
            kwargs["attention_mask"] = _bidirectional_mask(decoder, kwargs)
        return original(*args, **kwargs)

    return forward


def enable_bidirectional_attention(model: SentenceTransformer) -> None:
    """
    Make a decoder-backbone embedding model attend in both directions.

    The forward method is replaced on the instance rather than hooked, because
    ``sentence-transformers`` calls ``forward`` directly and so bypasses module
    hooks.  Only full-attention layers are supported; a model with sliding
    window layers is rejected rather than silently given a causal mask.

    Parameters
    ----------
    model : SentenceTransformer
        Loaded model whose first module wraps a decoder backbone.

    Raises
    ------
    TypeError
        If the backbone has no decoder layers or uses sliding-window attention.
    """
    decoder = _decoder(model)
    if getattr(decoder, "has_sliding_layers", False):
        raise TypeError("Sliding-window attention layers are not supported.")
    for layer in decoder.layers:
        layer.self_attn.is_causal = False
    decoder.forward = _with_bidirectional_mask(decoder)
