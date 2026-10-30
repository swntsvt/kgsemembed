"""Central registry of embedding model configurations and a lazy factory.

This module is the single authoritative source of embedding model metadata for
Phase 2 experiments: model identifiers, token limits, PPAS budgets, prompting
behaviour, pinned Hugging Face revisions, and recommended batch sizes.
Configuration,
lookup, device selection, and model loading are kept as separate
responsibilities so that importing this module has no side effects and never
instantiates a model.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from kgsemembed.embeddings.attention import enable_bidirectional_attention
from kgsemembed.utils.errors import ConfigurationError

PAD_TO_MULTIPLE_OF = 64
EXPECTED_DEVICE_PREFIX = "mps"
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
_OFFSET_POSITION_TYPES = frozenset({"roberta", "xlm-roberta", "mpnet"})


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration metadata for a supported embedding model.

    Attributes
    ----------
    model_key : str
        Short experiment key, e.g. ``"M1"``.
    model_id : str
        Hugging Face model identifier passed to ``SentenceTransformer``.
    revision : str
        Hugging Face commit the model is loaded at (#141): the revision every
        stored result was produced with, so a new upstream commit cannot
        change a re-run.
    max_tokens : int
        Encoder input limit in tokens, special tokens and prompt prefix
        included: the loader sets ``SentenceTransformer.max_seq_length`` to it
        (#183), so the model truncates every longer input there.
    ppas_budget : Optional[int]
        Verbalisation token budget for PPAS, or ``None`` when uncapped.
    query_prefix : Optional[str]
        Instruction prepended to query text, or ``None`` when unused.
    doc_prefix : Optional[str]
        Instruction prepended to document text, or ``None`` when unused.
    batch_size : int
        Recommended encoding batch size for this model.
    bidirectional_attention : bool
        Whether the model is a decoder fine-tuned without a causal mask, whose
        trained attention must be restored after loading (default ``False``).
    """

    model_key: str
    model_id: str
    revision: str
    max_tokens: int
    ppas_budget: Optional[int]
    query_prefix: Optional[str]
    doc_prefix: Optional[str]
    batch_size: int
    bidirectional_attention: bool = False


_BGE_LARGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "M1": ModelConfig(
        model_key="M1",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        max_tokens=256,
        ppas_budget=200,
        query_prefix=None,
        doc_prefix=None,
        batch_size=64,
    ),
    "M2": ModelConfig(
        model_key="M2",
        model_id="BAAI/bge-large-en-v1.5",
        revision=_BGE_LARGE_REVISION,
        max_tokens=512,
        ppas_budget=420,
        # BGE v1.5 recommends a retrieval instruction on queries only
        # ("Represent this sentence for searching relevant passages:") for
        # asymmetric query->passage search.  Ontology matching is symmetric, so
        # the prefix is omitted on both sides to avoid biasing similarity.
        query_prefix=None,
        doc_prefix=None,
        batch_size=32,
    ),
    "M2_uncapped": ModelConfig(
        model_key="M2_uncapped",
        model_id="BAAI/bge-large-en-v1.5",
        revision=_BGE_LARGE_REVISION,
        max_tokens=512,
        # The same model as M2 with the verbalisation budget removed, so a PPAS
        # ablation can hold the model fixed instead of confounding the PPAS
        # effect with a change of architecture.
        ppas_budget=None,
        query_prefix=None,
        doc_prefix=None,
        batch_size=64,
    ),
    "M3": ModelConfig(
        model_key="M3",
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        max_tokens=8192,
        ppas_budget=None,
        query_prefix=None,
        doc_prefix=None,
        batch_size=16,
    ),
    "M4": ModelConfig(
        model_key="M4",
        model_id="FremyCompany/BioLORD-2023",
        revision="167aab527b238a50ca65224e6319215d2ff4fc9f",
        # The model's own sentence_bert_config.json sets 128, which every M4
        # result before #183 was encoded at.  512 is its architectural limit
        # (514 positions, offset by 2), registered by the rule fixed on #183
        # before scripts/m4_length_check.py compared the two on validation
        # sources only (artifacts/issue-183/m4_length_check.md).
        max_tokens=512,
        ppas_budget=420,
        query_prefix=None,
        doc_prefix=None,
        batch_size=32,
    ),
    "M5": ModelConfig(
        model_key="M5",
        model_id="dunzhang/stella_en_1.5B_v5",
        revision="7817065102fd9e1b031fe874e910c01f40b2f001",
        max_tokens=512,
        ppas_budget=420,
        # Stella uses asymmetric prompting: an instruction is prepended to
        # queries only, while documents are encoded verbatim.  The query
        # instruction is required to reach reported retrieval quality and must
        # not be applied to documents.
        query_prefix="Instruct: Retrieve semantically similar text.\nQuery: {}",
        doc_prefix=None,
        batch_size=8,
        # Stella is Qwen2 fine-tuned with bidirectional attention, restored by
        # the repository's custom modelling code (auto_map).  The stock Qwen2
        # decoder is causal, so attention must be made bidirectional here.
        bidirectional_attention=True,
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """
    Return the configuration registered under ``model_key``.

    Parameters
    ----------
    model_key : str
        Supported model key, e.g. ``"M1"``.

    Returns
    -------
    ModelConfig
        The immutable configuration for ``model_key``.

    Raises
    ------
    KeyError
        If ``model_key`` is not registered. The message names the invalid key
        and lists the supported keys to aid diagnosis.
    """
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError:
        supported = ", ".join(MODEL_REGISTRY)
        raise KeyError(
            f"Unknown model key: {model_key!r}. "
            f"Supported model keys are: {supported}."
        ) from None


def _select_device() -> str:
    """
    Select the best available torch device.

    Preference order is Apple Silicon MPS, then CUDA, then CPU.

    Returns
    -------
    str
        One of ``"mps"``, ``"cuda"`` or ``"cpu"``.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_hf_revision(model: SentenceTransformer) -> str:
    """
    Determine the Hugging Face revision behind a loaded model.

    The commit hash recorded on the underlying transformer configuration is
    preferred. When that attribute is unavailable the tokenizer's source path
    is used instead, and when neither can be read the revision is unknown.

    Parameters
    ----------
    model : SentenceTransformer
        A loaded model instance.

    Returns
    -------
    str
        The commit hash, the tokenizer source path, or ``"unknown"``.
    """
    try:
        return model[0].auto_model.config._commit_hash
    except (AttributeError, TypeError, IndexError, KeyError):
        pass
    try:
        return model.tokenizer.name_or_path
    except (AttributeError, TypeError):
        return "unknown"


def _check_revision(config: ModelConfig, resolved: str) -> None:
    """
    Raise if a model resolved to a commit other than its pinned revision.

    Only a full commit hash is compared; the tokenizer-path fallback of
    :func:`_resolve_hf_revision` cannot be checked and is left to the record.

    Parameters
    ----------
    config : ModelConfig
        Configuration holding the pinned revision.
    resolved : str
        Revision reported by the loaded model.

    Raises
    ------
    ConfigurationError
        If ``resolved`` is a commit hash different from ``config.revision``.
    """
    if isinstance(resolved, str) and _COMMIT_HASH.fullmatch(resolved) and resolved != config.revision:
        raise ConfigurationError(
            f"{config.model_id} loaded at {resolved}, but {config.revision} is pinned."
        )


def require_device(
    model_info: Dict[str, str], allowed_device: Optional[str] = None
) -> Dict[str, object]:
    """
    Check that a loaded model runs on MPS, or on an explicitly allowed device.

    Every stored result was produced on MPS.  Another device is accepted only
    when named as ``allowed_device``, and the override is returned so that the
    result records it.

    Parameters
    ----------
    model_info : Dict[str, str]
        Metadata returned by :func:`load_sentence_transformer`.
    allowed_device : Optional[str]
        A non-MPS device the caller explicitly accepts, e.g. ``"cpu"``.

    Returns
    -------
    Dict[str, object]
        ``device`` and ``device_override`` (``True`` when the model runs on
        the explicitly allowed non-MPS device).

    Raises
    ------
    ConfigurationError
        If the model is neither on MPS nor on ``allowed_device``.
    """
    device = str(model_info.get("device", "unknown"))
    if device.startswith(EXPECTED_DEVICE_PREFIX):
        return {"device": device, "device_override": False}
    if allowed_device is not None and device == allowed_device:
        return {"device": device, "device_override": True}
    raise ConfigurationError(
        f"Model loaded on {device!r}, expected {EXPECTED_DEVICE_PREFIX!r}; "
        "pass --allow_device to accept it explicitly."
    )


def bucket_sequence_padding(model: SentenceTransformer) -> None:
    """
    Pad every tokenised batch to a multiple of ``PAD_TO_MULTIPLE_OF`` tokens.

    MPS keeps per-shape driver memory that ``torch.mps.empty_cache()`` does
    not release, so encoding a corpus whose batches take hundreds of distinct
    padded lengths (D4 instances under a 420-token budget) exhausts device
    memory.  Bucketing the length bounds the number of shapes.  Padded
    positions are masked, but the result is not exactly neutral: on the
    recorded D4 sample, padded and unpadded normalised embeddings differ by
    at most 3.9e-7 on M1, M2, M3 and M5 (2.2e-6 on an earlier M5 sample), and
    by up to 9.0e-5 on BioLORD (M4) for a few short texts
    (``scripts/padding_drift.py``; ``artifacts/issue-143/padding_drift.md``).

    Parameters
    ----------
    model : SentenceTransformer
        Loaded model whose first module tokenises the input text.
    """
    model[0].processing_kwargs = {"text": {"pad_to_multiple_of": PAD_TO_MULTIPLE_OF}}


def position_capacity(model: SentenceTransformer) -> int:
    """
    Return the longest input the model's position embeddings can take.

    RoBERTa-style models (RoBERTa, XLM-R, MPNet) number positions from the
    padding index plus one, so their usable length is that much shorter than
    ``max_position_embeddings`` (BioLORD: 514 - 2 = 512).  The tokenizer's
    ``model_max_length`` cannot serve: sentence-transformers sets it to the
    model's configured ``max_seq_length`` when loading.

    Parameters
    ----------
    model : SentenceTransformer
        The loaded model.

    Returns
    -------
    int
        Usable input length in tokens.
    """
    config = model[0].auto_model.config
    offset = config.pad_token_id + 1 if config.model_type in _OFFSET_POSITION_TYPES else 0
    return config.max_position_embeddings - offset


def set_input_limit(model: SentenceTransformer, config: ModelConfig) -> None:
    """
    Set the encoder's input limit to the registered one (#183).

    Parameters
    ----------
    model : SentenceTransformer
        The loaded model.
    config : ModelConfig
        Its registry entry; ``max_tokens`` is the limit.

    Raises
    ------
    ConfigurationError
        If the model's position embeddings cannot take inputs of that length.
    """
    capacity = position_capacity(model)
    if config.max_tokens > capacity:
        raise ConfigurationError(
            f"{config.model_key}: max_tokens {config.max_tokens} exceeds the model's position capacity {capacity}"
        )
    model.max_seq_length = config.max_tokens


def load_sentence_transformer(
    model_key: str,
) -> Tuple[SentenceTransformer, Dict[str, str]]:
    """
    Load the ``SentenceTransformer`` for ``model_key`` onto the best device.

    Parameters
    ----------
    model_key : str
        Supported model key, e.g. ``"M1"``.

    Returns
    -------
    Tuple[SentenceTransformer, Dict[str, str]]
        The model placed on the selected device, and provenance metadata
        holding ``model_key``, ``model_id``, ``device``, ``hf_revision`` (as
        loaded), ``hf_revision_pinned`` and ``max_seq_length`` (the input
        limit set from the registry).

    Raises
    ------
    ConfigurationError
        If the model resolves to a commit other than its pinned revision.
    """
    config = get_model_config(model_key)
    device = _select_device()
    print(f"Loading model {config.model_id}@{config.revision[:12]} on device {device}")
    model = SentenceTransformer(config.model_id, device=device, revision=config.revision)
    set_input_limit(model, config)
    bucket_sequence_padding(model)
    if config.bidirectional_attention:
        enable_bidirectional_attention(model)
    revision = _resolve_hf_revision(model)
    _check_revision(config, revision)
    model_info = {
        "model_key": model_key,
        "model_id": config.model_id,
        "device": device,
        "hf_revision": revision,
        "hf_revision_pinned": config.revision,
        "max_seq_length": model.max_seq_length,
    }
    return model, model_info
