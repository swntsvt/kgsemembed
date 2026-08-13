"""Central registry of embedding model configurations and a lazy factory.

This module is the single authoritative source of embedding model metadata for
Phase 2 experiments: model identifiers, token limits, PPAS budgets, prompting
behaviour, and recommended batch sizes.  Configuration,
lookup, device selection, and model loading are kept as separate
responsibilities so that importing this module has no side effects and never
instantiates a model.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer


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
    max_tokens : int
        Maximum input sequence length supported by the model.
    ppas_budget : Optional[int]
        Verbalisation token budget for PPAS, or ``None`` when uncapped.
    query_prefix : Optional[str]
        Instruction prepended to query text, or ``None`` when unused.
    doc_prefix : Optional[str]
        Instruction prepended to document text, or ``None`` when unused.
    batch_size : int
        Recommended encoding batch size for this model.
    trust_remote_code : bool
        Whether loading executes custom modelling code published in the model
        repository. Enabled only for models whose architecture is not part of
        ``transformers``.
    """

    model_key: str
    model_id: str
    max_tokens: int
    ppas_budget: Optional[int]
    query_prefix: Optional[str]
    doc_prefix: Optional[str]
    batch_size: int
    trust_remote_code: bool = False


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "M1": ModelConfig(
        model_key="M1",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=256,
        ppas_budget=200,
        query_prefix=None,
        doc_prefix=None,
        batch_size=64,
    ),
    "M2": ModelConfig(
        model_key="M2",
        model_id="BAAI/bge-large-en-v1.5",
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
    "M3": ModelConfig(
        model_key="M3",
        model_id="BAAI/bge-m3",
        max_tokens=8192,
        ppas_budget=None,
        query_prefix=None,
        doc_prefix=None,
        batch_size=16,
        # BGE-M3 is a stock XLM-RoBERTa architecture, so the long context is
        # available without executing modelling code from the repository.
        trust_remote_code=False,
    ),
    "M4": ModelConfig(
        model_key="M4",
        model_id="FremyCompany/BioLORD-2023",
        max_tokens=512,
        ppas_budget=420,
        query_prefix=None,
        doc_prefix=None,
        batch_size=32,
    ),
    "M5": ModelConfig(
        model_key="M5",
        model_id="dunzhang/stella_en_1.5B_v5",
        max_tokens=512,
        ppas_budget=420,
        # Stella uses asymmetric prompting: an instruction is prepended to
        # queries only, while documents are encoded verbatim.  The query
        # instruction is required to reach reported retrieval quality and must
        # not be applied to documents.
        query_prefix="Instruct: Retrieve semantically similar text.\nQuery: {}",
        doc_prefix=None,
        batch_size=8,
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
        holding ``model_key``, ``model_id``, ``device`` and ``hf_revision``.
    """
    config = get_model_config(model_key)
    device = _select_device()
    print(f"Loading model {config.model_id} on device {device}")
    model = SentenceTransformer(
        config.model_id, device=device, trust_remote_code=config.trust_remote_code
    )
    model_info = {
        "model_key": model_key,
        "model_id": config.model_id,
        "device": device,
        "hf_revision": _resolve_hf_revision(model),
    }
    return model, model_info
