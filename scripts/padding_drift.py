"""Measure how far bucketed sequence padding moves the embeddings (#143).

``load_sentence_transformer()`` pads every batch to a multiple of
``PAD_TO_MULTIPLE_OF`` tokens to bound MPS memory.  This script encodes the
same D4_instance V2+V6 source texts twice through ``EmbeddingEncoder`` as
the pipeline encodes sources (``role="source"``, so M5 gets its instruction
prefix; batch size reduced for long texts), once padded and once without
bucketing, and records the per-text maximum absolute difference of the
normalised embeddings.  It refuses any device but MPS.

Texts: the ``--n`` longest by character length plus ``--n`` drawn at random
(seed 42) from the rest, verbalised for each model's budget, so they reach
each model's token limit.  Run one model per process: an unpadded encode of
many distinct lengths accumulates MPS driver memory (the problem padding
solves), which exhausts the device for M3 at large ``--n``.

Each run adds or replaces its model's entry in ``--output`` (JSON) and
rewrites the Markdown table beside it.

Usage
-----
for m in M1 M2 M4 M5; do
    python scripts/padding_drift.py --model $m --n 256
done
python scripts/padding_drift.py --model M3 --n 32
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdflib import URIRef

from kgsemembed.datasets import load_dataset
from kgsemembed.embeddings.encoder import EmbeddingEncoder
from kgsemembed.embeddings.models import (
    get_model_config,
    load_sentence_transformer,
    require_device,
)
from kgsemembed.verbalisation.registry import build_verbaliser

_DEFAULT_OUTPUT = Path("artifacts/issue-143/padding_drift.json")
_ROLE = "source"
_STRATEGY = "V2+V6"


def sample_texts(model_key: str, data_dir: str, n: int) -> List[str]:
    """
    Verbalise D4_instance sources and pick the longest and a random sample.

    Parameters
    ----------
    model_key : str
        Model whose budget the verbaliser applies.
    data_dir : str
        Dataset root directory.
    n : int
        Number of longest texts, and of further random texts.

    Returns
    -------
    List[str]
        ``2 * n`` texts.
    """
    pair = next(p for p in load_dataset("D4", data_dir) if p.dataset_id == "D4_instance")
    verbaliser = build_verbaliser(_STRATEGY, model_key)
    texts = [
        verbaliser.verbalise(pair.source_graph, URIRef(uri), "instance")
        for uri in sorted(pair.source_entities)
    ]
    order = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))
    picked = order[:n] + random.sample(order[n:], n)
    return [texts[i] for i in picked]


def drift_per_text(encoder: EmbeddingEncoder, texts: List[str]) -> np.ndarray:
    """
    Encode ``texts`` padded and unpadded and return each text's drift.

    Parameters
    ----------
    encoder : EmbeddingEncoder
        Encoder whose model was loaded with bucketed padding.
    texts : List[str]
        Texts to encode.

    Returns
    -------
    np.ndarray
        Maximum absolute difference per text.
    """
    padded = encoder.encode_batch(texts, role=_ROLE, show_progress=False)
    module = encoder.model[0]
    bucketing = module.processing_kwargs
    module.processing_kwargs = {}
    try:
        plain = encoder.encode_batch(texts, role=_ROLE, show_progress=False)
    finally:
        module.processing_kwargs = bucketing
    return np.abs(padded - plain).max(axis=1)


def measure(model_key: str, data_dir: str, n: int) -> Dict[str, object]:
    """
    Measure padding drift of one model on its D4_instance sample.

    Parameters
    ----------
    model_key : str
        Registered model key.
    data_dir : str
        Dataset root directory.
    n : int
        Sample size per half (longest, random).

    Returns
    -------
    Dict[str, object]
        Sample description and drift statistics.
    """
    texts = sample_texts(model_key, data_dir, n)
    config = get_model_config(model_key)
    model, info = load_sentence_transformer(model_key)
    require_device(info)
    encoder = EmbeddingEncoder(model_key, model)
    prefixed = [encoder._apply_prefix(t, _ROLE) for t in texts]
    lengths = [
        len(model.tokenizer(t, truncation=True, max_length=config.max_tokens)["input_ids"])
        for t in prefixed
    ]
    drift = drift_per_text(encoder, texts)
    return {
        "model_id": config.model_id, "hf_revision": info["hf_revision"],
        "device": info["device"], "role": _ROLE,
        "batch_size": encoder._effective_batch_size(prefixed),
        "configured_batch_size": config.batch_size,
        "n_texts": len(texts), "median_tokens": int(np.median(lengths)),
        "max_tokens": int(max(lengths)), "max_abs_diff": float(drift.max()),
        "p99_abs_diff": float(np.quantile(drift, 0.99)),
        "n_above_1e-6": int((drift > 1e-6).sum()),
    }


def write_outputs(output: Path, model_key: str, entry: Dict[str, object]) -> None:
    """
    Merge ``entry`` into the JSON record and rewrite the Markdown table.

    Parameters
    ----------
    output : Path
        JSON path; the Markdown table is written beside it.
    model_key : str
        Key of the measured model.
    entry : Dict[str, object]
        Statistics returned by :func:`measure`.
    """
    record = json.loads(output.read_text()) if output.exists() else {}
    record[model_key] = entry
    record = dict(sorted(record.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown_table(record))


def markdown_table(record: Dict[str, Dict[str, object]]) -> str:
    """
    Render the drift record as a Markdown table.

    Parameters
    ----------
    record : Dict[str, Dict[str, object]]
        Per-model statistics.

    Returns
    -------
    str
        Markdown document.
    """
    lines = [
        "# Padding drift (#143)", "",
        f"D4_instance {_STRATEGY} source texts, padded vs unpadded, normalised "
        f"embeddings, encoded as `role=\"{_ROLE}\"` with `EmbeddingEncoder` "
        "batching (Batch = effective / configured). Generated by "
        "`scripts/padding_drift.py`.", "",
        "| Model | Device | Texts | Batch | Tokens median / max | Max abs diff | p99 | Texts > 1e-6 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, e in record.items():
        lines.append(
            f"| {key} | {e['device']} | {e['n_texts']} | {e['batch_size']} / "
            f"{e['configured_batch_size']} | {e['median_tokens']} / "
            f"{e['max_tokens']} | {e['max_abs_diff']:.2e} | {e['p99_abs_diff']:.2e} | "
            f"{e['n_above_1e-6']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments, measure one model and record it."""
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Model key, e.g. M2.")
    parser.add_argument("--n", type=int, default=256, help="Longest and random texts each.")
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="JSON record.")
    args = parser.parse_args()
    entry = measure(args.model, args.data_dir, args.n)
    write_outputs(args.output, args.model, entry)
    print(json.dumps({args.model: entry}, indent=2))


if __name__ == "__main__":
    main()
