"""Encoded input size of every condition on every column it ran on (#181).

For every registered condition but the sanity check C19, and every dataset
column the registry assigns it, this verbalises the texts the pipeline encodes
(every source of the pair and every unique candidate, exactly as
``run_experiment._process_pair`` does) and counts their tokens under the
model's own tokenizer at its pinned revision.  Counts are taken on the text the
encoder receives (M5's source prefix included, special tokens included) before
the model's cut; the encoded count caps each text there.  The cut is the
model's own ``max_seq_length`` (its ``sentence_bert_config.json`` at the pinned
revision), which is what ``SentenceTransformer.encode`` truncates at; it is
not the registry's ``max_tokens`` (M4: 128 against 512).

Token counts drive the encoder's cost and do not depend on hardware, so they
are the reproducible half of the cost report; the D1 wall-clock half is
``scripts/encode_timing.py``.  Tokenizer only: no model is loaded and nothing
under ``data/`` is written.

Usage
-----
python scripts/encoded_input_size.py --data_dir data/ \\
    --output_dir artifacts/issue-181/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgsemembed.candidates import load_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair, load_dataset  # noqa: E402
from kgsemembed.embeddings.encoder import EmbeddingEncoder  # noqa: E402
from kgsemembed.embeddings.models import get_model_config  # noqa: E402
from kgsemembed.pipeline.conditions import (  # noqa: E402
    EXPERIMENT_CONDITIONS,
    SANITY_CHECK_GROUP,
    ExperimentCondition,
)
from kgsemembed.pipeline.run_experiment import (  # noqa: E402
    _unique_candidate_uris,
    _verbalise_uris,
)
from kgsemembed.verbalisation.registry import build_verbaliser  # noqa: E402

DATASETS = ("D1", "D2", "D3", "D4", "D5")
COLUMNS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
ROLES = ("source", "candidate")
JSON_NAME = "encoded_input_size.json"
MARKDOWN_NAME = "encoded_input_size.md"

TokenLengths = Callable[[List[str]], List[int]]


def compared_conditions(dataset_id: str) -> List[ExperimentCondition]:
    """
    Return the compared conditions registered on a dataset, in registry order.

    Parameters
    ----------
    dataset_id : str
        Loader key, ``"D1"`` to ``"D5"``.

    Returns
    -------
    List[ExperimentCondition]
        Every condition listing the dataset, the sanity-check group excluded.
    """
    return [
        condition
        for condition in EXPERIMENT_CONDITIONS
        if dataset_id in condition.datasets and condition.ablation_group != SANITY_CHECK_GROUP
    ]


def token_length_counter(model_key: str) -> TokenLengths:
    """
    Build a batch token counter from the model's tokenizer at its pinned revision.

    Parameters
    ----------
    model_key : str
        Registered model key, e.g. ``"M2"``.

    Returns
    -------
    TokenLengths
        Function mapping texts to their encoded lengths, special tokens
        included and nothing truncated.
    """
    config = get_model_config(model_key)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)

    def lengths(texts: List[str]) -> List[int]:
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    return lengths


def max_seq_length(model_key: str) -> int:
    """
    Return the length the model truncates at, from its pinned configuration.

    Parameters
    ----------
    model_key : str
        Registered model key, e.g. ``"M4"``.

    Returns
    -------
    int
        ``max_seq_length`` of the model's ``sentence_bert_config.json``.
    """
    config = get_model_config(model_key)
    path = hf_hub_download(config.model_id, "sentence_bert_config.json", revision=config.revision)
    return int(json.loads(Path(path).read_text())["max_seq_length"])


def encoder_inputs(
    pair: AlignmentPair, condition: ExperimentCondition, data_dir: str
) -> Dict[str, List[str]]:
    """
    Verbalise the pair's encoder inputs as the pipeline does, prefixes applied.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair to verbalise.
    condition : ExperimentCondition
        Condition supplying the strategy and model key.
    data_dir : str
        Dataset and candidate root directory.

    Returns
    -------
    Dict[str, List[str]]
        Texts per role, every source and every unique candidate.
    """
    candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    verbaliser = build_verbaliser(condition.strategy_name, condition.model_key)
    uris = {"source": pair.source_entities, "candidate": _unique_candidate_uris(pair, candidates)}
    graphs = {"source": pair.source_graph, "candidate": pair.target_graph}
    prefixer = EmbeddingEncoder(condition.model_key, None)
    return {
        role: [
            prefixer._apply_prefix(text, role)
            for text in _verbalise_uris(verbaliser, graphs[role], uris[role], pair.entity_type, role)
        ]
        for role in ROLES
    }


def length_summary(lengths: List[int], max_tokens: int) -> Dict[str, float]:
    """
    Summarise token lengths of one role on one column.

    Parameters
    ----------
    lengths : List[int]
        Full encoded length of every text.
    max_tokens : int
        The model's token limit; longer texts are truncated by the model.

    Returns
    -------
    Dict[str, float]
        Text count, total and encoded (capped) tokens, mean, median, p95 and
        maximum length, and the number of texts over the limit.
    """
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        return {"n_texts": 0, "tokens": 0, "encoded_tokens": 0, "n_over_limit": 0}
    return {
        "n_texts": int(values.size),
        "tokens": int(values.sum()),
        "encoded_tokens": int(np.minimum(values, max_tokens).sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
        "n_over_limit": int((values > max_tokens).sum()),
    }


def column_lengths(
    pairs: List[AlignmentPair], condition: ExperimentCondition, data_dir: str, counter: TokenLengths
) -> Dict[str, List[int]]:
    """
    Pool token lengths per role over the pairs of one column (D3: 21 pairs).

    Parameters
    ----------
    pairs : List[AlignmentPair]
        The column's alignment pairs.
    condition : ExperimentCondition
        Condition to verbalise for.
    data_dir : str
        Dataset and candidate root directory.
    counter : TokenLengths
        The condition model's token counter.

    Returns
    -------
    Dict[str, List[int]]
        Lengths per role, concatenated over the pairs.
    """
    pooled: Dict[str, List[int]] = {role: [] for role in ROLES}
    for pair in pairs:
        texts = encoder_inputs(pair, condition, data_dir)
        for role in ROLES:
            pooled[role].extend(counter(texts[role]))
    return pooled


def condition_record(
    condition: ExperimentCondition, pooled: Dict[str, List[int]], n_pairs: int
) -> Dict[str, object]:
    """
    Build one condition × column record from pooled lengths.

    Parameters
    ----------
    condition : ExperimentCondition
        The condition measured.
    pooled : Dict[str, List[int]]
        Token lengths per role.
    n_pairs : int
        Pairs in the column.

    Returns
    -------
    Dict[str, object]
        Condition identity, the model's truncation length (and the registry's
        ``max_tokens`` beside it) and a summary per role and in total.
    """
    limit = max_seq_length(condition.model_key)
    every = pooled["source"] + pooled["candidate"]
    return {
        "condition_id": condition.condition_id,
        "strategy": condition.strategy_name,
        "model_key": condition.model_key,
        "max_seq_length": limit,
        "registry_max_tokens": get_model_config(condition.model_key).max_tokens,
        "n_pairs": n_pairs,
        **{role: length_summary(pooled[role], limit) for role in ROLES},
        "total": length_summary(every, limit),
    }


def measure_dataset(
    dataset_id: str, data_dir: str, counters: Dict[str, TokenLengths]
) -> List[Dict[str, object]]:
    """
    Measure every compared condition on every column of one dataset.

    Parameters
    ----------
    dataset_id : str
        Loader key; ``"D4"`` yields its schema and instance columns.
    data_dir : str
        Dataset and candidate root directory.
    counters : Dict[str, TokenLengths]
        Token counters by model key, filled on first use.

    Returns
    -------
    List[Dict[str, object]]
        One record per condition and column, with its ``column``.
    """
    pairs = load_dataset(dataset_id, data_dir)
    records = []
    for column in sorted({pair.dataset_id for pair in pairs}, key=COLUMNS.index):
        column_pairs = [pair for pair in pairs if pair.dataset_id == column]
        for condition in compared_conditions(dataset_id):
            if condition.model_key not in counters:
                counters[condition.model_key] = token_length_counter(condition.model_key)
            counter = counters[condition.model_key]
            pooled = column_lengths(column_pairs, condition, data_dir, counter)
            records.append({"column": column, **condition_record(condition, pooled, len(column_pairs))})
            print(f"{column} {condition.condition_id}: {records[-1]['total']['tokens']} tokens", flush=True)
    return records


def _row(record: Dict[str, object]) -> str:
    total, source, candidate = record["total"], record["source"], record["candidate"]
    cells = [
        record["column"], record["condition_id"], record["strategy"], record["model_key"],
        f"{total['n_texts']:,}", f"{source['mean']:.1f}", f"{candidate['mean']:.1f}",
        f"{total['median']:.0f}", f"{total['p95']:.0f}", f"{total['max']:,}",
        f"{total['encoded_tokens']:,}", f"{total['n_over_limit']:,} / {record['max_seq_length']}",
    ]
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def render_markdown(records: List[Dict[str, object]]) -> str:
    """
    Render the inventory as a Markdown table, one row per condition and column.

    Parameters
    ----------
    records : List[Dict[str, object]]
        Records from :func:`measure_dataset`.

    Returns
    -------
    str
        The Markdown document.
    """
    header = [
        "# Encoded input size per condition and column (#181)",
        "",
        "Generated by `scripts/encoded_input_size.py` (tokenizer only). Texts: every source "
        "and every unique candidate the pipeline encodes, as the encoder receives them "
        "(M5 source prefix and special tokens included). Mean/median/p95/max are full "
        "lengths before the model's limit, its own `max_seq_length` (M4 BioLORD: 128, not "
        "the registry's 512); *encoded tokens* caps each text at that limit; "
        "*over limit* counts texts the model truncates. D3 pools its 21 pairs. "
        "C19 (sanity check, identical to C10) is left out.",
        "",
        "| Column | Cond. | Strategy | Model | Texts | Mean src | Mean cand | Median | P95 | Max "
        "| Encoded tokens | Over limit / limit |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(header + [_row(record) for record in records]) + "\n"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        ``data_dir``, ``output_dir`` and ``datasets``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="artifacts/issue-181/")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    return parser.parse_args()


def main() -> None:
    """Measure every dataset and write the JSON and Markdown inventory."""
    np.random.seed(42)
    random.seed(42)
    args = parse_args()
    counters: Dict[str, TokenLengths] = {}
    records = [
        record
        for dataset_id in args.datasets
        for record in measure_dataset(dataset_id, args.data_dir, counters)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / JSON_NAME).write_text(json.dumps(records, indent=2) + "\n")
    (output_dir / MARKDOWN_NAME).write_text(render_markdown(records))


if __name__ == "__main__":
    main()
