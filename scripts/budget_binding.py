"""How often the token budget binds in the budgeted main conditions (#183).

For every budgeted condition (a V3–V7 component under a model with a budget)
and every dataset column the registry assigns it, this verbalises the texts
the pipeline encodes (every source and every unique candidate) *without* the
budget, by building the same strategy under ``M2_uncapped`` (no budget), and
counts their tokens with the condition model's tokenizer at its pinned
revision, special tokens included.  A text binds when that full length exceeds
the budget: the pipeline then dropped items from it (or, rarely, cut it at a
word).  Texts over the encoder limit are those the model would have truncated
without a budget.

Only M2 and M4 carry budgeted conditions, and neither adds a prompt prefix, so
the full length is the encoder input the unbudgeted text would give.
Tokenizer only: no model is loaded and nothing under ``data/`` is written.

Usage
-----
python scripts/budget_binding.py --data_dir data/ --output_dir artifacts/issue-183/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoded_input_size import (  # noqa: E402
    COLUMNS,
    DATASETS,
    ROLES,
    TokenLengths,
    compared_conditions,
    token_length_counter,
)
from kgsemembed.candidates import load_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair, load_dataset  # noqa: E402
from kgsemembed.embeddings.models import get_model_config  # noqa: E402
from kgsemembed.pipeline.conditions import ExperimentCondition  # noqa: E402
from kgsemembed.pipeline.run_experiment import _unique_candidate_uris, _verbalise_uris  # noqa: E402
from kgsemembed.verbalisation.budget import budget_tokens, strategy_is_budgeted  # noqa: E402
from kgsemembed.verbalisation.registry import build_verbaliser  # noqa: E402

UNBUDGETED_MODEL = "M2_uncapped"
JSON_NAME = "budget_binding.json"
MARKDOWN_NAME = "budget_binding.md"


def budgeted_conditions(dataset_id: str) -> List[ExperimentCondition]:
    """
    Return the compared conditions on a dataset whose strategy has a budget.

    Parameters
    ----------
    dataset_id : str
        Loader key, ``"D1"`` to ``"D5"``.

    Returns
    -------
    List[ExperimentCondition]
        Budgeted conditions in registry order, the sanity check excluded.
    """
    return [c for c in compared_conditions(dataset_id) if strategy_is_budgeted(c.strategy_name, c.model_key)]


def unbudgeted_texts(pair: AlignmentPair, strategy: str, data_dir: str) -> Dict[str, List[str]]:
    """
    Verbalise the pair's encoder inputs with the strategy but no budget.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair to verbalise.
    strategy : str
        Registered strategy name.
    data_dir : str
        Dataset and candidate root directory.

    Returns
    -------
    Dict[str, List[str]]
        Full texts per role: every source and every unique candidate.
    """
    candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    uris = {"source": pair.source_entities, "candidate": _unique_candidate_uris(pair, candidates)}
    graphs = {"source": pair.source_graph, "candidate": pair.target_graph}
    return {
        role: _verbalise_uris(
            build_verbaliser(strategy, UNBUDGETED_MODEL, role), graphs[role], uris[role], pair.entity_type, role
        )
        for role in ROLES
    }


def binding_summary(lengths: List[int], budget: int, limit: int) -> Dict[str, float]:
    """
    Summarise how often full lengths exceed the budget and the encoder limit.

    Parameters
    ----------
    lengths : List[int]
        Full encoded length of every text.
    budget : int
        The condition's token budget.
    limit : int
        The encoder's input limit.

    Returns
    -------
    Dict[str, float]
        Texts, texts over the budget and over the limit, total full tokens and
        tokens beyond the budget.
    """
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "n_texts": int(values.size),
        "n_over_budget": int((values > budget).sum()),
        "n_over_limit": int((values > limit).sum()),
        "tokens": int(values.sum()),
        "tokens_beyond_budget": int(np.maximum(values - budget, 0).sum()),
    }


def cell_record(
    column: str, condition: ExperimentCondition, pooled: Dict[str, List[int]]
) -> Dict[str, object]:
    """
    Build one condition × column record from pooled full lengths.

    Parameters
    ----------
    column : str
        Dataset column.
    condition : ExperimentCondition
        The budgeted condition.
    pooled : Dict[str, List[int]]
        Full token lengths per role.

    Returns
    -------
    Dict[str, object]
        Identity, budget, limit and a binding summary per role and in total.
    """
    budget = budget_tokens(condition.model_key)
    limit = get_model_config(condition.model_key).max_tokens
    summaries = {role: binding_summary(pooled[role], budget, limit) for role in ROLES}
    total = binding_summary(pooled["source"] + pooled["candidate"], budget, limit)
    return {
        "column": column, "condition_id": condition.condition_id, "strategy": condition.strategy_name,
        "model_key": condition.model_key, "token_budget": budget, "max_seq_length": limit,
        **summaries, "total": total,
    }


def column_lengths(
    pairs: List[AlignmentPair], strategy: str, data_dir: str, counter: TokenLengths
) -> Dict[str, List[int]]:
    """
    Pool full token lengths per role over the pairs of one column.

    Parameters
    ----------
    pairs : List[AlignmentPair]
        The column's alignment pairs.
    strategy : str
        Registered strategy name.
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
        texts = unbudgeted_texts(pair, strategy, data_dir)
        for role in ROLES:
            pooled[role].extend(counter(texts[role]))
    return pooled


def measure_dataset(dataset_id: str, data_dir: str, counters: Dict[str, TokenLengths]) -> List[Dict[str, object]]:
    """
    Measure every budgeted condition on every column of one dataset.

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
        One record per condition and column.
    """
    pairs = load_dataset(dataset_id, data_dir)
    records = []
    for column in sorted({pair.dataset_id for pair in pairs}, key=COLUMNS.index):
        column_pairs = [pair for pair in pairs if pair.dataset_id == column]
        for condition in budgeted_conditions(dataset_id):
            if condition.model_key not in counters:
                counters[condition.model_key] = token_length_counter(condition.model_key)
            pooled = column_lengths(column_pairs, condition.strategy_name, data_dir, counters[condition.model_key])
            records.append(cell_record(column, condition, pooled))
            print(f"{column} {condition.condition_id}: {records[-1]['total']['n_over_budget']} over budget", flush=True)
    return records


def _share(part: int, whole: int) -> str:
    return f"{part:,} ({100 * part / whole:.1f}%)" if whole else "0"


def _row(record: Dict[str, object]) -> str:
    source, candidate, total = record["source"], record["candidate"], record["total"]
    cells = [
        record["column"], record["condition_id"], record["strategy"], record["model_key"],
        f"{total['n_texts']:,}", _share(source["n_over_budget"], source["n_texts"]),
        _share(candidate["n_over_budget"], candidate["n_texts"]),
        _share(total["n_over_budget"], total["n_texts"]), _share(total["n_over_limit"], total["n_texts"]),
        f"{100 * total['tokens_beyond_budget'] / total['tokens']:.1f}%",
    ]
    return "| " + " | ".join(cells) + " |"


def render_markdown(records: List[Dict[str, object]]) -> str:
    """
    Render the binding counts as a Markdown table.

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
        "# How often the token budget binds (#183)",
        "",
        "Generated by `scripts/budget_binding.py` (tokenizer only). Texts: every source and every "
        "unique candidate the pipeline encodes, verbalised by the condition's strategy without a "
        "budget and counted with the condition model's tokenizer (special tokens included). "
        "*Over budget*: texts whose full length exceeds the budget (420 for M2 and M4), so the "
        "pipeline dropped items from them. *Over limit*: texts over the encoder's 512 tokens, "
        "which the model would truncate without a budget. *Beyond budget*: share of all full "
        "tokens above the budget. D3 pools its 21 pairs.",
        "",
        "| Column | Cond. | Strategy | Model | Texts | Over budget (src) | Over budget (cand) "
        "| Over budget | Over limit | Beyond budget |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
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
    parser.add_argument("--output_dir", default="artifacts/issue-183/")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    return parser.parse_args()


def main() -> None:
    """Measure every dataset and write the JSON and Markdown tables."""
    np.random.seed(42)
    random.seed(42)
    args = parse_args()
    counters: Dict[str, TokenLengths] = {}
    records = [record for dataset_id in args.datasets for record in measure_dataset(dataset_id, args.data_dir, counters)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / JSON_NAME).write_text(json.dumps(records, indent=2) + "\n")
    (output_dir / MARKDOWN_NAME).write_text(render_markdown(records))


if __name__ == "__main__":
    main()
