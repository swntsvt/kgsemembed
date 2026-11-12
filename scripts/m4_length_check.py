"""M4 (BioLORD-2023) input limit: 128 against 512, on validation sources only (#183).

BioLORD-2023's ``sentence_bert_config.json`` sets ``max_seq_length`` 128, while
its architecture takes 512 tokens.  This script compares the two limits, each
with the verbalisation budget it implies (``min(420, limit)``: 128 and 420),
for C6, C16, C11 and C17 on D1 and D2.  It encodes and ranks only the sources
with a validation reference, so no test source is ever ranked, and scores
them by re-ranking MRR as the pipeline does (known pairs removed).

The rule fixed on #183 before this run: register 512 unless, on D1 or D2, the
95% paired bootstrap interval of validation MRR (512 − 128) for C6 or C16 lies
entirely below 0 (#140 settings: 10,000 replicates, seed 42).  C11 and C17 are
reported, not part of the rule.

The tree must be clean under ``src/`` and ``scripts/``.  Output:
``artifacts/issue-183/m4_length_check.{json,md}``.  Loads M4 once per limit.

Usage
-----
python scripts/m4_length_check.py --data_dir data/ --output_dir artifacts/issue-183/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encode_timing import code_state  # noqa: E402
from kgsemembed.candidates import load_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair, load_dataset  # noqa: E402
from kgsemembed.embeddings import EmbeddingEncoder  # noqa: E402
from kgsemembed.embeddings.models import MODEL_REGISTRY, load_sentence_transformer  # noqa: E402
from kgsemembed.evaluation.paired_tests import (  # noqa: E402
    TestSettings,
    aligned_arrays,
    bootstrap_deltas,
    observed_values,
    percentile_interval,
)
from kgsemembed.evaluation.per_source import gold_rank_records  # noqa: E402
from kgsemembed.evaluation.population import judgeable_ranked_lists  # noqa: E402
from kgsemembed.pipeline.conditions import get_condition  # noqa: E402
from kgsemembed.pipeline.run_experiment import (  # noqa: E402
    _build_ranked_lists,
    _release_model,
    _unique_candidate_uris,
    _verbalise_uris,
)
from kgsemembed.verbalisation.registry import build_verbaliser  # noqa: E402

LIMITS = (128, 512)
CONDITIONS = ("C6", "C16", "C11", "C17")
RULE_CONDITIONS = ("C6", "C16")
DATASETS = ("D1", "D2")
Values = Dict[str, Tuple[float, float]]


@contextmanager
def m4_limit(limit: int) -> Iterator[None]:
    """
    Register ``limit`` as M4's input limit for the duration of the block.

    Parameters
    ----------
    limit : int
        Input limit in tokens.

    Yields
    ------
    None
    """
    original = MODEL_REGISTRY["M4"]
    MODEL_REGISTRY["M4"] = dataclasses.replace(original, max_tokens=limit)
    try:
        yield
    finally:
        MODEL_REGISTRY["M4"] = original


def validation_pair(pair: AlignmentPair) -> AlignmentPair:
    """
    Restrict a pair to the sources of its validation references.

    Parameters
    ----------
    pair : AlignmentPair
        The full pair.

    Returns
    -------
    AlignmentPair
        The same pair whose ``source_entities`` are the validation reference
        sources only.

    Raises
    ------
    ValueError
        If a validation source also has a test reference.
    """
    sources = {source for source, _ in pair.val_refs}
    if sources & {source for source, _ in pair.test_refs}:
        raise ValueError(f"{pair.pair_name}: validation and test sources overlap")
    return dataclasses.replace(pair, source_entities=sorted(sources))


def verbalised(pair: AlignmentPair, strategy: str, candidates: Dict[str, List[str]]) -> Dict[str, object]:
    """
    Verbalise the pair's sources and unique candidates under M4's current budget.

    Parameters
    ----------
    pair : AlignmentPair
        The validation-only pair.
    strategy : str
        Registered strategy.
    candidates : Dict[str, List[str]]
        Stored stage-1 candidate lists.

    Returns
    -------
    Dict[str, object]
        ``sources`` and ``candidates`` texts, and candidate ``uris``.
    """
    uris = _unique_candidate_uris(pair, candidates)
    source = build_verbaliser(strategy, "M4", "source")
    candidate = build_verbaliser(strategy, "M4", "candidate")
    return {
        "sources": _verbalise_uris(source, pair.source_graph, pair.source_entities, pair.entity_type, "sources"),
        "uris": uris,
        "candidates": _verbalise_uris(candidate, pair.target_graph, uris, pair.entity_type, "candidates"),
    }


def rank_values(pair: AlignmentPair, texts: Dict[str, object], encoder: EmbeddingEncoder, candidates: dict) -> Values:
    """
    Encode, rank and score the validation reference sources.

    Parameters
    ----------
    pair : AlignmentPair
        The validation-only pair.
    texts : Dict[str, object]
        From :func:`verbalised`.
    encoder : EmbeddingEncoder
        Encoder of M4 at the current limit.
    candidates : dict
        Stored stage-1 candidate lists.

    Returns
    -------
    Values
        Per source, ``(reciprocal best gold rank or 0, 1)``.
    """
    sources = encoder.encode_batch(texts["sources"], role="source", show_progress=False)
    targets = encoder.encode_batch(texts["candidates"], role="candidate", show_progress=False)
    index = {uri: position for position, uri in enumerate(texts["uris"])}
    lists = judgeable_ranked_lists(pair, _build_ranked_lists(pair, sources, candidates, index, targets))
    records = gold_rank_records(lists, pair.val_refs)
    return {r["source"]: (1.0 / r["best_gold_rank"] if r["best_gold_rank"] else 0.0, 1.0) for r in records}


def run_limit(limit: int, pairs: List[AlignmentPair], data_dir: str) -> Dict[str, dict]:
    """
    Score every condition on every pair with M4 at ``limit``.

    Parameters
    ----------
    limit : int
        M4's input limit.
    pairs : List[AlignmentPair]
        Validation-only pairs of D1 and D2.
    data_dir : str
        Dataset and candidate root directory.

    Returns
    -------
    Dict[str, dict]
        Keyed ``"{condition}/{dataset}"``: unit ``values`` and ``truncation``.
    """
    out: Dict[str, dict] = {}
    with m4_limit(limit):
        model, _ = load_sentence_transformer("M4")
        encoder = EmbeddingEncoder("M4", model)
        for condition_id in CONDITIONS:
            for pair in pairs:
                candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
                texts = verbalised(pair, get_condition(condition_id).strategy_name, candidates)
                out[f"{condition_id}/{pair.dataset_id}"] = {
                    "values": rank_values(pair, texts, encoder, candidates),
                    "truncation": {role: encoder.truncation_summary(texts[key], role)
                                   for role, key in (("source", "sources"), ("candidate", "candidates"))},
                }
        _release_model(model)
    return out


def comparison(short: dict, long: dict) -> dict:
    """
    Compare one condition and dataset at 128 (A) and 512 (B).

    Parameters
    ----------
    short, long : dict
        Entries of :func:`run_limit` at 128 and 512.

    Returns
    -------
    dict
        Sources, MRR at each limit, their difference and its 95% interval.
    """
    pairs = [aligned_arrays(short["values"], long["values"], "mrr")]
    mrr_128, mrr_512 = observed_values(pairs, "mrr")
    low, high = percentile_interval(bootstrap_deltas(pairs, "mrr", TestSettings()))
    return {"n_sources": len(short["values"]), "mrr_128": mrr_128, "mrr_512": mrr_512,
            "delta": mrr_512 - mrr_128, "ci_low": low, "ci_high": high,
            "truncation_128": short["truncation"], "truncation_512": long["truncation"]}


def decide(rows: Dict[str, dict]) -> dict:
    """
    Apply the rule fixed on #183 before the run.

    Parameters
    ----------
    rows : Dict[str, dict]
        Comparisons keyed ``"{condition}/{dataset}"``.

    Returns
    -------
    dict
        ``limit`` (512 or 128) and the rows that, if any, kept it at 128.
    """
    worse = [key for key, row in rows.items() if key.split("/")[0] in RULE_CONDITIONS and row["ci_high"] < 0]
    return {"limit": 128 if worse else 512, "worse_at_512": worse,
            "rule": "512 unless the 95% interval of validation MRR (512 - 128) for C6 or C16 "
                    "lies entirely below 0 on D1 or D2"}


def render_markdown(report: dict) -> str:
    """
    Render the comparison and decision as Markdown.

    Parameters
    ----------
    report : dict
        The JSON report.

    Returns
    -------
    str
        The Markdown document.
    """
    lines = [
        "# M4 input limit: 128 against 512 (#183)", "",
        "Generated by `scripts/m4_length_check.py` at commit " + report["git_commit"][:7] + ". Validation "
        "reference sources only (no test source ranked); re-ranking MRR as the pipeline computes it; "
        "95% paired bootstrap interval, 10,000 replicates, seed 42. Budget: 128 at 128, 420 at 512.", "",
        f"**Rule (fixed on #183 before the run):** {report['decision']['rule']}.", "",
        f"**Decision: register {report['decision']['limit']}.**", "",
        "| Cond. | Dataset | Sources | MRR 128 | MRR 512 | Δ (512 − 128) | 95% interval | Truncated at 128 | Truncated at 512 |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for key, row in report["rows"].items():
        cut = {limit: sum(row[f"truncation_{limit}"][role]["n_truncated"] for role in ("source", "candidate"))
               for limit in LIMITS}
        lines.append(f"| {key.split('/')[0]} | {key.split('/')[1]} | {row['n_sources']:,} | {row['mrr_128']:.4f} "
                     f"| {row['mrr_512']:.4f} | {row['delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                     f"| {cut[128]:,} | {cut[512]:,} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Compare M4 at 128 and 512 on validation sources and write the decision."""
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="artifacts/issue-183/")
    args = parser.parse_args()
    state = code_state()
    pairs = [validation_pair(pair) for dataset in DATASETS for pair in load_dataset(dataset, args.data_dir)]
    short, long = (run_limit(limit, pairs, args.data_dir) for limit in LIMITS)
    rows = {key: comparison(short[key], long[key]) for key in short}
    report = {**state, "budgets": {str(limit): min(420, limit) for limit in LIMITS},
              "rows": rows, "decision": decide(rows)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "m4_length_check.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "m4_length_check.md").write_text(render_markdown(report))


if __name__ == "__main__":
    main()
