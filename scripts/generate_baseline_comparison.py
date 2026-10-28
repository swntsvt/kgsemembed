"""Compare every C1–C20 result with the stage-1 n-gram baseline B0 (#137).

Reads the C-condition results and the B0 results (``ngram_baseline``) and
writes a Markdown report of counts and metrics only (no entity text and no URIs):

- a population check: every C-result is compared with B0 on the same pair by
  ``candidates_sha256``, the ``population`` block and ``evaluation_protocol``;
  mismatching results are listed, and every table excludes them.  Under a
  partial reference (D4) the ranked-source counts are left out of the check:
  an unmatched source is judged only when its top-ranked target is a
  reference target, so those counts depend on each method's ranking;
- B0 per dataset column (MRR, Hits@1, Recall@1, Recall@10, F1, candidate
  recall@20); Hits@1 of a result stored before #139 is derived from its
  Recall@1 with the ``--ceilings`` reference counts, as in the results tables
  and its test-source gold-rank distribution from the B0 sidecars;
- per metric, each condition's difference from B0 per dataset column, with
  counts above/below B0 and the best condition;
- a D5 section setting every D5 condition against B0.

Dataset columns follow the main table: D3 is the unweighted mean over its 21
pairs.  One pair per other column.  The sanity check C19 (a duplicate of C10)
is left out, as in every results table.  This report is descriptive; the
paired intervals and tests of every condition against B0 are in
``artifacts/issue-165/baseline_paired_tests.md`` (#165).

Usage
-----
python scripts/generate_baseline_comparison.py \\
    --results_dir data/results/ --baseline_dir data/results_ngram_baseline/ \\
    --output artifacts/issue-137/baseline_comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kgsemembed.evaluation.aggregator import hits_at_1_or_none, strict_mean
from kgsemembed.evaluation.metrics import EVALUATION_PROTOCOL_VERSION
from kgsemembed.evaluation.reference_counts import DEFAULT_COUNTS_PATH, PairCounts, load_reference_counts
from kgsemembed.pipeline.conditions import SANITY_CHECK_GROUP, get_condition

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_baseline_comparison")

_BASELINE_ID = "B0"
_DATASETS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
_METRICS = ("mrr", "hits_at_1", "recall_at_1", "recall_at_10", "f1")
_METRIC_NAMES = {
    "mrr": "MRR", "hits_at_1": "Hits@1", "recall_at_1": "Recall@1",
    "recall_at_10": "Recall@10", "f1": "F1"}
_MATCH_FIELDS = ("candidates_sha256", "population", "evaluation_protocol")
_PREDICTION_DEPENDENT = ("n_val_ranked_sources", "n_test_ranked_sources")
_RANK_BINS = (("1", 1, 1), ("2–5", 2, 5), ("6–10", 6, 10), ("11–20", 11, 20))

PairKey = Tuple[str, str]


def _read_payloads(paths: List[Path]) -> List[dict]:
    """Parse result files, skipping unreadable ones with a warning."""
    payloads = []
    for path in paths:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Skipping %s: %s", path, exc)
    return payloads


def _is_reported(condition_id: str) -> bool:
    try:
        return get_condition(condition_id).ablation_group != SANITY_CHECK_GROUP
    except KeyError:
        return False


def load_condition_payloads(results_dir: str) -> List[dict]:
    """
    Load every reported C-condition result written under protocol 2.

    Reported means registered and outside the sanity-check group, so C19,
    which must reproduce C10, is never counted against B0 twice.

    Parameters
    ----------
    results_dir : str
        Root of ``{condition}/{dataset}/{pair}_results.json``.

    Returns
    -------
    List[dict]
        Parsed result payloads.
    """
    payloads = _read_payloads(sorted(Path(results_dir).glob("*/*/*_results.json")))
    return [
        p for p in payloads
        if _is_reported(str(p.get("condition_id")))
        and p.get("evaluation_protocol") == EVALUATION_PROTOCOL_VERSION
    ]


def load_baseline(baseline_dir: str) -> Tuple[Dict[PairKey, dict], Dict[PairKey, List[dict]]]:
    """
    Load B0 results and their gold-rank sidecars keyed by (dataset, pair).

    Parameters
    ----------
    baseline_dir : str
        Root holding ``B0/{dataset}/{pair}_results.json`` and ``_entities.json``.

    Returns
    -------
    Tuple[Dict[PairKey, dict], Dict[PairKey, List[dict]]]
        Result payloads and test-source records.
    """
    root = Path(baseline_dir) / _BASELINE_ID
    results = {(p["dataset_id"], p["pair_name"]): p for p in _read_payloads(sorted(root.glob("*/*_results.json")))}
    sidecars = {}
    for key in results:
        path = root / key[0] / f"{key[1]}_entities.json"
        sidecars[key] = json.loads(path.read_text(encoding="utf-8"))["test_sources"]
    return results, sidecars


def mismatch_reasons(payload: dict, baseline: Optional[dict]) -> List[str]:
    """
    List why a C-result is not on the same candidates and populations as B0.

    Parameters
    ----------
    payload : dict
        C-condition result.
    baseline : Optional[dict]
        B0 result of the same pair, or ``None`` when B0 has none.

    Returns
    -------
    List[str]
        Mismatching fields, ``["no B0 result"]``, or empty when matched.
    """
    if baseline is None:
        return ["no B0 result"]
    return [field for field in _MATCH_FIELDS if _comparable(payload, field) != _comparable(baseline, field)]


def _comparable(payload: dict, field: str) -> object:
    """Return a field, without prediction-dependent counts under a partial reference."""
    value = payload.get(field)
    if field != "population" or not isinstance(value, dict) or value.get("reference_complete", True):
        return value
    return {key: item for key, item in value.items() if key not in _PREDICTION_DEPENDENT}


def split_matched(
    payloads: List[dict], baseline: Dict[PairKey, dict]
) -> Tuple[List[dict], List[Tuple[dict, List[str]]]]:
    """Separate C-results that match B0's candidates and populations from the rest."""
    matched, mismatched = [], []
    for payload in payloads:
        reasons = mismatch_reasons(payload, baseline.get((payload["dataset_id"], payload["pair_name"])))
        if reasons:
            mismatched.append((payload, reasons))
        else:
            matched.append(payload)
    return matched, mismatched


def _metric_value(payload: dict, metric: str, counts: Optional[Dict[PairKey, PairCounts]]) -> float:
    if metric != "hits_at_1":
        return float(payload["metrics"][metric])
    value = hits_at_1_or_none(payload, counts)
    return np.nan if value is None else value


def column_means(payloads: List[dict], counts: Optional[Dict[PairKey, PairCounts]] = None) -> pd.DataFrame:
    """
    Average each metric per condition and dataset column.

    Hits@1 not recorded in a result (before #139) is derived from its
    Recall@1 with ``counts``, as in ``report.md``; when it cannot be, that
    column's mean is missing, never an average of the other pairs, and a
    warning names how many results lack it.

    Parameters
    ----------
    payloads : List[dict]
        Result payloads (C-conditions and/or B0).
    counts : Optional[Dict[PairKey, PairCounts]]
        Per-pair reference counts (``scripts/metric_ceilings.py``), or
        ``None`` to use recorded values only.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(condition_id, dataset_id)``, one column per metric.
    """
    rows = [
        {"condition_id": p["condition_id"], "dataset_id": p["dataset_id"],
         **{metric: _metric_value(p, metric, counts) for metric in _METRICS}}
        for p in payloads
    ]
    frame = pd.DataFrame(rows)
    _warn_missing_hits_at_1(frame)
    return frame.groupby(["condition_id", "dataset_id"]).agg(strict_mean)


def _warn_missing_hits_at_1(frame: pd.DataFrame) -> None:
    missing = int(frame["hits_at_1"].isna().sum())
    if missing:
        _LOGGER.warning(
            "%d of %d results lack Hits@1 and cannot derive it; their columns show no Hits@1 mean, "
            "and the Hits@1 tallies leave those conditions out.", missing, len(frame),
        )


def _fmt(value: Optional[float], signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _cell(means: pd.DataFrame, condition_id: str, dataset_id: str, metric: str) -> Optional[float]:
    key = (condition_id, dataset_id)
    if key not in means.index or pd.isna(means.loc[key, metric]):
        return None
    return float(means.loc[key, metric])


def _delta(means: pd.DataFrame, condition_id: str, dataset_id: str, metric: str) -> Optional[float]:
    value = _cell(means, condition_id, dataset_id, metric)
    base = _cell(means, _BASELINE_ID, dataset_id, metric)
    return None if value is None or base is None else value - base


def _conditions(means: pd.DataFrame) -> List[str]:
    ids = {c for c, _ in means.index if c != _BASELINE_ID}
    return sorted(ids, key=lambda c: int(c[1:]))


def recall_at_20_by_column(baseline: Dict[PairKey, dict]) -> Dict[str, float]:
    """Unweighted pair mean of B0's candidate recall@20 per dataset column."""
    frame = pd.DataFrame(
        [{"dataset_id": d, "r20": p["candidate_recall_at_20"]} for (d, _), p in baseline.items()]
    )
    return frame.groupby("dataset_id")["r20"].mean().to_dict()


def rank_distribution(records: List[dict]) -> Dict[str, int]:
    """
    Bin test reference sources by B0's best gold rank.

    Parameters
    ----------
    records : List[dict]
        Sidecar records of one or more pairs.

    Returns
    -------
    Dict[str, int]
        Counts per rank bin, plus ``> 20`` (ranked, no gold in the list) and
        ``unranked`` (no ranked list at all), and ``n``.
    """
    counts = {label: 0 for label, _, _ in _RANK_BINS}
    counts.update({"> 20": 0, "unranked": 0, "n": len(records)})
    for record in records:
        rank = record["best_gold_rank"]
        if not record["ranked"]:
            counts["unranked"] += 1
        elif rank is None:
            counts["> 20"] += 1
        else:
            counts[next(label for label, low, high in _RANK_BINS if low <= rank <= high)] += 1
    return counts


def _population_section(matched: List[dict], mismatched: List[Tuple[dict, List[str]]]) -> List[str]:
    lines = ["## Population check", "",
             f"{len(matched)} C-results share B0's `candidates_sha256`, `population` block and "
             f"`evaluation_protocol` on the same pair; {len(mismatched)} do not and are excluded below. "
             "Under D4's partial reference the judgeable ranked-source counts are excluded from the "
             "check, because they depend on each method's top-ranked target.", ""]
    rows = [[p["condition_id"], p["dataset_id"], p["pair_name"], ", ".join(r)] for p, r in mismatched]
    return lines + (_table(["Condition", "Dataset", "Pair", "Mismatch"], rows) + [""] if rows else [])


def _baseline_section(means: pd.DataFrame, r20: Dict[str, float]) -> List[str]:
    rows = [
        [name, *(_fmt(_cell(means, _BASELINE_ID, d, m)) for d in _DATASETS)]
        for m, name in _METRIC_NAMES.items()
    ]
    rows.append(["Candidate recall@20", *(_fmt(r20.get(d)) for d in _DATASETS)])
    return ["## B0: stage-1 n-gram order", "", *_table(["Metric", *_DATASETS], rows), ""]


def _distribution_section(sidecars: Dict[PairKey, List[dict]]) -> List[str]:
    labels = [label for label, _, _ in _RANK_BINS] + ["> 20", "unranked", "n"]
    rows = []
    for dataset_id in _DATASETS:
        records = [r for (d, _), recs in sidecars.items() if d == dataset_id for r in recs]
        counts = rank_distribution(records)
        rows.append([dataset_id, *(str(counts[label]) for label in labels)])
    title = "## B0 best gold rank per test reference source (counts; D3 summed over pairs)"
    return [title, "", *_table(["Dataset", *labels], rows), ""]


def _delta_section(means: pd.DataFrame, metric: str) -> List[str]:
    conditions = _conditions(means)
    rows = [[c, *(_fmt(_delta(means, c, d, metric), signed=True) for d in _DATASETS)] for c in conditions]
    rows.append(["above / below / equal B0", *(_tally(means, conditions, d, metric) for d in _DATASETS)])
    rows.append(["best condition", *(_best(means, conditions, d, metric) for d in _DATASETS)])
    name = _METRIC_NAMES[metric]
    return [f"## {name}: condition − B0", "", *_table(["Condition", *_DATASETS], rows), ""]


def _tally(means: pd.DataFrame, conditions: List[str], dataset_id: str, metric: str) -> str:
    deltas = [d for c in conditions if (d := _delta(means, c, dataset_id, metric)) is not None]
    return f"{sum(d > 0 for d in deltas)} / {sum(d < 0 for d in deltas)} / {sum(d == 0 for d in deltas)}"


def _best(means: pd.DataFrame, conditions: List[str], dataset_id: str, metric: str) -> str:
    values = {c: v for c in conditions if (v := _cell(means, c, dataset_id, metric)) is not None}
    if not values:
        return "—"
    best = max(values, key=values.get)
    return f"{best} ({_fmt(values[best])})"


def _d5_section(means: pd.DataFrame) -> List[str]:
    rows = [[_BASELINE_ID, *(_fmt(_cell(means, _BASELINE_ID, "D5", m)) for m in _METRICS), *["—"] * len(_METRICS)]]
    for c in _conditions(means):
        if _cell(means, c, "D5", "mrr") is None:
            continue
        values = [_fmt(_cell(means, c, "D5", m)) for m in _METRICS]
        rows.append([c, *values, *(_fmt(_delta(means, c, "D5", m), signed=True) for m in _METRICS)])
    names = [_METRIC_NAMES[m] for m in _METRICS]
    header = ["Condition", *names, *(f"Δ{name}" for name in names)]
    return ["## D5: every condition against B0", "", *_table(header, rows), ""]


def build_report(
    results_dir: str, baseline_dir: str, counts: Optional[Dict[PairKey, PairCounts]] = None
) -> List[str]:
    """
    Build the Markdown comparison of C1–C20 against B0.

    Parameters
    ----------
    results_dir : str
        Root of the C-condition results.
    baseline_dir : str
        Root of the B0 results.
    counts : Optional[Dict[PairKey, PairCounts]]
        Per-pair reference counts for deriving older results' Hits@1.

    Returns
    -------
    List[str]
        Report lines.
    """
    baseline, sidecars = load_baseline(baseline_dir)
    matched, mismatched = split_matched(load_condition_payloads(results_dir), baseline)
    means = column_means(matched + list(baseline.values()), counts)
    lines = ["# Stage-1 n-gram baseline (B0) against C1–C20", "", _SCOPE, ""]
    lines += _population_section(matched, mismatched) + _baseline_section(means, recall_at_20_by_column(baseline))
    lines += _distribution_section(sidecars)
    for metric in _METRICS:
        lines += _delta_section(means, metric)
    return lines + _d5_section(means)


_SCOPE = (
    "Every row ranks the same stored top-20 candidate lists for the same validation and test "
    "populations, with the threshold tuned on validation only (evaluation protocol 2). B0 is "
    "the stage-1 character 3-gram cosine order of those lists; a condition's difference from "
    "B0 is what embedding re-ranking adds or costs. D3 is the unweighted mean over 21 pairs; "
    "every other column is one pair. The sanity check C19 is left out. Descriptive; "
    "the paired intervals and tests against B0 are in `artifacts/issue-165/baseline_paired_tests.md`."
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare C1–C20 results with the B0 n-gram baseline.")
    parser.add_argument("--results_dir", default="data/results/", help="C-condition results root.")
    parser.add_argument("--baseline_dir", default="data/results_ngram_baseline/", help="B0 results root.")
    parser.add_argument(
        "--output", default="artifacts/issue-137/baseline_comparison.md", help="Markdown output path."
    )
    parser.add_argument(
        "--ceilings", default=str(DEFAULT_COUNTS_PATH),
        help="Per-pair reference counts (scripts/metric_ceilings.py) for older results' Hits@1.",
    )
    return parser


def _load_counts(path: str) -> Optional[Dict[PairKey, PairCounts]]:
    if not Path(path).is_file():
        _LOGGER.warning("No reference counts at %s; older results' Hits@1 omitted.", path)
        return None
    return load_reference_counts(path)


def main() -> None:
    """Write the baseline comparison report."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args.results_dir, args.baseline_dir, _load_counts(args.ceilings))
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    _LOGGER.info("Wrote %s", output)


if __name__ == "__main__":
    main()
