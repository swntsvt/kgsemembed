"""Per-dataset paired intervals behind the claims of Figures 1 and 2 (#149).

Both figures plot point estimates; this script computes the #140 paired
intervals their claims must rest on, from the per-source sidecars.

- **Figure 1** (D3, C3 V2+V6/M2 vs C10 V2+V8/M2): C10 − C3 over all 21 pairs,
  over the CMT/EDAS pairs the figure highlights and over the other pairs, the
  difference between those two subgroups, and each pair on its own.  The
  subgroup difference gets two intervals: resampling test sources within each
  pair (as every #140 interval does) and resampling whole pairs within each
  subgroup, which also reflects how much pairs differ from one another.
- **Figure 2** (D1/D2, V2+V8 per model, plus C17 V1/M4): every pair of plotted
  conditions on each dataset, B − A with A the earlier bar.  These are
  exploratory: the figure's own family per dataset gets Holm-adjusted MRR
  p-values, separate from the registered comparisons in ``report.md``.

Every interval is a 95% percentile bootstrap and every p-value a paired
randomisation test, with ``paired_tests.N_REPLICATES`` replicates and seed
``paired_tests.SEED``.  As in ``paired_tests.run_paired_tests``, a comparison
whose sidecars are missing, stale or describe different references, or a
Figure 1 subgroup with no pair, is recorded as not tested with its reason
instead of stopping the run.  Figure 2 tests every bar expected on a dataset
(all but those in ``DatasetSpec.not_run``), so a missing result is a not-tested
comparison and withholds the dataset's Holm p-values rather than shrinking its
family.  A Figure 2 result whose strategy or model is not its bar's is
refused, as the figure itself refuses it.  Writes
``figure_claims.json`` and ``figure_claims.md``.

Usage
-----
python scripts/figure_claims.py --results_dir data/results/ \\
    --output_dir artifacts/issue-149/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgsemembed.evaluation.paired_tests import (  # noqa: E402
    DECISION_METRIC,
    METRICS,
    UNTESTABLE,
    PairArrays,
    SidecarCache,
    TestSettings,
    aligned_arrays,
    bootstrap_deltas,
    check_same_references,
    holm_adjust,
    observed_values,
    paired_test,
    percentile_interval,
)
from kgsemembed.utils.errors import DataError  # noqa: E402
from scripts.figure_specs import (  # noqa: E402
    BARS,
    DATASETS,
    SPARSE_PAIRS,
    BarSpec,
    DatasetSpec,
    check_condition,
)

FIG1_A, FIG1_B, FIG1_DATASET = "C3", "C10", "D3"
SUBGROUPS = ("all", "cmt_edas", "other")
_METRIC_NAMES = {"mrr": "MRR", "hits_at_1": "Hits@1", "f1": "F1"}

MetricArrays = Dict[str, Dict[str, PairArrays]]


def column_arrays(cache: SidecarCache, comparison: dict) -> MetricArrays:
    """
    Check two conditions share their references once, then align every metric.

    Parameters
    ----------
    cache : SidecarCache
        Sidecar cache of the results directory.
    comparison : dict
        ``condition_a``, ``condition_b`` and ``dataset_id``.

    Returns
    -------
    MetricArrays
        Metric mapped to pair name mapped to its aligned ``(a, b)`` arrays.

    Raises
    ------
    DataError
        If the two conditions' sidecars describe different references or setups.
    """
    condition_a, condition_b, dataset_id = (comparison[k] for k in ("condition_a", "condition_b", "dataset_id"))
    check_same_references(cache.sidecars(condition_a, dataset_id),
                          cache.sidecars(condition_b, dataset_id))
    out = {}
    for metric in METRICS:
        values_a = cache.values(condition_a, dataset_id, metric)
        values_b = cache.values(condition_b, dataset_id, metric)
        out[metric] = {name: aligned_arrays(values_a[name], values_b[name], metric) for name in sorted(values_a)}
    return out


def subgroup_pairs(arrays: Dict[str, PairArrays], subgroup: str) -> Dict[str, PairArrays]:
    """
    Return the pairs of a Figure 1 subgroup: ``all``, ``cmt_edas`` or ``other``.

    Parameters
    ----------
    arrays : Dict[str, PairArrays]
        Every D3 pair's aligned arrays.
    subgroup : str
        One of :data:`SUBGROUPS`.

    Returns
    -------
    Dict[str, PairArrays]
        The subgroup's pairs.

    Raises
    ------
    DataError
        If the subgroup has no pair, e.g. on a partial D3 run.
    """
    keep = subgroup == "cmt_edas"
    pairs = {name: pair for name, pair in arrays.items()
             if subgroup == "all" or (name in SPARSE_PAIRS) == keep}
    if not pairs:
        raise DataError(f"Figure 1 subgroup {subgroup!r} has no D3 pair with results")
    return pairs


def pair_delta(pair: PairArrays, metric: str) -> float:
    """
    Return one pair's observed B − A value.

    Parameters
    ----------
    pair : PairArrays
        The pair's aligned ``(a, b)`` unit arrays.
    metric : str
        One of ``paired_tests.METRICS``.

    Returns
    -------
    float
        The pair's metric under B minus that under A.
    """
    value_a, value_b = observed_values([pair], metric)
    return value_b - value_a


def pair_level_interval(
    deltas_x: List[float], deltas_y: List[float], settings: TestSettings
) -> Optional[tuple[float, float]]:
    """
    Interval of the difference of two subgroup means, resampling whole pairs.

    Parameters
    ----------
    deltas_x, deltas_y : List[float]
        Observed per-pair deltas of the two subgroups.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    Optional[tuple[float, float]]
        Percentile interval of ``mean(x*) − mean(y*)``, or ``None`` when a
        subgroup has fewer than two pairs: resampling one pair has no
        between-pair spread, so its interval would be falsely narrow.
    """
    if min(len(deltas_x), len(deltas_y)) < 2:
        return None
    rng = np.random.default_rng(settings.seed)
    x, y = np.asarray(deltas_x), np.asarray(deltas_y)
    means_x = x[rng.integers(0, len(x), (settings.n_replicates, len(x)))].mean(axis=1)
    means_y = y[rng.integers(0, len(y), (settings.n_replicates, len(y)))].mean(axis=1)
    return percentile_interval(means_x - means_y)


def subgroup_difference(
    arrays: Dict[str, PairArrays], metric: str, settings: TestSettings
) -> dict:
    """
    Compare the CMT/EDAS subgroup's mean delta with the other pairs'.

    Parameters
    ----------
    arrays : Dict[str, PairArrays]
        Every D3 pair's aligned arrays.
    metric : str
        One of ``paired_tests.METRICS``.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    dict
        ``delta`` (CMT/EDAS mean delta minus the others'), ``ci_sources``
        (test sources resampled within pairs; the other pairs use seed + 1,
        so the two subgroups' replicates are independent) and ``ci_pairs``
        (pairs resampled within each subgroup; ``None`` below two pairs).
    """
    sparse = list(subgroup_pairs(arrays, "cmt_edas").values())
    other = list(subgroup_pairs(arrays, "other").values())
    deltas_s = [pair_delta(pair, metric) for pair in sparse]
    deltas_o = [pair_delta(pair, metric) for pair in other]
    other_settings = replace(settings, seed=settings.seed + 1)
    within = bootstrap_deltas(sparse, metric, settings) - bootstrap_deltas(other, metric, other_settings)
    return {"delta": float(np.mean(deltas_s) - np.mean(deltas_o)),
            "ci_sources": percentile_interval(within),
            "ci_pairs": pair_level_interval(deltas_s, deltas_o, settings)}


def _direction(row: dict) -> str:
    """Name the condition an interval favours, or ``none`` when it covers 0."""
    if row["ci_low"] > 0:
        return "B"
    return "A" if row["ci_high"] < 0 else "none"


def per_pair_rows(arrays: Dict[str, PairArrays], metric: str, settings: TestSettings) -> List[dict]:
    """Test every D3 pair on its own and name the condition its interval favours."""
    rows = []
    for name, pair in arrays.items():
        row = {"pair": name, "cmt_edas": name in SPARSE_PAIRS, **paired_test([pair], metric, settings)}
        rows.append({**row, "favours": _direction(row)})
    return rows


def figure1_block(arrays: Dict[str, PairArrays], metric: str, settings: TestSettings) -> dict:
    """
    Compute one metric's Figure 1 subgroups, subgroup difference and pairs.

    Parameters
    ----------
    arrays : Dict[str, PairArrays]
        Every D3 pair's aligned arrays for this metric.
    metric : str
        One of ``paired_tests.METRICS``.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    dict
        ``tested`` (true), ``subgroups``, ``difference`` and ``pairs``.
    """
    return {
        "tested": True,
        "subgroups": {s: paired_test(list(subgroup_pairs(arrays, s).values()), metric, settings)
                      for s in SUBGROUPS},
        "difference": subgroup_difference(arrays, metric, settings),
        "pairs": per_pair_rows(arrays, metric, settings),
    }


def figure1(cache: SidecarCache, settings: TestSettings) -> Dict[str, dict]:
    """
    Compute every Figure 1 interval for every metric.

    Parameters
    ----------
    cache : SidecarCache
        Sidecar cache of the results directory.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    Dict[str, dict]
        Metric mapped to ``subgroups`` (C10 − C3 over all, CMT/EDAS and other
        pairs), ``difference`` (:func:`subgroup_difference`) and ``pairs``
        (:func:`per_pair_rows`), with ``tested`` true; or, when the sidecars
        cannot be compared or a subgroup is empty, ``tested`` false and the
        ``reason``.  Only loading and checking the sidecars is caught: an
        error while computing the intervals propagates.
    """
    comparison = {"condition_a": FIG1_A, "condition_b": FIG1_B, "dataset_id": FIG1_DATASET}
    try:
        arrays = column_arrays(cache, comparison)
        for subgroup in SUBGROUPS:
            subgroup_pairs(arrays[DECISION_METRIC], subgroup)
    except UNTESTABLE as exc:
        return {metric: {"tested": False, "reason": str(exc)} for metric in METRICS}
    return {metric: figure1_block(arrays[metric], metric, settings) for metric in METRICS}


def _check_bar_result(results_dir: Path, bar: BarSpec, dataset: DatasetSpec) -> None:
    """Refuse the result the figure plots for a bar if it is mislabelled; a missing one passes."""
    path = results_dir / bar.condition_id / dataset.dataset_id / f"{dataset.pair_name}_results.json"
    if path.exists():
        check_condition(json.loads(path.read_text()), bar)


def figure2_conditions(results_dir: Path, dataset: DatasetSpec) -> List[str]:
    """
    Return the Figure 2 conditions expected on a dataset, in bar order.

    Parameters
    ----------
    results_dir : Path
        Root of the result tree.
    dataset : DatasetSpec
        ``D1`` or ``D2``, with the conditions never run on it.

    Returns
    -------
    List[str]
        Condition IDs of every bar not in ``dataset.not_run``, whether or not
        its result exists: a missing one makes its comparisons not tested.

    Raises
    ------
    ValueError
        If the result the figure plots for a bar has another strategy or model.
    """
    expected = [bar for bar in BARS if bar.condition_id not in dataset.not_run]
    for bar in expected:
        _check_bar_result(results_dir, bar, dataset)
    return [bar.condition_id for bar in expected]


def comparison_rows(cache: SidecarCache, comparison: dict, settings: TestSettings) -> List[dict]:
    """
    Test one Figure 2 comparison on one dataset for every metric.

    Parameters
    ----------
    cache : SidecarCache
        Sidecar cache of the results directory.
    comparison : dict
        ``dataset_id``, ``condition_a`` and ``condition_b``.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    List[dict]
        One row per metric; ``tested`` is false with a ``reason`` when the
        sidecars are missing, stale, malformed or describe different references.
    """
    try:
        arrays = column_arrays(cache, comparison)
    except UNTESTABLE as exc:
        return [{**comparison, "metric": m, "tested": False, "reason": str(exc)} for m in METRICS]
    return [{**comparison, "metric": m, "tested": True,
             **paired_test(list(arrays[m].values()), m, settings)} for m in METRICS]


def add_holm(rows: List[dict]) -> List[dict]:
    """
    Add Holm-adjusted p-values to the decision-metric rows of one dataset.

    The family is every decision-metric comparison of the dataset.  If any of
    them was not tested, the family is incomplete and no row is adjusted:
    adjusting over the tested ones alone would shrink the family and lower
    every ``p_holm``.

    Parameters
    ----------
    rows : List[dict]
        One dataset's rows from :func:`comparison_rows`.

    Returns
    -------
    List[dict]
        The same rows; tested decision-metric rows gain ``family_size`` and
        ``p_holm`` (``None`` when the family is incomplete).
    """
    decision = [row for row in rows if row["metric"] == DECISION_METRIC]
    complete = all(row["tested"] for row in decision)
    adjusted = holm_adjust([row["p_value"] for row in decision]) if complete else [None] * len(decision)
    for row, p_holm in zip(decision, adjusted):
        if row["tested"]:
            row.update(family_size=len(decision), p_holm=p_holm)
    return rows


def figure2(results_dir: Path, cache: SidecarCache, settings: TestSettings) -> List[dict]:
    """
    Compute every Figure 2 pairwise interval on each dataset and metric.

    Parameters
    ----------
    results_dir : Path
        Root of the result tree, to find which conditions ran on each dataset.
    cache : SidecarCache
        Sidecar cache of the same directory.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    List[dict]
        One row per dataset, comparison and metric; tested MRR rows carry
        ``p_holm`` (Holm within the dataset's Figure 2 comparisons; ``None``
        when one of them was not tested).

    Raises
    ------
    ValueError
        If a result's strategy or model is not its bar's.
    """
    rows = []
    for dataset in DATASETS:
        conditions = figure2_conditions(results_dir, dataset)
        comparisons = [{"dataset_id": dataset.dataset_id, "condition_a": a, "condition_b": b}
                       for a, b in combinations(conditions, 2)]
        dataset_rows = [row for c in comparisons for row in comparison_rows(cache, c, settings)]
        rows += add_holm(sorted(dataset_rows, key=lambda row: METRICS.index(row["metric"])))
    return rows


def _fmt(value: float) -> str:
    """Format a signed metric difference."""
    return f"{value:+.4f}"


def _ci(low: float, high: float) -> str:
    """Format an interval."""
    return f"[{low:+.4f}, {high:+.4f}]"


def _table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> List[str]:
    """Render a Markdown table."""
    header = list(header)
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _cell_text(text: str) -> str:
    """Make free text safe for one Markdown table cell: no pipes, no line breaks."""
    return " ".join(text.split()).replace("|", "\\|")


def _untested_cells(reason: str, width: int) -> List[str]:
    """Render ``width`` cells for a comparison that was not tested: its reason, then blanks."""
    return [f"not tested: {_cell_text(reason)}"] + [""] * (width - 1)


def _pairs_ci(interval: Optional[list]) -> str:
    """Format the pair-level interval, or say why there is none."""
    return "n/a (a subgroup has fewer than 2 pairs)" if interval is None else _ci(*interval)


def figure1_markdown(fig1: Dict[str, dict]) -> List[str]:
    """Render the Figure 1 subgroup and subgroup-difference table."""
    rows = []
    for metric, block in fig1.items():
        if not block["tested"]:
            rows.append([_METRIC_NAMES[metric]] + _untested_cells(block["reason"], 7))
            continue
        for subgroup in SUBGROUPS:
            r = block["subgroups"][subgroup]
            rows.append([_METRIC_NAMES[metric], subgroup, str(r["n_pairs"]), f"{r['value_a']:.4f}",
                         f"{r['value_b']:.4f}", _fmt(r["delta"]), _ci(r["ci_low"], r["ci_high"]),
                         f"{r['p_value']:.4f}"])
        d = block["difference"]
        rows.append([_METRIC_NAMES[metric], "cmt_edas − other", "", "", "", _fmt(d["delta"]),
                     f"sources {_ci(*d['ci_sources'])}; pairs {_pairs_ci(d['ci_pairs'])}", ""])
    return _table(["Metric", "Pairs", "n", "C3", "C10", "Δ (C10 − C3)", "95% CI", "p"], rows)


def pair_count_markdown(fig1: Dict[str, dict]) -> List[str]:
    """Render, per metric, how many pairs' own intervals favour each condition."""
    rows = []
    for metric, block in ((m, b) for m, b in fig1.items() if b["tested"]):
        for subgroup in ("cmt_edas", "other"):
            pairs = [p for p in block["pairs"] if p["cmt_edas"] == (subgroup == "cmt_edas")]
            count = [p["favours"] for p in pairs].count
            rows.append([_METRIC_NAMES[metric], subgroup, str(len(pairs)),
                         str(sum(p["delta"] > 0 for p in pairs)), str(count("B")), str(count("A"))])
    return _table(["Metric", "Pairs", "n", "C10 ahead (point)", "CI favours C10",
                   "CI favours C3"], rows)


def _holm_cell(row: dict) -> str:
    """Format a Holm p-value, a dash for metrics without a decision, or why it is missing."""
    if "p_holm" not in row:
        return "—"
    return "n/a (family incomplete)" if row["p_holm"] is None else f"{row['p_holm']:.4f}"


def _figure2_cells(r: dict) -> List[str]:
    """Render one Figure 2 row, or its reason when it was not tested."""
    keys = [r["dataset_id"], _METRIC_NAMES[r["metric"]], f"{r['condition_a']} vs {r['condition_b']}"]
    if not r["tested"]:
        return keys + _untested_cells(r["reason"], 6)
    return keys + [f"{r['value_a']:.4f}", f"{r['value_b']:.4f}", _fmt(r["delta"]),
                   _ci(r["ci_low"], r["ci_high"]), f"{r['p_value']:.4f}", _holm_cell(r)]


def figure2_markdown(rows: List[dict]) -> List[str]:
    """Render the Figure 2 pairwise table."""
    cells = [_figure2_cells(r) for r in rows]
    return _table(["Dataset", "Metric", "A vs B", "A", "B", "Δ (B − A)", "95% CI", "p",
                   "p (Holm)"], cells)


def render_markdown(fig1: Dict[str, dict], fig2: List[dict]) -> str:
    """
    Render both figures' intervals as a Markdown report.

    Parameters
    ----------
    fig1 : Dict[str, dict]
        Output of :func:`figure1`.
    fig2 : List[dict]
        Output of :func:`figure2`.

    Returns
    -------
    str
        The report.
    """
    lines = ["# Figure claims: paired intervals (#149)", "",
             "_Generated by `scripts/figure_claims.py`; 95% percentile bootstrap, paired "
             "randomisation p, 10,000 replicates, seed 42 (#140 settings)._", "",
             "## Figure 1: D3, C10 (V2+V8/M2) − C3 (V2+V6/M2)", ""]
    lines += figure1_markdown(fig1) + ["", "Pairs whose own interval excludes 0:", ""]
    lines += pair_count_markdown(fig1)
    lines += ["", "## Figure 2: D1/D2, pairwise (exploratory; Holm within dataset, MRR only)", ""]
    return "\n".join(lines + figure2_markdown(fig2)) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Paired intervals behind Figures 1 and 2 (#149).")
    parser.add_argument("--results_dir", type=Path, default=Path("data/results/"),
                        help="Root of the result tree, with per-source sidecars.")
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/issue-149/"),
                        help="Directory for figure_claims.json and figure_claims.md.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Compute both figures' intervals and write them as JSON and Markdown."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args(argv)
    cache, settings = SidecarCache(args.results_dir), TestSettings()
    fig1, fig2 = figure1(cache, settings), figure2(args.results_dir, cache, settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"figure1": fig1, "figure2": fig2}
    (args.output_dir / "figure_claims.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = render_markdown(fig1, fig2)
    (args.output_dir / "figure_claims.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
