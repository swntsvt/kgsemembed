"""Markdown report of the controlled PPAS ablation (#122).

Reads the result JSON files and ``_entities.json`` sidecars written by
``python -m kgsemembed.pipeline.ppas_ablation`` and reports, per dataset pair
and token budget:

* representation coverage and cost: sampled entities, token usage, model
  truncation, selection and encoding time;
* effectiveness of the full reference, PPAS, deterministic selection, and the
  random policy as a mean, standard deviation and 95% t-interval over seeds;
* how many sampled selections PPAS and deterministic selection share;
* paired differences in MRR and Recall@1 over test sources with 95% paired
  bootstrap intervals, overall and split by whether sampling was invoked for
  the source or its gold target;
* MRR by source degree (eligible-item count).

The bootstrap resamples test sources of a single alignment pair, so its
intervals describe uncertainty conditional on that dataset and candidate set,
not generalisation to other datasets.  No significance test is run.

Usage
-----
python scripts/generate_ppas_ablation_report.py \\
    --results_dir data/results_ppas_ablation/ \\
    --output data/results_ppas_ablation/report.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from kgsemembed.verbalisation.selection import DETERMINISTIC, FULL, PPAS, RANDOM

_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 42
_CONFIDENCE = 0.95
_METRICS = ("f1", "precision", "recall", "threshold", "mrr", "recall_at_1", "recall_at_5", "recall_at_10")
_DEGREE_BINS = ((0, 0), (1, 5), (6, 10), (11, 20), (21, 10**9))
_ALL, _SAMPLED, _UNSAMPLED = "all test sources", "sampling invoked", "sampling not invoked"

Run = Dict[str, object]


def load_runs(results_dir: Path) -> Dict[Tuple[str, str], List[Run]]:
    """
    Load every ablation result with its sidecar, grouped by dataset pair.

    Parameters
    ----------
    results_dir : Path
        Root written by the ablation runner.

    Returns
    -------
    Dict[Tuple[str, str], List[Run]]
        ``(dataset_id, pair_name)`` to result payloads, each carrying its
        sidecar under ``"_entities"``.
    """
    grouped: Dict[Tuple[str, str], List[Run]] = defaultdict(list)
    for path in sorted(results_dir.glob("*/*/*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sidecar = path.with_name(path.name.replace("_results.json", "_entities.json"))
        payload["_entities"] = json.loads(sidecar.read_text(encoding="utf-8"))
        grouped[(payload["dataset_id"], payload["pair_name"])].append(payload)
    return dict(grouped)


def _policy(run: Run) -> str:
    return run["ablation"]["policy"]


def _families(runs: List[Run]) -> Dict[int, List[Run]]:
    """Group capped runs by budget, each family led by the full reference."""
    full = [r for r in runs if _policy(r) == FULL]
    families: Dict[int, List[Run]] = defaultdict(list)
    for run in runs:
        if run["token_budget"] is not None:
            families[run["token_budget"]].append(run)
    return {budget: full + members for budget, members in sorted(families.items(), reverse=True)}


def _by_policy(family: List[Run], policy: str) -> List[Run]:
    return sorted((r for r in family if _policy(r) == policy), key=lambda r: r["ablation"]["seed"] or 0)


def t_interval(values: List[float]) -> Tuple[float, float, float, float]:
    """
    Return the mean, sample SD and 95% t-interval of seed-level values.

    Parameters
    ----------
    values : List[float]
        One value per random seed.

    Returns
    -------
    Tuple[float, float, float, float]
        ``(mean, sd, low, high)``; SD and interval are NaN for one value.
    """
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sd = float(array.std(ddof=1))
    half = stats.t.ppf((1 + _CONFIDENCE) / 2, array.size - 1) * sd / np.sqrt(array.size)
    return mean, sd, mean - half, mean + half


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _metric_row(label: str, run: Run) -> List[str]:
    return [label] + [_fmt(run["metrics"][m]) for m in _METRICS]


def _random_rows(randoms: List[Run]) -> List[List[str]]:
    """Summarise the random policy as mean ± SD and its t-interval per metric."""
    summaries = [t_interval([r["metrics"][m] for r in randoms]) for m in _METRICS]
    seeds = ",".join(str(r["ablation"]["seed"]) for r in randoms)
    return [
        [f"Random mean ± SD (seeds {seeds})"] + [f"{_fmt(s[0])} ± {_fmt(s[1])}" for s in summaries],
        ["Random 95% t-interval"] + [f"[{_fmt(s[2])}, {_fmt(s[3])}]" for s in summaries],
    ]


def effectiveness_section(family: List[Run]) -> str:
    """Tabulate threshold-dependent and ranking metrics for one budget."""
    rows = [_metric_row(label, run) for policy, label in
            ((FULL, "Full (uncapped reference)"), (PPAS, "PPAS"), (DETERMINISTIC, "Deterministic"))
            for run in _by_policy(family, policy)]
    randoms = _by_policy(family, RANDOM)
    rows += _random_rows(randoms) if randoms else []
    rows += [_metric_row(f"Random seed {r['ablation']['seed']}", r) for r in randoms]
    return _table(["Policy", *_METRICS], rows)


def _side_cells(run: Run, side: str) -> List[str]:
    summary = run["ablation"][side]
    n = max(summary["n_entities"], 1)
    return [
        f"{summary['n_sampling_invoked']} ({summary['n_sampling_invoked'] / n:.1%})",
        _fmt(summary["tokens"]["mean"], 1),
        str(summary["n_model_truncated"]),
    ]


def _runtime_cells(run: Run) -> List[str]:
    runtime = run["ablation"]["runtime"]
    selection = sum(runtime[s]["selection_seconds"] for s in runtime)
    encode = sum(runtime[s]["encode_seconds"] for s in runtime)
    encoded = sum(runtime[s]["n_texts_encoded"] for s in runtime)
    return [_fmt(selection, 1), str(encoded), _fmt(encode, 1)]


def coverage_section(family: List[Run]) -> str:
    """Tabulate sampling coverage, token usage, truncation and runtime per run."""
    rows = []
    for run in family:
        tokens = sum(run["ablation"][s]["model_input_tokens_total"] for s in ("source", "candidate"))
        oversized = sum(run["ablation"][s]["n_oversized_items"] for s in ("source", "candidate"))
        rows.append([run["condition_id"], *_side_cells(run, "source"), *_side_cells(run, "candidate"),
                     str(tokens), str(oversized), *_runtime_cells(run)])
    header = ["Condition", "Src sampled", "Src mean tok", "Src truncated", "Cand sampled",
              "Cand mean tok", "Cand truncated", "Model input tokens", "Oversized items",
              "Selection s", "New texts encoded", "Encode s"]
    return _table(header, rows)


def identity_section(family: List[Run]) -> str:
    """Count sampled entities whose PPAS and deterministic selections differ."""
    ppas, det = _by_policy(family, PPAS), _by_policy(family, DETERMINISTIC)
    if not ppas or not det:
        return "_PPAS or deterministic run missing._"
    rows = []
    for side in ("source", "candidate"):
        first = ppas[0]["_entities"]["sampled_fingerprints"][side]
        second = det[0]["_entities"]["sampled_fingerprints"][side]
        differ = sum(first[uri] != second.get(uri) for uri in first)
        rows.append([side, str(len(first)), str(differ), f"{differ / max(len(first), 1):.1%}"])
    return _table(["Side", "Sampled entities", "PPAS ≠ deterministic", "Share"], rows)


def _per_source(run: Run) -> Dict[str, Tuple[float, float]]:
    """Map each test source to its reciprocal rank and hit@1."""
    out = {}
    for record in run["_entities"]["test_sources"]:
        rank = record["best_rank"]
        out[record["source"]] = (1.0 / rank if rank else 0.0, float(rank == 1))
    return out


def _random_mean_per_source(randoms: List[Run]) -> Dict[str, Tuple[float, float]]:
    """Average each test source's RR and hit@1 over the random seeds."""
    per_seed = [_per_source(run) for run in randoms]
    return {s: tuple(np.mean([seed[s] for seed in per_seed], axis=0)) for s in per_seed[0]}


def _strata(family: List[Run]) -> Dict[str, set]:
    """Split test sources by whether sampling was invoked for them or their gold target."""
    capped = next(r for r in family if r["token_budget"] is not None)
    records = capped["_entities"]["test_sources"]
    sampled = {r["source"] for r in records if r["source_sampled"] or r["target_sampled"]}
    every = {r["source"] for r in records}
    return {_ALL: every, _SAMPLED: sampled, _UNSAMPLED: every - sampled}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Return the mean paired difference ``a - b`` and its 95% bootstrap interval.

    Parameters
    ----------
    a, b : np.ndarray
        Per-source values of two conditions, aligned by source.
    rng : np.random.Generator
        Seeded generator for resampling.

    Returns
    -------
    Tuple[float, float, float]
        ``(difference, low, high)``; NaN bounds for an empty input.
    """
    diff = a - b
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    indices = rng.integers(0, diff.size, size=(_BOOTSTRAP_RESAMPLES, diff.size))
    means = diff[indices].mean(axis=1)
    alpha = (1 - _CONFIDENCE) / 2
    return float(diff.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _contrasts(family: List[Run]) -> List[Tuple[str, Dict, Dict]]:
    """Return the pre-specified contrasts available in a family."""
    series = {label: _per_source(runs[0]) for label, runs in
              (("PPAS", _by_policy(family, PPAS)), ("Deterministic", _by_policy(family, DETERMINISTIC)),
               ("Full", _by_policy(family, FULL))) if runs}
    randoms = _by_policy(family, RANDOM)
    if randoms:
        series["Random (seed mean)"] = _random_mean_per_source(randoms)
    wanted = (("PPAS", "Deterministic"), ("PPAS", "Random (seed mean)"),
              ("Deterministic", "Random (seed mean)"), ("PPAS", "Full"))
    return [(f"{a} − {b}", series[a], series[b]) for a, b in wanted if a in series and b in series]


def _contrast_row(name: str, a: Dict, b: Dict, sources: set, rng: np.random.Generator) -> List[str]:
    ordered = sorted(sources)
    cells = [name, str(len(ordered))]
    for column in (0, 1):
        va = np.array([a[s][column] for s in ordered])
        vb = np.array([b[s][column] for s in ordered])
        diff, low, high = paired_bootstrap(va, vb, rng)
        cells.append(f"{_fmt(diff)} [{_fmt(low)}, {_fmt(high)}]")
    return cells


def contrast_section(family: List[Run]) -> str:
    """Tabulate paired MRR and Recall@1 differences by sampling stratum."""
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    rows = []
    for stratum, sources in _strata(family).items():
        for name, a, b in _contrasts(family):
            rows.append([stratum, *_contrast_row(name, a, b, sources, rng)])
    return _table(["Stratum", "Contrast", "Test sources", "Δ MRR [95% CI]", "Δ R@1 [95% CI]"], rows)


def degree_section(family: List[Run]) -> str:
    """Tabulate MRR by source eligible-item count for each policy."""
    labelled = [(f"{_policy(r)}" + (f" s{r['ablation']['seed']}" if r["ablation"]["seed"] is not None else ""), r)
                for r in family if _policy(r) != RANDOM]
    randoms = _by_policy(family, RANDOM)
    degree = {r["source"]: r["source_degree"] for r in family[0]["_entities"]["test_sources"]}
    series = [(label, _per_source(run)) for label, run in labelled]
    if randoms:
        series.append(("random (seed mean)", _random_mean_per_source(randoms)))
    rows = []
    for low, high in _DEGREE_BINS:
        sources = [s for s, d in degree.items() if low <= d <= high]
        if sources:
            label = f"{low}" if low == high else (f"{low}+" if high >= 10**9 else f"{low}–{high}")
            rows.append([label, str(len(sources))] +
                        [_fmt(float(np.mean([values[s][0] for s in sources]))) for _l, values in series])
    return _table(["Source degree", "Test sources", *[label for label, _ in series]], rows)


def _provenance_lines(runs: List[Run]) -> List[str]:
    first = runs[0]
    block = first["ablation"]
    return [
        f"- Model `{first['model_id']}` revision `{first['hf_revision']}`; tokenizer `{block['tokenizer']}`",
        f"- Candidates `{first['candidates_file']}` sha256 `{first['candidates_sha256'][:16]}…` "
        f"(identical across runs: {len({r['candidates_sha256'] for r in runs}) == 1})",
        f"- Git commits: {sorted({r['git_commit'][:10] for r in runs})}; dirty: {sorted({str(r['git_dirty']) for r in runs})}",
        f"- Representation: {block['representation']}",
        f"- Budget counting: {block['token_counting']}",
        f"- Deterministic order: {block['deterministic_order']}; PPAS tier config sha256 `{block['tier_config_sha256'][:16]}…`",
        f"- Threshold: {block['threshold_procedure']}",
        f"- Test references: {first['population']['n_test_refs']}; validation references: {first['population']['n_val_refs']}",
    ]


def pair_section(dataset_id: str, pair_name: str, runs: List[Run]) -> str:
    """Render every budget family of one dataset pair."""
    parts = [f"## {dataset_id} / {pair_name}", "\n".join(_provenance_lines(runs))]
    for budget, family in _families(runs).items():
        parts += [
            f"### Budget {budget} tokens",
            "#### Coverage and cost", coverage_section(family),
            "#### Effectiveness (test population)", effectiveness_section(family),
            "#### Sampled selections: PPAS vs deterministic", identity_section(family),
            "#### Paired differences over test sources (95% paired bootstrap)", contrast_section(family),
            "#### MRR by source degree (eligible items)", degree_section(family),
        ]
    return "\n\n".join(parts)


_CAVEATS = """## Reading this report

- Each dataset contributes one alignment pair, so there is no replication at
  the dataset level and no significance test is reported.  Bootstrap intervals
  resample test sources of that pair; sources share candidate targets, so the
  intervals are approximate and conditional on this dataset and candidate set.
- The random policy's t-interval is over five seeds and describes selection
  variability only.
- Entities whose full text fits the budget are byte-identical, and share one
  cached embedding, under every policy; differences arise only from sampled
  entities.  The "sampling not invoked" stratum can still move when a sampled
  candidate competes in an unsampled source's list.
- F1 depends on a threshold tuned on the validation population; MRR and
  Recall@k do not.  Read the ranking metrics first.
- The full reference differs from the capped runs in input length, and inputs
  beyond 512 tokens are truncated by the model ("truncated" columns)."""


def build_report(results_dir: Path) -> str:
    """
    Render the full ablation report.

    Parameters
    ----------
    results_dir : Path
        Root written by the ablation runner.

    Returns
    -------
    str
        Markdown report.
    """
    grouped = load_runs(results_dir)
    if not grouped:
        raise FileNotFoundError(f"No ablation results under {results_dir}.")
    sections = ["# Controlled PPAS ablation (#122)", _CAVEATS]
    sections += [pair_section(ds, pair, runs) for (ds, pair), runs in sorted(grouped.items(), reverse=True)]
    return "\n\n".join(sections) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report the controlled PPAS ablation.")
    parser.add_argument("--results_dir", default="data/results_ppas_ablation/", help="Ablation result root.")
    parser.add_argument("--output", default=None, help="Markdown output path; prints when omitted.")
    return parser


def main() -> None:
    """Build the report and write or print it."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    report = build_report(Path(args.results_dir))
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
