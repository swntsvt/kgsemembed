"""Markdown report of the controlled PPAS ablation (#122, #128).

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

Controlled-policy runs (#128, ``ablation.policy_label`` P1-P7) are reported in
their own section, keyed by label rather than by selection policy, so P1 and
P3 (both deterministic) or P2, P4, P5 and P7 (all PPAS) never collide.  Runs
are combined only when their run signature matches: model and tokenizer
revision, candidate-set hash, evaluation protocol, budget, test-source set and
smoke restriction.  Each signature group is reported separately, and labels a
group lacks are listed as missing.  The section reports matching quality,
representation cost, selection changes between policies, and paired
contrasts that separate filtering (P3 − P1), ordering (P4 − P3, P2 − P1,
P7 − P6) and vocabulary (P5 − P2, P6 − P3) effects.

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
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from kgsemembed.pipeline.ablation_conditions import ABLATION_CONDITIONS
from kgsemembed.verbalisation.selection import DETERMINISTIC, FULL, PPAS, RANDOM

_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 42
_CONFIDENCE = 0.95
_METRICS = ("f1", "precision", "recall", "threshold", "mrr", "recall_at_1", "recall_at_5", "recall_at_10")
_DEGREE_BINS = ((0, 0), (1, 5), (6, 10), (11, 20), (21, 10**9))
_ALL, _SAMPLED, _UNSAMPLED = "all test sources", "sampling invoked", "sampling not invoked"
_CHANGED = "selection differs for source or a gold target"
_UNCHANGED = "source and gold targets unchanged"
_CONTROLLED_LABELS = tuple(f"P{i}" for i in range(1, 8))
_CONTROLLED_CONTRASTS = (
    ("P3", "P1", "filtering, deterministic order"),
    ("P4", "P2", "filtering, PPAS order"),
    ("P4", "P3", "ordering, eligibility fixed (base)"),
    ("P2", "P1", "ordering, all predicates eligible"),
    ("P5", "P2", "vocabulary, PPAS order"),
    ("P6", "P3", "vocabulary, covered eligibility"),
    ("P7", "P6", "ordering, eligibility fixed ({extended})"),
    ("P7", "P5", "filtering, {extended} PPAS"),
)
"""(a, b, effect): a − b; ``{extended}`` names the group's extended vocabulary."""
_RANKING_METRICS = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "f1", "precision", "recall", "threshold")

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
    """Render the #122 budget families and the #128 controlled policies of one pair."""
    legacy = [r for r in runs if not is_controlled(r)]
    controlled = [r for r in runs if is_controlled(r)]
    parts = [f"## {dataset_id} / {pair_name}"]
    if legacy:
        parts += ["### Policy ablation (#122)", "\n".join(_provenance_lines(legacy))]
        parts += _legacy_family_parts(legacy)
    if controlled:
        parts += controlled_parts(controlled)
    return "\n\n".join(parts)


def _legacy_family_parts(runs: List[Run]) -> List[str]:
    """Render every #122 budget family of one pair."""
    parts: List[str] = []
    for budget, family in _families(runs).items():
        parts += [
            f"### Budget {budget} tokens",
            "#### Coverage and cost", coverage_section(family),
            "#### Effectiveness (test population)", effectiveness_section(family),
            "#### Sampled selections: PPAS vs deterministic", identity_section(family),
            "#### Paired differences over test sources (95% paired bootstrap)", contrast_section(family),
            "#### MRR by source degree (eligible items)", degree_section(family),
        ]
    return parts


def is_controlled(run: Run) -> bool:
    """Whether a run belongs to the #128 controlled-policy family."""
    return bool(run["ablation"].get("policy_label"))


def _label(run: Run) -> str:
    return run["ablation"]["policy_label"]


def run_signature(run: Run) -> Tuple:
    """
    Return the settings two runs must share to be compared.

    Parameters
    ----------
    run : Run
        Result payload with its sidecar.

    Returns
    -------
    Tuple
        Model and tokenizer revision, candidate hash, protocol, budget,
        smoke restriction, and a digest of the test-source set.
    """
    block = run["ablation"]
    sources = sorted(r["source"] for r in run["_entities"]["test_sources"])
    digest = hashlib.sha256(json.dumps(sources).encode("utf-8")).hexdigest()[:16]
    return (run["model_id"], run["hf_revision"], block["tokenizer_revision"], run["candidates_sha256"],
            run["evaluation_protocol"], run["token_budget"], block["smoke_max_sources"], digest)


def signature_groups(runs: List[Run]) -> List[Tuple[Tuple, Dict[str, Run]]]:
    """
    Partition controlled runs by signature, keyed by label within each group.

    Parameters
    ----------
    runs : List[Run]
        Controlled runs of one pair.

    Returns
    -------
    List[Tuple[Tuple, Dict[str, Run]]]
        ``(signature, {label: run})``, largest group first.

    Raises
    ------
    ValueError
        If one signature holds two runs with the same label.
    """
    groups: Dict[Tuple, Dict[str, Run]] = defaultdict(dict)
    for run in runs:
        group = groups[run_signature(run)]
        if _label(run) in group:
            raise ValueError(f"Two {_label(run)} runs share one signature: "
                             f"{group[_label(run)]['condition_id']}, {run['condition_id']}.")
        group[_label(run)] = run
    return sorted(groups.items(), key=lambda item: (-len(item[1]), str(item[0])))


def _signature_lines(signature: Tuple, group: Dict[str, Run]) -> List[str]:
    model, revision, tokenizer_rev, candidates, protocol, budget, smoke, sources = signature
    first = next(iter(group.values()))
    missing = [label for label in expected_labels(first["dataset_id"], budget) if label not in group]
    return [
        f"- Model `{model}` revision `{revision}`; tokenizer revision `{tokenizer_rev}`",
        f"- Candidates sha256 `{candidates[:16]}…`; evaluation protocol {protocol}; budget {budget}",
        f"- Population: {'SMOKE, ' + str(smoke) + ' sources' if smoke else 'full'}; "
        f"test-source set `{sources}`; test references {first['population']['n_test_refs']}",
        f"- Git commits: {sorted({r['git_commit'][:10] for r in group.values()})}; "
        f"dirty: {sorted({str(r['git_dirty']) for r in group.values()})}",
        f"- Missing policies: {', '.join(missing) if missing else 'none'}",
    ]


def _matrix_row(run: Run) -> List[str]:
    block = run["ablation"]
    vocabulary = block["predicate_config"]
    return [_label(run), f"`{run['condition_id']}`", run["dataset_id"], str(run["token_budget"]),
            block["filter_mode"], vocabulary["name"], f"`{vocabulary['config_hash'][:12]}`",
            block["order_mode"], block["status"]]


def matrix_section(group: Dict[str, Run]) -> str:
    """Tabulate each controlled run's dataset, budget, filter, vocabulary and order."""
    rows = [_matrix_row(group[label]) for label in _CONTROLLED_LABELS if label in group]
    header = ["Policy", "Condition", "Dataset", "Budget", "Filter", "Vocabulary", "Config hash",
              "Order", "Status"]
    return _table(header, rows)


def quality_section(group: Dict[str, Run]) -> str:
    """Tabulate matching quality of each controlled run."""
    rows = [[label] + [_fmt(group[label]["metrics"][m]) for m in _RANKING_METRICS]
            for label in _CONTROLLED_LABELS if label in group]
    return _table(["Policy", *_RANKING_METRICS], rows)


def _cost_row(label: str, run: Run, side: str) -> List[str]:
    summary = run["ablation"][side]
    budget = summary.get("budget", {})
    n = max(summary["n_entities"], 1)
    return [
        label, side, _fmt(summary["tokens"]["mean"], 1), _fmt(summary["tokens"]["median"], 0),
        str(summary["tokens"]["max"]), str(budget.get("n_over_budget", "n/a")),
        _fmt(budget.get("utilisation_mean"), 3), f"{summary['n_sampling_invoked'] / n:.1%}",
        str(summary["n_items_before_filter"]), str(summary["n_filtered_items"]),
        str(summary["eligible_items"]["total"]), str(summary["n_selected_items"]),
        str(summary["n_skipped_items"]), str(summary["n_tiered_items"]),
        f"{summary['vocabulary_item_coverage']:.1%}", _fmt(summary["predicate_coverage_mean"], 3),
    ]


def cost_section(group: Dict[str, Run]) -> str:
    """Tabulate token cost, budget use and item accounting per run and side."""
    rows = [_cost_row(label, group[label], side) for label in _CONTROLLED_LABELS if label in group
            for side in ("source", "candidate")]
    header = ["Policy", "Side", "Mean tok", "Median tok", "Max tok", "Over budget", "Utilisation",
              "Sampled", "Items", "Filtered", "Eligible", "Selected", "Skipped", "Tiered",
              "Vocab coverage", "Predicate coverage"]
    return _table(header, rows)


def _fingerprints(run: Run, side: str) -> Dict[str, str]:
    return run["_entities"]["selection_fingerprints"][side]


def changed_entities(a: Run, b: Run, side: str) -> set:
    """
    Return entities of a side whose selected item set differs between two runs.

    Parameters
    ----------
    a, b : Run
        Runs of one signature group.
    side : str
        ``"source"`` or ``"candidate"``.

    Returns
    -------
    set
        URIs with different selection fingerprints.
    """
    first, second = _fingerprints(a, side), _fingerprints(b, side)
    return {uri for uri in first if first[uri] != second.get(uri)}


def _extended_vocabulary(group: Dict[str, Run]) -> str:
    """Name the non-base vocabulary of a group, or a placeholder when it has none."""
    names = {r["ablation"]["predicate_config"]["name"] for r in group.values()} - {"base"}
    return ", ".join(sorted(names)) or "extended"


def _available_contrasts(group: Dict[str, Run]) -> List[Tuple[str, str, str]]:
    extended = _extended_vocabulary(group)
    return [(a, b, effect.format(extended=extended))
            for a, b, effect in _CONTROLLED_CONTRASTS if a in group and b in group]


def expected_labels(dataset_id: str, budget: Optional[int]) -> List[str]:
    """
    Return the controlled labels registered for a dataset at a budget.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.
    budget : Optional[int]
        Token budget.

    Returns
    -------
    List[str]
        Registered policy labels, in label order.
    """
    return sorted({c.policy_label for c in ABLATION_CONDITIONS
                   if c.is_controlled and dataset_id in c.datasets and c.token_budget == budget})


def change_section(group: Dict[str, Run]) -> str:
    """Count entities whose selected set changes for each contrast."""
    rows = []
    for a, b, effect in _available_contrasts(group):
        cells = [f"{a} vs {b}", effect]
        for side in ("source", "candidate"):
            n = len(_fingerprints(group[a], side))
            changed = len(changed_entities(group[a], group[b], side))
            cells.append(f"{changed} / {n} ({changed / max(n, 1):.1%})")
        rows.append(cells + ["0 (canonical rendering)"])
    return _table(["Contrast", "Effect", "Sources changed", "Candidates changed",
                   "Same set, order changed"], rows)


def _changed_sources(a: Run, b: Run) -> set:
    """Test sources whose own selection or any gold target's selection differs."""
    sources = changed_entities(a, b, "source")
    targets = changed_entities(a, b, "candidate")
    return {r["source"] for r in a["_entities"]["test_sources"]
            if r["source"] in sources or any(t in targets for t in r.get("gold_targets", []))}


def controlled_contrast_section(group: Dict[str, Run]) -> str:
    """Tabulate paired MRR and Recall@1 differences with 95% bootstrap intervals."""
    rows = []
    for a, b, effect in _available_contrasts(group):
        every = set(_per_source(group[a]))
        changed = _changed_sources(group[a], group[b])
        for stratum, sources in ((_ALL, every), (_CHANGED, changed), (_UNCHANGED, every - changed)):
            rng = np.random.default_rng(_BOOTSTRAP_SEED)
            name = f"{a} − {b}"
            row = _contrast_row(name, _per_source(group[a]), _per_source(group[b]), sources, rng)
            rows.append([row[0], effect, stratum, *row[1:]])
    return _table(["Contrast", "Effect", "Test sources", "n", "Δ MRR [95% CI]", "Δ R@1 [95% CI]"], rows)


def controlled_parts(runs: List[Run]) -> List[str]:
    """Render every signature group of the controlled policies of one pair."""
    parts = ["### Controlled policies (#128)"]
    for index, (signature, group) in enumerate(signature_groups(runs), start=1):
        parts += [
            f"#### Run group {index}", "\n".join(_signature_lines(signature, group)),
            "##### Execution matrix", matrix_section(group),
            "##### Matching quality (test population)", quality_section(group),
            "##### Representation cost and item accounting", cost_section(group),
            "##### Selection changes between policies", change_section(group),
            "##### Paired differences over test sources (95% paired bootstrap, unit = test source)",
            controlled_contrast_section(group),
        ]
    return parts


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


_CONTROLLED_CAVEATS = """## Reading the controlled-policy section (#128)

- Selected items are always rendered in canonical order, so two policies that
  select the same item set produce byte-identical text.  An ordering policy
  can therefore change *which* items are kept but never their presentation
  order: "same set, different order" is zero by construction, and the
  selection-change counts capture the whole effect of a policy.
- Bootstrap intervals resample the test sources of one pair (the unit of
  resampling) with their paired per-source differences, 2,000 resamples, one
  fixed-seed generator per contrast.  They quantify uncertainty across source
  entities within that one evaluation, not generalisation to other KG pairs.
- "Source and gold targets unchanged" is not an unaffected stratum: a
  source's rank still moves when a policy changes the text of competing
  (non-gold) candidates in its list.  Read it as the distractor effect.
- Runs are combined only within one run signature; a smoke run is never
  combined with a full-population run."""


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
    sections = ["# Controlled PPAS ablation (#122, #128)", _CAVEATS, _CONTROLLED_CAVEATS]
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
