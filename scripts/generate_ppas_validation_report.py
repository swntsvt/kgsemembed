"""Validation report of the predicate-eligibility effect (#128 follow-up).

Combines the #128 D1 reference at 48 tokens (read-only, from
``data/results_ppas_controlled/``) with the validation runs in
``data/results_ppas_validation/``: the base-vocabulary policies P1-P4 on D1 at
32 and 64 tokens, and P1-P7 on D2 at 48 tokens (P5-P7 with the D2 vocabulary
``obo_extended``).

Two analyses are kept apart and never pooled:

* **D1 budget sensitivity** -- P1-P4 at 32, 48 and 64 tokens.  Each budget is
  its own run signature; rows for different budgets sit side by side, but
  every bootstrap resamples the test sources of one budget only.  The budgets
  share test sources, so differences between budgets are descriptive.
* **D2 replication** -- P1-P7 at 48 tokens on one other pair.

The primary contrasts are covered vs all eligibility under deterministic
(P3 − P1) and PPAS order (P4 − P2), and PPAS vs deterministic order under all
(P2 − P1) and covered eligibility (P4 − P3).  Each is reported over all test
sources, over those whose own or a gold target's selection changes, and over
the complement, which moves only through competing candidates.  A contrast
whose two policies select identical texts on both sides is reported as not
identifiable rather than as a zero effect.

Usage
-----
python scripts/generate_ppas_validation_report.py \\
    --reference_dir data/results_ppas_controlled/ \\
    --results_dir data/results_ppas_validation/ \\
    --output data/results_ppas_validation/report.md

# Post-#137 D1 re-run (#164)
python scripts/generate_ppas_validation_report.py \\
    --reference_dir data/results_issue164/ppas_controlled/ \\
    --results_dir data/results_issue164/ppas_validation/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_ppas_ablation_report as base_report  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import CONTROLLED_BUDGET, VALIDATION_BUDGETS  # noqa: E402
from kgsemembed.pipeline.ppas_ablation import dataset_file_digests  # noqa: E402

Run = Dict[str, object]
Group = Dict[str, Run]

D1_BUDGETS = tuple(sorted((*VALIDATION_BUDGETS, CONTROLLED_BUDGET)))
BASE_LABELS = ("P1", "P2", "P3", "P4")
PRIMARY_CONTRASTS = (
    ("P3", "P1", "covered vs all, deterministic order"),
    ("P4", "P2", "covered vs all, PPAS order"),
    ("P2", "P1", "PPAS vs deterministic, all eligible"),
    ("P4", "P3", "PPAS vs deterministic, covered eligible"),
)
VOCABULARY_CONTRASTS = (
    ("P5", "P2", "extended vs base vocabulary, PPAS order, all eligible"),
    ("P6", "P3", "extended vs base vocabulary, covered, deterministic"),
    ("P7", "P6", "PPAS vs deterministic, extended covered"),
    ("P7", "P5", "covered vs all, extended vocabulary, PPAS order"),
)
STRATA = (base_report._ALL, base_report._CHANGED, base_report._UNCHANGED)
_RANKING = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10")
_THRESHOLDED = ("f1", "precision", "recall", "threshold")
_NOT_IDENTIFIABLE = "not identifiable: identical selections on both sides"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _controlled_runs(results_dir: Path, source: str) -> List[Run]:
    """Load the controlled runs of one results root, tagging their origin."""
    runs = [r for runs in base_report.load_runs(results_dir).values() for r in runs
            if base_report.is_controlled(r)]
    for run in runs:
        run["_source_dir"] = source
    return runs


def collect_runs(reference_dir: Path, results_dir: Path) -> List[Run]:
    """
    Load the D1 reference and the validation runs, refusing mixed provenance.

    Parameters
    ----------
    reference_dir : Path
        #128 results root; only its D1 base-policy runs are used.
    results_dir : Path
        Validation results root.

    Returns
    -------
    List[Run]
        Reference P1-P4 runs followed by every validation run.

    Raises
    ------
    ValueError
        If the validation root holds a D1 run at the reference budget, the
        reference root holds anything but D1 at that budget, or the D1 runs
        were scored with different numbers of known pairs (#137).
    """
    reference = _controlled_runs(reference_dir, str(reference_dir))
    validation = _controlled_runs(results_dir, str(results_dir))
    if any(not _is_reference(r) for r in reference):
        raise ValueError(f"{reference_dir} holds runs other than D1 at {CONTROLLED_BUDGET} tokens.")
    if any(_is_reference(r) for r in validation):
        raise ValueError(f"{results_dir} holds D1 runs at {CONTROLLED_BUDGET}; that is the #128 reference.")
    runs = [r for r in reference if base_report._label(r) in BASE_LABELS] + validation
    check_one_known_pair_count([r for r in runs if r["dataset_id"] == "D1"])
    return runs


def check_one_known_pair_count(runs: List[Run]) -> None:
    """
    Refuse D1 budget rows scored on different populations.

    Parameters
    ----------
    runs : List[Run]
        D1 reference and validation runs, shown side by side by budget.

    Raises
    ------
    ValueError
        If the runs record more than one ``population.n_train_refs``.
    """
    counts = sorted({r["population"].get("n_train_refs", 0) for r in runs})
    if len(counts) > 1:
        raise ValueError(f"D1 runs were scored with {counts} known pairs; pass a reference and "
                         "validation tree from the same population.")


def _is_reference(run: Run) -> bool:
    return run["dataset_id"] == "D1" and run["token_budget"] == CONTROLLED_BUDGET


def groups_of(runs: List[Run], dataset_id: str) -> List[Tuple[Tuple, Group]]:
    """
    Return a dataset's signature groups, full populations first, by budget.

    Parameters
    ----------
    runs : List[Run]
        Controlled runs of every dataset.
    dataset_id : str
        Dataset to group.

    Returns
    -------
    List[Tuple[Tuple, Group]]
        ``(signature, {label: run})`` ordered by smoke flag, then budget.
    """
    groups = base_report.signature_groups([r for r in runs if r["dataset_id"] == dataset_id])
    return sorted(groups, key=lambda item: (item[0][6] is not None, item[0][5]))


def full_groups(groups: List[Tuple[Tuple, Group]]) -> Dict[int, Group]:
    """Map each budget to its full-population group; smoke groups are dropped."""
    full: Dict[int, Group] = {}
    for signature, group in groups:
        budget, smoke = signature[5], signature[6]
        if smoke is None and budget not in full:
            full[budget] = group
    return full


# ---------------------------------------------------------------------------
# Execution record
# ---------------------------------------------------------------------------


def signature_digest(run: Run) -> str:
    """Return a short digest of a run's comparison signature."""
    encoded = json.dumps(list(base_report.run_signature(run)), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _dataset_digest(run: Run) -> str:
    recorded = run.get("dataset_files_sha256")
    if not recorded:
        return "not recorded"
    encoded = json.dumps(recorded, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _record_row(run: Run) -> List[str]:
    block = run["ablation"]
    vocabulary = block["predicate_config"]
    smoke = block["smoke_max_sources"]
    return [
        run["dataset_id"], str(run["token_budget"]), base_report._label(run), f"`{run['condition_id']}`",
        f"{vocabulary['name']} `{vocabulary['config_hash'][:12]}`", f"`{run['_source_dir']}`",
        f"`{run['git_commit'][:8]}`", str(run["git_dirty"]), f"`{run['hf_revision'][:8]}`",
        f"`{block['tokenizer_revision'][:8]}`", f"`{run['candidates_sha256'][:12]}`",
        f"`{_dataset_digest(run)}`", str(run["evaluation_protocol"]),
        f"smoke ({smoke})" if smoke is not None else "full", block["status"], f"`{signature_digest(run)}`",
    ]


def execution_record(runs: List[Run]) -> str:
    """Tabulate the provenance of every run used by the report."""
    ordered = sorted(runs, key=lambda r: (r["dataset_id"], r["token_budget"], base_report._label(r)))
    header = ["Dataset", "Budget", "Policy", "Condition", "Vocabulary", "Results root", "Commit", "Dirty",
              "Model rev", "Tokenizer rev", "Candidates", "Dataset files", "Protocol", "Population",
              "Status", "Signature"]
    return base_report._table(header, [_record_row(r) for r in ordered])


def expected_runs() -> List[Tuple[str, int, str]]:
    """Return every (dataset, budget, label) the validation analysis needs."""
    d1 = [("D1", budget, label) for budget in D1_BUDGETS for label in BASE_LABELS]
    d2 = [("D2", CONTROLLED_BUDGET, label) for label in base_report.expected_labels("D2", CONTROLLED_BUDGET)]
    return d1 + d2


def completeness_section(runs: List[Run]) -> str:
    """List expected runs that are missing, failed or only smoke-tested."""
    complete = {(r["dataset_id"], r["token_budget"], base_report._label(r)) for r in runs
                if r["ablation"]["status"] == "completed" and r["ablation"]["smoke_max_sources"] is None}
    gaps = [f"- {d} at {b} tokens, {label}: missing, failed or smoke only"
            for d, b, label in expected_runs() if (d, b, label) not in complete]
    total = len(expected_runs())
    lines = [f"Expected full-population runs: {total}; complete: {total - len(gaps)}."]
    return "\n".join(lines + (gaps or ["- None missing, failed or smoke-only."]))


def dataset_version_section(runs: List[Run], data_dir: Path) -> str:
    """Compare recorded dataset digests with the files on disk and their mtimes."""
    lines = []
    for dataset_id in sorted({r["dataset_id"] for r in runs}):
        current = dataset_file_digests(dataset_id, data_dir)
        recorded = [r for r in runs if r["dataset_id"] == dataset_id and r.get("dataset_files_sha256")]
        agree = all(r["dataset_files_sha256"] == current for r in recorded)
        lines.append(f"- {dataset_id}: {len(recorded)} runs recorded file digests; "
                     f"all equal the files on disk: {agree}.")
        lines += [f"  - `{rel}` sha256 `{(digest or 'missing')[:16]}…`, modified {_mtime(data_dir / rel)}"
                  for rel, digest in current.items()]
    return "\n".join(lines)


def _mtime(path: Path) -> str:
    if not path.exists():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _audit_row(run: Run) -> List[str]:
    block = run["ablation"]
    cells = [run["dataset_id"], str(run["token_budget"]), base_report._label(run)]
    for side in ("source", "candidate"):
        summary = block[side]
        cells += [str(summary["tokens"]["max"]), str(summary.get("budget", {}).get("n_over_budget", "n/a")),
                  str(summary["n_model_truncated"]),
                  f"{summary['n_sampling_invoked'] / max(summary['n_entities'], 1):.1%}"]
    return cells


def budget_audit_section(runs: List[Run]) -> str:
    """Tabulate maximum tokens, budget violations, truncation and sampling per run."""
    ordered = sorted(runs, key=lambda r: (r["dataset_id"], r["token_budget"], base_report._label(r)))
    header = ["Dataset", "Budget", "Policy", "Src max tok", "Src over", "Src truncated", "Src sampled",
              "Cand max tok", "Cand over", "Cand truncated", "Cand sampled"]
    return base_report._table(header, [_audit_row(r) for r in ordered])


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def identifiable(a: Run, b: Run) -> bool:
    """Whether two runs select a different text for at least one entity."""
    return any(base_report.changed_entities(a, b, side) for side in ("source", "candidate"))


def strata_sources(a: Run, b: Run) -> Dict[str, set]:
    """Return the test sources of each stratum of a contrast."""
    every = set(base_report._per_source(a))
    changed = base_report._changed_sources(a, b)
    return {base_report._ALL: every, base_report._CHANGED: changed, base_report._UNCHANGED: every - changed}


def contrast_estimates(a: Run, b: Run) -> Dict[str, Dict[str, object]]:
    """
    Estimate paired MRR and Recall@1 differences, a − b, in each stratum.

    The generator is reseeded per stratum and draws MRR then Recall@1, exactly
    as the #128 report does, so both reports print identical intervals.

    Parameters
    ----------
    a, b : Run
        Runs of one signature group.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Stratum to ``{"n", "mrr", "r1"}``; each metric is ``(diff, low, high)``.
    """
    per_a, per_b = base_report._per_source(a), base_report._per_source(b)
    estimates = {}
    for stratum, sources in strata_sources(a, b).items():
        ordered = sorted(sources)
        rng = np.random.default_rng(base_report._BOOTSTRAP_SEED)
        metrics = [base_report.paired_bootstrap(np.array([per_a[s][c] for s in ordered]),
                                                np.array([per_b[s][c] for s in ordered]), rng)
                   for c in (0, 1)]
        estimates[stratum] = {"n": len(ordered), "mrr": metrics[0], "r1": metrics[1]}
    return estimates


def _interval(values: Tuple[float, float, float]) -> str:
    diff, low, high = values
    return f"{base_report._fmt(diff)} [{base_report._fmt(low)}, {base_report._fmt(high)}]"


def _contrast_rows(group: Group, contrasts: Tuple, budget: int) -> List[List[str]]:
    rows = []
    for a, b, effect in contrasts:
        if a not in group or b not in group:
            continue
        name = f"{a} − {b}"
        if not identifiable(group[a], group[b]):
            rows.append([name, effect, str(budget), base_report._ALL, "-", _NOT_IDENTIFIABLE, "-"])
            continue
        for stratum, values in contrast_estimates(group[a], group[b]).items():
            rows.append([name, effect, str(budget), stratum, str(values["n"]),
                         _interval(values["mrr"]), _interval(values["r1"])])
    return rows


def contrast_table(groups: Dict[int, Group], contrasts: Tuple) -> str:
    """Tabulate stratified paired contrasts for each budget's group, never pooled."""
    rows = [row for a, b, effect in contrasts for budget, group in sorted(groups.items())
            for row in _contrast_rows(group, ((a, b, effect),), budget)]
    header = ["Contrast", "Effect", "Budget", "Test sources", "n", "Δ MRR [95% CI]", "Δ R@1 [95% CI]"]
    return base_report._table(header, rows)


def _change_cells(a: Run, b: Run) -> List[str]:
    cells = []
    for side in ("source", "candidate"):
        n = len(base_report._fingerprints(a, side))
        changed = len(base_report.changed_entities(a, b, side))
        cells.append(f"{changed} / {n} ({changed / max(n, 1):.1%})")
    return cells


def change_table(groups: Dict[int, Group], contrasts: Tuple) -> str:
    """Count sources and candidates whose selected item set changes, per budget."""
    rows = [[f"{a} vs {b}", effect, str(budget), *_change_cells(group[a], group[b])]
            for a, b, effect in contrasts for budget, group in sorted(groups.items())
            if a in group and b in group]
    return base_report._table(["Contrast", "Effect", "Budget", "Sources changed", "Candidates changed"], rows)


# ---------------------------------------------------------------------------
# Metric and cost tables
# ---------------------------------------------------------------------------


def metric_table(groups: Dict[int, Group], metrics: Tuple[str, ...], labels: Tuple[str, ...]) -> str:
    """Tabulate the given metrics per budget and policy."""
    rows = [[str(budget), label, *[base_report._fmt(group[label]["metrics"][m]) for m in metrics]]
            for budget, group in sorted(groups.items()) for label in labels if label in group]
    return base_report._table(["Budget", "Policy", *metrics], rows)


def _cost_cells(run: Run, side: str) -> List[str]:
    summary = run["ablation"][side]
    tokens, budget = summary["tokens"], summary.get("budget", {})
    return [side, f"{tokens['mean']:.1f}", str(int(tokens["median"])), str(tokens["max"]),
            base_report._fmt(budget.get("utilisation_mean"), 3),
            f"{summary['n_sampling_invoked'] / max(summary['n_entities'], 1):.1%}",
            str(summary.get("n_items_before_filter", "n/a")), str(summary.get("n_filtered_items", "n/a")),
            str(summary["n_selected_items"])]


def cost_table(groups: Dict[int, Group], labels: Tuple[str, ...]) -> str:
    """Tabulate token use, utilisation, sampling and item accounting per budget and policy."""
    rows = [[str(budget), label, *_cost_cells(group[label], side)]
            for budget, group in sorted(groups.items()) for label in labels if label in group
            for side in ("source", "candidate")]
    header = ["Budget", "Policy", "Side", "Mean tok", "Median tok", "Max tok", "Utilisation", "Sampled",
              "Items", "Filtered", "Selected"]
    return base_report._table(header, rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_d1_metrics(groups: Dict[int, Group], path: Path) -> None:
    """Plot MRR and Recall@1 of P1-P4 against the D1 budget."""
    plt = _pyplot()
    budgets = sorted(groups)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, metric, title in zip(axes, ("mrr", "recall_at_1"), ("MRR", "Recall@1")):
        for label in BASE_LABELS:
            points = [(b, groups[b][label]["metrics"][metric]) for b in budgets if label in groups[b]]
            axis.plot(*zip(*points), marker="o", label=label)
        axis.set(title=f"D1 {title} by budget", xlabel="Token budget", xticks=budgets)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _stratum_series(groups: Dict[int, Group], a: str, b: str, stratum: str) -> List[Tuple[int, Tuple]]:
    return [(budget, contrast_estimates(group[a], group[b])[stratum]["mrr"])
            for budget, group in sorted(groups.items())
            if a in group and b in group and identifiable(group[a], group[b])]


def plot_d1_contrasts(groups: Dict[int, Group], path: Path) -> None:
    """Plot each primary contrast's Δ MRR with 95% intervals against the D1 budget, per stratum."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(PRIMARY_CONTRASTS), figsize=(14, 3.6), sharey=True)
    for axis, (a, b, effect) in zip(axes, PRIMARY_CONTRASTS):
        for offset, stratum in zip((-1.2, 0.0, 1.2), STRATA):
            series = _stratum_series(groups, a, b, stratum)
            if series:
                x = [budget + offset for budget, _ in series]
                diff = np.array([v[0] for _, v in series])
                err = np.array([[v[0] - v[1], v[2] - v[0]] for _, v in series]).T
                axis.errorbar(x, diff, yerr=err, marker="o", capsize=3, label=stratum)
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.set(title=f"{a} − {b}\n{effect}", xlabel="Token budget", xticks=sorted(groups))
        axis.title.set_fontsize(9)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Δ MRR (95% paired bootstrap)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_d2_contrasts(group: Group, path: Path) -> None:
    """Plot D2 Δ MRR with 95% intervals over all test sources for every available contrast."""
    plt = _pyplot()
    rows = [(f"{a} − {b}", contrast_estimates(group[a], group[b])[base_report._ALL]["mrr"])
            for a, b, _e in (*PRIMARY_CONTRASTS, *VOCABULARY_CONTRASTS)
            if a in group and b in group and identifiable(group[a], group[b])]
    fig, axis = plt.subplots(figsize=(6, 0.45 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    diff = np.array([v[0] for _, v in rows])
    err = np.array([[v[0] - v[1], v[2] - v[0]] for _, v in rows]).T
    axis.errorbar(diff, y, xerr=err, fmt="o", capsize=3)
    axis.set(yticks=y, yticklabels=[name for name, _ in rows], xlabel="Δ MRR, all test sources (95% CI)",
             title="D2 at 48 tokens (identifiable contrasts)")
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


_READING = """## Reading this report

- **Two analyses, never pooled.**  D1 budget sensitivity (P1-P4 at 32, 48,
  64) and the D2 replication (P1-P7 at 48) are separate.  Each bootstrap
  resamples the test sources of one pair at one budget: 2,000 resamples,
  seed 42 per contrast and stratum.  Intervals describe variation across test
  sources within that evaluation, not across KG pairs, and sources share
  candidate targets, so they are approximate.  No significance test is run.
- **Budgets share test sources.**  Differences between budgets are
  descriptive; no interval is given for them.
- **The 48-token D1 rows are the #128 reference**, read from
  `{reference_dir}` and not rerun.
- **Ranking metrics first.**  MRR and Recall@k do not depend on the
  validation-tuned threshold; F1, precision and recall do, and are tabulated
  separately.
- **Strata.**  "Selection differs for source or a gold target" holds sources
  whose own text or a gold target's text changes; its complement ("source and
  gold targets unchanged") moves only because competing candidates change.
- **Not identifiable.**  A contrast whose policies select identical texts for
  every entity on both sides is reported as not identifiable: the data cannot
  distinguish the two policies, which is not evidence of no effect.
- **What "covered" means.**  Under `base`, covered predicates are exactly the
  PPAS tier predicates.  Its OBO synonym entries use
  `http://purl.obolibrary.org/obo/`, which neither D1 nor D2 uses, so covered
  eligibility keeps SNOMED's SKOS synonyms (D1 sources) but drops FMA and
  OBO-in-OWL synonyms (D1 candidates, both D2 sides).  `base` is unchanged
  from #128.  D2's P5-P7 use `obo_extended`, which adds the oboInOwl synonym
  family and definition to class tier 0 and IDs untiered."""


def d1_parts(groups: Dict[int, Group], figures_dir: Path, figure_prefix: str) -> List[str]:
    """Render the D1 budget-sensitivity analysis."""
    plot_d1_metrics(groups, figures_dir / "d1_metrics_by_budget.png")
    plot_d1_contrasts(groups, figures_dir / "d1_contrasts_by_budget.png")
    return [
        "## D1 budget sensitivity (P1-P4; 48 = #128 reference)",
        "### Ranking metrics (threshold-independent)", metric_table(groups, _RANKING, BASE_LABELS),
        f"![D1 metrics by budget]({figure_prefix}d1_metrics_by_budget.png)",
        "### Threshold-dependent metrics (validation-tuned threshold)", metric_table(groups, _THRESHOLDED, BASE_LABELS),
        "### Primary contrasts (paired bootstrap per budget)", contrast_table(groups, PRIMARY_CONTRASTS),
        f"![D1 contrasts by budget]({figure_prefix}d1_contrasts_by_budget.png)",
        "### Selection changes", change_table(groups, PRIMARY_CONTRASTS),
        "### Representation cost and item accounting", cost_table(groups, BASE_LABELS),
    ]


def d2_parts(group: Group, figures_dir: Path, figure_prefix: str) -> List[str]:
    """Render the D2 replication analysis."""
    labels = tuple(base_report.expected_labels("D2", CONTROLLED_BUDGET))
    single = {CONTROLLED_BUDGET: group}
    plot_d2_contrasts(group, figures_dir / "d2_contrasts.png")
    return [
        "## D2 replication at 48 tokens (P1-P4 base, P5-P7 obo_extended)",
        "### Ranking metrics (threshold-independent)", metric_table(single, _RANKING, labels),
        "### Threshold-dependent metrics (validation-tuned threshold)", metric_table(single, _THRESHOLDED, labels),
        "### Primary contrasts", contrast_table(single, PRIMARY_CONTRASTS),
        "### Vocabulary contrasts (obo_extended)", contrast_table(single, VOCABULARY_CONTRASTS),
        f"![D2 contrasts]({figure_prefix}d2_contrasts.png)",
        "### Selection changes", change_table(single, (*PRIMARY_CONTRASTS, *VOCABULARY_CONTRASTS)),
        "### Representation cost and item accounting", cost_table(single, labels),
    ]


def detail_parts(groups: List[Tuple[Tuple, Group]], dataset_id: str) -> List[str]:
    """Render every signature group of a dataset with the #128 per-group tables."""
    parts = []
    for signature, group in groups:
        parts += [f"### {dataset_id}, budget {signature[5]}" + (" (SMOKE)" if signature[6] else ""),
                  "\n".join(base_report._signature_lines(signature, group)),
                  "#### Execution matrix", base_report.matrix_section(group),
                  "#### Representation cost and item accounting", base_report.cost_section(group),
                  "#### Paired contrasts", base_report.controlled_contrast_section(group)]
    return parts


def build_report(reference_dir: Path, results_dir: Path, figures_dir: Path, data_dir: Path) -> str:
    """
    Render the validation report and its figures.

    Parameters
    ----------
    reference_dir : Path
        #128 results root holding the D1 reference at 48 tokens.
    results_dir : Path
        Validation results root.
    figures_dir : Path
        Directory the PNG figures are written to.
    data_dir : Path
        Dataset root, for the dataset-version check.

    Returns
    -------
    str
        Markdown report; figures are linked relative to ``results_dir``.
    """
    runs = collect_runs(reference_dir, results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = _relative_prefix(figures_dir, results_dir)
    d1, d2 = groups_of(runs, "D1"), groups_of(runs, "D2")
    reading = _READING.format(reference_dir=f"{reference_dir.as_posix().rstrip('/')}/")
    sections = ["# Predicate-eligibility validation (#128 follow-up)", reading,
                "## Execution record", completeness_section(runs), execution_record(runs),
                "### Dataset versions", dataset_version_section(runs, data_dir),
                "### Budget audit", budget_audit_section(runs)]
    if full_groups(d1):
        sections += d1_parts(full_groups(d1), figures_dir, prefix)
    if full_groups(d2):
        sections += d2_parts(next(iter(full_groups(d2).values())), figures_dir, prefix)
    sections += ["## Per-group detail", *detail_parts(d1, "D1"), *detail_parts(d2, "D2")]
    return "\n\n".join(sections) + "\n"


def _relative_prefix(figures_dir: Path, results_dir: Path) -> str:
    try:
        return f"{figures_dir.resolve().relative_to(results_dir.resolve())}/"
    except ValueError:
        return f"{figures_dir.resolve()}/"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report the predicate-eligibility validation.")
    parser.add_argument("--reference_dir", default="data/results_ppas_controlled/", help="#128 results root.")
    parser.add_argument("--results_dir", default="data/results_ppas_validation/", help="Validation results root.")
    parser.add_argument("--output", default=None, help="Markdown path; defaults to <results_dir>/report.md.")
    parser.add_argument("--figures_dir", default=None, help="Defaults to <results_dir>/figures/.")
    parser.add_argument("--data_dir", default="data/", help="Dataset root, for the version check.")
    return parser


def main() -> None:
    """Build the validation report and write it."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    output = Path(args.output) if args.output else results_dir / "report.md"
    report = build_report(Path(args.reference_dir), results_dir, figures_dir, Path(args.data_dir))
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
