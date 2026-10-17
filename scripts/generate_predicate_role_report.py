"""Report of the predicate-role ablation (#131).

Reads the ``PR-*`` results in ``data/results_predicate_roles/`` and, read-only,
the #128 lexicographic P1 runs at 48 tokens (D1 from
``data/results_ppas_controlled/``, D2 from ``data/results_ppas_validation/``),
used only to measure the ordering artefact.

Per dataset, never pooled across datasets or budgets, it reports:

* an execution record, completeness against ``ROLE_MATRIX``, dataset
  versions, and a budget and truncation audit (``full`` arms have no budget;
  only the model's 512-token input limit applies);
* tokens and items each arm removes, per role and side;
* ranking metrics, then threshold-dependent metrics, per budget and arm;
* removal effects ``R-x − A0`` and ``A-core − A0`` at each budget, and
  ``C+x − A-core`` without a budget, stratified by whether the source's own or
  a gold target's text changed;
* budget costs ``arm@b − arm@none``;
* the per-source difference-in-differences ``(R−A0)@b − (R−A0)@none``, which
  separates a role's removal effect from its interaction with the budget;
* the ordering artefact ``A0@48 (core-first) − P1@48 (lexicographic)``.

Every interval is a paired bootstrap over test sources of one pair (2,000
resamples, seed 42, 95%).  No significance test is run, and two pairs do not
establish generalisation.

Usage
-----
python scripts/generate_predicate_role_report.py \\
    --results_dir data/results_predicate_roles/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_ppas_ablation_report as base_report  # noqa: E402
import generate_ppas_validation_report as validation  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import ROLE_MATRIX  # noqa: E402

Run = Dict[str, object]
Group = Dict[str, Run]
Budget = Optional[int]

REMOVAL_ARMS = ("R-pref", "R-syn", "R-def", "R-id", "R-nonEn", "R-disj", "A-core")
PRIMARY_ARMS = ("R-pref", "R-syn", "R-def", "R-id")
ADDED_ARMS = ("C+pref", "C+syn", "C+def", "C+id")
LINK_BUDGET = 48
_RANKING = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10")
_THRESHOLDED = ("f1", "precision", "recall", "threshold")
_LINK_DIRS = {"D1": "data/results_ppas_controlled/", "D2": "data/results_ppas_validation/"}


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------


def arm(run: Run) -> str:
    """Return a run's role arm."""
    return run["ablation"]["role_arm"]


def budget_name(budget: Budget) -> str:
    """Render a budget, ``None`` as ``none``."""
    return "none" if budget is None else str(budget)


def load_role_runs(results_dir: Path) -> List[Run]:
    """Load every predicate-role run of a results root."""
    runs = [r for rs in base_report.load_runs(results_dir).values() for r in rs if r["ablation"].get("role_arm")]
    for run in runs:
        run["_source_dir"] = str(results_dir)
    return runs


def load_link_runs(dataset_id: str) -> List[Run]:
    """Load the lexicographic P1 run at the link budget of a dataset, read-only."""
    root = Path(_LINK_DIRS[dataset_id])
    if not root.exists():
        return []
    runs = [r for rs in base_report.load_runs(root).values() for r in rs
            if r["dataset_id"] == dataset_id and r["ablation"].get("policy_label") == "P1"
            and r["token_budget"] == LINK_BUDGET and r["ablation"]["smoke_max_sources"] is None]
    for run in runs:
        run["_source_dir"] = str(root)
    return runs


def full_groups(runs: List[Run], dataset_id: str) -> Dict[Budget, Group]:
    """
    Map each budget of a dataset to its full-population arms, one run per arm.

    Parameters
    ----------
    runs : List[Run]
        Predicate-role runs.
    dataset_id : str
        Dataset to group.

    Returns
    -------
    Dict[Budget, Group]
        Budget (``None`` = no budget) to ``{arm: run}``.

    Raises
    ------
    ValueError
        If two full runs share a dataset, budget and arm, or a budget's runs
        do not share one run signature.
    """
    groups: Dict[Budget, Group] = {}
    for run in runs:
        if run["dataset_id"] != dataset_id or run["ablation"]["smoke_max_sources"] is not None:
            continue
        group = groups.setdefault(run["token_budget"], {})
        if arm(run) in group:
            raise ValueError(f"Two full {arm(run)} runs on {dataset_id} at {budget_name(run['token_budget'])}.")
        group[arm(run)] = run
    for budget, group in groups.items():
        if len({base_report.run_signature(r) for r in group.values()}) != 1:
            raise ValueError(f"{dataset_id} at {budget_name(budget)}: runs differ in run signature.")
    return groups


def same_population(a: Run, b: Run) -> bool:
    """Whether two runs share model, candidates, protocol, smoke flag and test sources."""
    sa, sb = base_report.run_signature(a), base_report.run_signature(b)
    return sa[:5] == sb[:5] and sa[6:] == sb[6:]


# ---------------------------------------------------------------------------
# Execution record
# ---------------------------------------------------------------------------


def expected_runs() -> List[Tuple[str, Budget, str]]:
    """Return every (dataset, budget, arm) in :data:`ROLE_MATRIX`."""
    return [(d, b, a) for d, plan in ROLE_MATRIX.items() for b, arms in plan.items() for a in arms]


def completeness_section(runs: List[Run]) -> str:
    """List matrix cells that are missing, failed or only smoke-tested."""
    complete = {(r["dataset_id"], r["token_budget"], arm(r)) for r in runs
                if r["ablation"]["status"] == "completed" and r["ablation"]["smoke_max_sources"] is None}
    expected = expected_runs()
    gaps = [f"- {d} at {budget_name(b)}, {a}: missing, failed or smoke only"
            for d, b, a in expected if (d, b, a) not in complete]
    lines = [f"Expected full-population runs: {len(expected)}; complete: {len(expected) - len(gaps)}."]
    return "\n".join(lines + (gaps or ["- None missing, failed or smoke-only."]))


def _record_row(run: Run) -> List[str]:
    block = run["ablation"]
    role_map = block.get("role_map") or {}
    label = arm(run) if block.get("role_arm") else f"{block['policy_label']} (link)"
    smoke = block["smoke_max_sources"]
    return [run["dataset_id"], budget_name(run["token_budget"]), label, f"`{run['condition_id']}`",
            block["order_mode"], f"`{role_map.get('config_hash', '-')[:12]}`", f"`{run['_source_dir']}`",
            f"`{run['git_commit'][:8]}`", str(run["git_dirty"]), f"`{run['hf_revision'][:8]}`",
            f"`{run['candidates_sha256'][:12]}`", f"`{validation._dataset_digest(run)}`",
            str(run["evaluation_protocol"]), f"smoke ({smoke})" if smoke is not None else "full",
            block["status"], f"`{validation.signature_digest(run)}`"]


def execution_record(runs: List[Run]) -> str:
    """Tabulate the provenance of every run the report uses."""
    ordered = sorted(runs, key=lambda r: (r["dataset_id"], r["token_budget"] or 0, r["condition_id"]))
    header = ["Dataset", "Budget", "Arm", "Condition", "Order", "Role map", "Results root", "Commit", "Dirty",
              "Model rev", "Candidates", "Dataset files", "Protocol", "Population", "Status", "Signature"]
    return base_report._table(header, [_record_row(r) for r in ordered])


def _audit_row(run: Run) -> List[str]:
    cells = [run["dataset_id"], budget_name(run["token_budget"]), arm(run)]
    for side in ("source", "candidate"):
        summary = run["ablation"][side]
        over = summary.get("budget", {}).get("n_over_budget", "n/a") if run["token_budget"] else "n/a"
        cells += [str(summary["tokens"]["max"]), str(over), str(summary["n_model_truncated"]),
                  str(summary.get("n_truncated_tokens", "n/a")),
                  f"{summary['n_sampling_invoked'] / max(summary['n_entities'], 1):.1%}"]
    return cells


def audit_section(runs: List[Run]) -> str:
    """Tabulate maximum tokens, budget violations, model truncation and sampling per run."""
    ordered = sorted(runs, key=lambda r: (r["dataset_id"], r["token_budget"] or 0, r["condition_id"]))
    header = ["Dataset", "Budget", "Arm"] + [f"{s} {c}" for s in ("Src", "Cand")
                                             for c in ("max tok", "over budget", "truncated texts",
                                                       "truncated tokens", "sampled")]
    return base_report._table(header, [_audit_row(r) for r in ordered])


def _removal_rows(run: Run) -> List[List[str]]:
    rows = []
    for side in ("source", "candidate"):
        for role, entry in sorted(run["ablation"][side].get("role_removal", {}).items()):
            rows.append([run["dataset_id"], arm(run), side, role, str(entry["n_entities"]),
                         str(entry["n_items"]), str(entry["item_tokens"])])
    return rows


def removal_section(runs: List[Run]) -> str:
    """Tabulate entities, items and item tokens each arm removes, per role and side (no-budget runs)."""
    unbudgeted = sorted((r for r in runs if r["token_budget"] is None), key=lambda r: (r["dataset_id"], r["condition_id"]))
    rows = [row for run in unbudgeted for row in _removal_rows(run)]
    return base_report._table(["Dataset", "Arm", "Side", "Role", "Entities", "Items", "Item tokens"], rows)


# ---------------------------------------------------------------------------
# Metrics and contrasts
# ---------------------------------------------------------------------------


def metric_table(groups: Dict[Budget, Group], metrics: Tuple[str, ...]) -> str:
    """Tabulate metrics per budget and arm."""
    rows = [[budget_name(b), a, *[base_report._fmt(g[a]["metrics"][m]) for m in metrics]]
            for b, g in sorted(groups.items(), key=lambda item: item[0] or 0) for a in g]
    return base_report._table(["Budget", "Arm", *metrics], rows)


def _contrast_rows(name: str, a: Run, b: Run, budget: str) -> List[List[str]]:
    if not validation.identifiable(a, b):
        return [[name, budget, base_report._ALL, "-", validation._NOT_IDENTIFIABLE, "-"]]
    return [[name, budget, stratum, str(v["n"]), validation._interval(v["mrr"]), validation._interval(v["r1"])]
            for stratum, v in validation.contrast_estimates(a, b).items()]


def _table_of(rows: List[List[str]]) -> str:
    return base_report._table(["Contrast", "Budget", "Test sources", "n", "Δ MRR [95% CI]", "Δ R@1 [95% CI]"], rows)


def removal_contrasts(groups: Dict[Budget, Group]) -> str:
    """Tabulate ``arm − A0`` per budget for every removal arm present."""
    rows = [row for b, g in sorted(groups.items(), key=lambda item: item[0] or 0) if "A0" in g
            for a in REMOVAL_ARMS if a in g for row in _contrast_rows(f"{a} − A0", g[a], g["A0"], budget_name(b))]
    return _table_of(rows)


def added_contrasts(groups: Dict[Budget, Group]) -> str:
    """Tabulate ``C+x − A-core`` without a budget."""
    group = groups.get(None, {})
    rows = [row for a in ADDED_ARMS if a in group and "A-core" in group
            for row in _contrast_rows(f"{a} − A-core", group[a], group["A-core"], "none")]
    return _table_of(rows)


def budget_cost_contrasts(groups: Dict[Budget, Group]) -> str:
    """Tabulate ``arm@b − arm@none`` for every arm run at a budget."""
    unbudgeted = groups.get(None, {})
    rows = [row for b, g in sorted((k, v) for k, v in groups.items() if k is not None) for a in g
            if a in unbudgeted and same_population(g[a], unbudgeted[a])
            for row in _contrast_rows(f"{a}@{b} − {a}@none", g[a], unbudgeted[a], str(b))]
    return _table_of(rows)


def did_estimate(removed_b: Run, base_b: Run, removed_n: Run, base_n: Run) -> Tuple[int, Tuple, Tuple]:
    """
    Bootstrap the per-source difference-in-differences of MRR and Recall@1.

    Each source contributes ``(R − A0)@b − (R − A0)@none``; its four runs are
    resampled together.

    Parameters
    ----------
    removed_b, base_b : Run
        Removal arm and A0 at the budget.
    removed_n, base_n : Run
        Removal arm and A0 without a budget.

    Returns
    -------
    Tuple[int, Tuple, Tuple]
        Number of test sources, and ``(diff, low, high)`` for MRR and Recall@1.
    """
    per = [base_report._per_source(r) for r in (removed_b, base_b, removed_n, base_n)]
    sources = sorted(set.intersection(*(set(p) for p in per)))
    rng = np.random.default_rng(base_report._BOOTSTRAP_SEED)
    estimates = []
    for column in (0, 1):
        values = np.array([(per[0][s][column] - per[1][s][column]) - (per[2][s][column] - per[3][s][column])
                           for s in sources])
        estimates.append(base_report.paired_bootstrap(values, np.zeros_like(values), rng))
    return len(sources), estimates[0], estimates[1]


def did_section(groups: Dict[Budget, Group]) -> str:
    """Tabulate the difference-in-differences of every removal arm run with and without a budget."""
    unbudgeted = groups.get(None, {})
    rows = []
    for b, g in sorted((k, v) for k, v in groups.items() if k is not None):
        for a in REMOVAL_ARMS:
            if a in g and a in unbudgeted and "A0" in g and "A0" in unbudgeted:
                n, mrr, r1 = did_estimate(g[a], g["A0"], unbudgeted[a], unbudgeted["A0"])
                rows.append([f"({a} − A0)@{b} − ({a} − A0)@none", str(b), str(n),
                             validation._interval(mrr), validation._interval(r1)])
    return base_report._table(["Contrast", "Budget", "n", "Δ MRR [95% CI]", "Δ R@1 [95% CI]"], rows)


def order_section(groups: Dict[Budget, Group], links: List[Run]) -> str:
    """Tabulate ``A0@48 (core-first) − P1@48 (lexicographic)`` when both share a population."""
    a0 = groups.get(LINK_BUDGET, {}).get("A0")
    usable = [r for r in links if a0 is not None and same_population(a0, r)]
    if not usable:
        return "No comparable lexicographic P1 run at 48 tokens was found; the ordering artefact is not reported."
    return _table_of(_contrast_rows("A0 core-first − P1 lexicographic", a0, usable[0], str(LINK_BUDGET)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_removal_effects(groups: Dict[Budget, Group], dataset_id: str, path: Path) -> bool:
    """Plot ``arm − A0`` Δ MRR with 95% intervals over all test sources, one marker per budget."""
    plt = validation._pyplot()
    budgets = sorted(groups, key=lambda b: (b is not None, b or 0))
    arms = [a for a in REMOVAL_ARMS if any(a in groups[b] for b in budgets)]
    if not arms:
        return False
    fig, axis = plt.subplots(figsize=(7, 0.55 * len(arms) + 1.5))
    for offset, b in zip(np.linspace(-0.25, 0.25, len(budgets)), budgets):
        points = [(i, validation.contrast_estimates(groups[b][a], groups[b]["A0"])[base_report._ALL]["mrr"])
                  for i, a in enumerate(arms) if a in groups[b] and "A0" in groups[b]
                  and validation.identifiable(groups[b][a], groups[b]["A0"])]
        if points:
            y = np.array([len(arms) - 1 - i + offset for i, _ in points])
            diff = np.array([v[0] for _, v in points])
            err = np.array([[v[0] - v[1], v[2] - v[0]] for _, v in points]).T
            axis.errorbar(diff, y, xerr=err, fmt="o", capsize=3, label=f"budget {budget_name(b)}")
    axis.set(yticks=range(len(arms)), yticklabels=[f"{a} − A0" for a in reversed(arms)],
             xlabel="Δ MRR, all test sources (95% paired bootstrap)", title=f"{dataset_id}: removal effects")
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


_READING = """## Reading this report

- **Arms.**  A0 keeps every item; R-x removes role x; A-core keeps only
  `rdfs:label`, `rdf:type` and `rdfs:subClassOf`; C+x keeps core plus role x.
  Removal applies to both sides.  Roles come from `roles_v1`, fixed from
  predicate declarations and values before any run.
- **Budgets.**  "none" is the `full` order: every remaining item, joined in
  canonical order, with only the model's 512-token input limit (truncated
  texts and tokens are in the audit).  Budgeted arms use core-first order, so
  a budget never drops `rdfs:label` for an annotation that sorts earlier.
- **Effects.**  `R − A0` without a budget is the removal effect with no
  budget involved.  The difference-in-differences `(R − A0)@b − (R − A0)@none`
  is the part of the removal effect that depends on the budget (freed tokens
  re-allocated, or annotations crowding out other items).  `arm@b − arm@none`
  is the cost of the budget itself.
- **Strata.**  "Selection differs for source or a gold target" holds sources
  whose own text or a gold target's text changes; the complement moves only
  through competing candidates.  An entity without the removed role keeps
  byte-identical text, which makes it a built-in negative control.
- **Not identifiable.**  Arms whose role is absent from a dataset select the
  same text as A0 and are reported as not identifiable, not as a null effect.
- **Uncertainty.**  Paired bootstrap over test sources of one pair, 2,000
  resamples, seed 42; budgets and datasets are never pooled.  Intervals are
  conditional on one evaluation per pair, approximate because sources share
  candidates, and carry no significance test.  F1, precision and recall depend
  on the validation-tuned threshold and are tabulated apart from ranking
  metrics.
- **Preferred names duplicate labels** for 99% of D1 entities (after case and
  underscore normalisation), so R-pref mostly removes a repeated name."""


def dataset_parts(dataset_id: str, groups: Dict[Budget, Group], links: List[Run],
                  figures_dir: Path, prefix: str) -> List[str]:
    """Render one dataset's analysis."""
    figure = figures_dir / f"{dataset_id.lower()}_removal_effects.png"
    parts = [f"## {dataset_id}",
             "### Ranking metrics (threshold-independent)", metric_table(groups, _RANKING),
             "### Threshold-dependent metrics (validation-tuned threshold)", metric_table(groups, _THRESHOLDED),
             "### Removal effects: arm − A0, per budget", removal_contrasts(groups)]
    if plot_removal_effects(groups, dataset_id, figure):
        parts.append(f"![{dataset_id} removal effects]({prefix}{figure.name})")
    return parts + [
        "### Removal × budget: difference-in-differences", did_section(groups),
        "### Budget cost: arm@budget − arm@none", budget_cost_contrasts(groups),
        "### Core plus one role: C+x − A-core (no budget)", added_contrasts(groups),
        "### Ordering artefact at 48 tokens", order_section(groups, links),
    ]


def build_report(results_dir: Path, figures_dir: Path, data_dir: Path) -> str:
    """
    Render the predicate-role report and its figures.

    Parameters
    ----------
    results_dir : Path
        Predicate-role results root.
    figures_dir : Path
        Directory the PNG figures are written to.
    data_dir : Path
        Dataset root, for the dataset-version check.

    Returns
    -------
    str
        Markdown report; figures are linked relative to ``results_dir``.
    """
    runs = load_role_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No predicate-role results under {results_dir}.")
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = validation._relative_prefix(figures_dir, results_dir)
    links = {d: load_link_runs(d) for d in ROLE_MATRIX}
    sections = ["# Predicate-role ablation (#131)", _READING,
                "## Execution record", completeness_section(runs),
                execution_record(runs + [r for rs in links.values() for r in rs]),
                "### Dataset versions", validation.dataset_version_section(runs, data_dir),
                "### Budget and truncation audit", audit_section(runs),
                "### Removed by role (no-budget arms)", removal_section(runs)]
    for dataset_id in ROLE_MATRIX:
        groups = full_groups(runs, dataset_id)
        if groups:
            sections += dataset_parts(dataset_id, groups, links[dataset_id], figures_dir, prefix)
    return "\n\n".join(sections) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report the predicate-role ablation.")
    parser.add_argument("--results_dir", default="data/results_predicate_roles/", help="Results root.")
    parser.add_argument("--output", default=None, help="Markdown path; defaults to <results_dir>/report.md.")
    parser.add_argument("--figures_dir", default=None, help="Defaults to <results_dir>/figures/.")
    parser.add_argument("--data_dir", default="data/", help="Dataset root, for the version check.")
    return parser


def main() -> None:
    """Build the predicate-role report and write it."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    output = Path(args.output) if args.output else results_dir / "report.md"
    output.write_text(build_report(results_dir, figures_dir, Path(args.data_dir)), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
