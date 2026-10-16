"""Pre-run inventory and run status of the controlled PPAS ablation (#122, #128).

Uses only the M2 tokenizer and the dataset graphs; no model is loaded and no
alignment is scored.  For each dataset it reports the eligible-item and
full-representation length distributions, recomputes the outcome-blind stress
budget rule, and, for every budget registered for the dataset, how many entities
each capped policy samples and how many sampled selections differ between
policies.  Run it before the experiment to confirm that sampling is invoked
often enough for a comparison to mean anything.

For the #128 controlled policies (P1-P7) registered on a dataset it reports,
per side, how many entities each policy samples, the items its filter
removes, the vocabulary's item coverage, and how many entities change their
selected item set across each pre-specified contrast.

With ``--results_dir`` it instead prints, without loading any tokenizer or
graph, the status of every registered condition on every requested dataset:
``completed``, ``smoke`` (run with ``--max_sources``), ``failed`` (unreadable
result, missing sidecar, or a status other than completed) or ``missing``.
A registered condition or an existing directory is never reported as run.

Usage
-----
python scripts/ppas_ablation_inventory.py --datasets D5 D1 --data_dir data/
python scripts/ppas_ablation_inventory.py --datasets D1 \\
    --results_dir data/results_ppas_controlled/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

from kgsemembed.datasets import load_dataset
from kgsemembed.embeddings.models import get_model_config
from kgsemembed.pipeline.ablation_conditions import (
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    AblationCondition,
    stress_budget,
)
from kgsemembed.pipeline.ppas_ablation import SideItems, build_pair_context
from kgsemembed.verbalisation.predicate_config import get_predicate_config
from kgsemembed.verbalisation.selection import (
    DETERMINISTIC,
    FULL,
    PPAS,
    RANDOM,
    PolicySelector,
    Selection,
    TokenCounter,
    make_token_counter,
)

_PERCENTILES = (50, 90, 99, 100)
_BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
_CONTRASTS = (("P3", "P1"), ("P4", "P3"), ("P2", "P1"), ("P5", "P2"), ("P6", "P3"), ("P7", "P6"), ("P7", "P5"))


def _selections(policy: str, budget: int | None, seed: int | None,
                side: SideItems, count_tokens: TokenCounter) -> List[Selection]:
    """Select every entity of a side under one policy."""
    selector = PolicySelector(policy, count_tokens, budget, seed)
    return [selector.select(items, uri, etype)
            for uri, etype, items in zip(side.uris, side.entity_types, side.items)]


def _condition_selections(condition: AblationCondition, side: SideItems,
                          count_tokens: TokenCounter) -> List[Selection]:
    """Select every entity of a side exactly as the runner does for a condition."""
    selector = PolicySelector(
        order_mode=condition.order_mode, count_tokens=count_tokens, budget=condition.token_budget,
        seed=condition.seed, filter_mode=condition.filter_mode,
        predicate_config=get_predicate_config(condition.predicate_config),
    )
    return [selector.select(items, uri, etype)
            for uri, etype, items in zip(side.uris, side.entity_types, side.items)]


def _controlled_report(dataset_id: str, sides: Dict[str, SideItems], count_tokens: TokenCounter) -> None:
    """Print sampling, filtering and contrast-level selection changes of P1-P7."""
    controlled = [c for c in ABLATION_CONDITIONS if c.is_controlled and dataset_id in c.datasets]
    if not controlled:
        return
    print("\n### Controlled policies (#128)\n")
    print("| policy | condition | side | sampled | filtered items | vocab coverage | tiered items | max tokens |")
    print("|---|---|---|---|---|---|---|---|")
    chosen: Dict[tuple, List[Selection]] = {}
    for condition in controlled:
        for name, side in sides.items():
            selections = _condition_selections(condition, side, count_tokens)
            chosen[(condition.policy_label, name)] = selections
            print(_controlled_row(condition, name, selections))
    print("\n| contrast | side | entities whose selected set changes |")
    print("|---|---|---|")
    for a, b in _CONTRASTS:
        for name in sides:
            if (a, name) in chosen and (b, name) in chosen:
                n = len(chosen[(a, name)])
                print(f"| {a} vs {b} | {name} | {_differ(chosen[(a, name)], chosen[(b, name)])} / {n} |")


def _controlled_row(condition: AblationCondition, side_name: str, selections: List[Selection]) -> str:
    """Format one controlled policy's accounting on one side."""
    before = sum(s.n_eligible + s.n_filtered_items for s in selections)
    covered = sum(s.n_covered_items for s in selections)
    sampled = sum(s.sampling_invoked for s in selections)
    return (
        f"| {condition.policy_label} | {condition.condition_id} | {side_name} | "
        f"{sampled} ({sampled / max(len(selections), 1):.1%}) | "
        f"{sum(s.n_filtered_items for s in selections)} | {covered / max(before, 1):.1%} | "
        f"{sum(s.n_tiered_items for s in selections)} | {max((s.n_tokens for s in selections), default=0)} |"
    )


def run_status(result_path: Path) -> str:
    """
    Classify one result file.

    Parameters
    ----------
    result_path : Path
        Expected ``{pair}_results.json`` path.

    Returns
    -------
    str
        ``"missing"``, ``"failed"``, ``"smoke"`` or ``"completed"``.
    """
    if not result_path.exists():
        return "missing"
    sidecar = result_path.with_name(result_path.name.replace("_results.json", "_entities.json"))
    try:
        block = json.loads(result_path.read_text(encoding="utf-8"))["ablation"]
    except (ValueError, KeyError):
        return "failed"
    if block.get("status") != "completed" or not sidecar.exists():
        return "failed"
    return "smoke" if block.get("smoke_max_sources") is not None else "completed"


def status_rows(results_dir: Path, dataset_id: str, pair_names: List[str]) -> List[List[str]]:
    """
    Return the status of every condition registered on a dataset, per pair.

    Parameters
    ----------
    results_dir : Path
        Ablation result root.
    dataset_id : str
        Dataset whose registered conditions are listed.
    pair_names : List[str]
        Pairs expected for the dataset.

    Returns
    -------
    List[List[str]]
        ``[condition_id, policy_label, pair_name, status]`` rows.
    """
    rows = []
    for condition in (c for c in ABLATION_CONDITIONS if dataset_id in c.datasets):
        for pair_name in pair_names:
            path = results_dir / condition.condition_id / dataset_id / f"{pair_name}_results.json"
            rows.append([condition.condition_id, condition.policy_label or "-", pair_name, run_status(path)])
    return rows


def _print_status(results_dir: Path, dataset_id: str, pair_names: List[str]) -> None:
    """Print the run-status table of one dataset."""
    rows = status_rows(results_dir, dataset_id, pair_names)
    print(f"\n## Run status: {dataset_id} in {results_dir}\n")
    print("| condition | policy | pair | status |")
    print("|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    counts = {status: sum(r[3] == status for r in rows) for status in ("completed", "smoke", "failed", "missing")}
    print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()))


_PAIR_NAMES = {"D1": ["d1_snomed_fma"], "D5": ["D_W_15K_V2"]}


def _percentiles(values: List[int]) -> str:
    """Format the p50/p90/p99/max of ``values``."""
    return "/".join(str(int(v)) for v in np.percentile(values, _PERCENTILES))


def _differ(first: List[Selection], second: List[Selection]) -> int:
    """Count entities whose selected item sets differ."""
    return sum(a.selected != b.selected for a, b in zip(first, second))


def _budget_rows(budget: int, side_name: str, side: SideItems,
                 count_tokens: TokenCounter) -> List[str]:
    """Report sampling coverage and policy disagreement for one side and budget."""
    by_policy = {p: _selections(p, budget, None, side, count_tokens) for p in (PPAS, DETERMINISTIC)}
    rand = [_selections(RANDOM, budget, seed, side, count_tokens) for seed in (0, 1)]
    sampled = sum(s.sampling_invoked for s in by_policy[PPAS])
    n = len(side.uris)
    tiered = sum(s.sampling_invoked and s.selected != d.selected
                 for s, d in zip(by_policy[PPAS], by_policy[DETERMINISTIC]))
    return [
        f"| {budget} | {side_name} | {n} | {sampled} ({sampled / max(n, 1):.1%}) | "
        f"{tiered} | {_differ(by_policy[PPAS], rand[0])} | {_differ(rand[0], rand[1])} | "
        f"{sum(s.n_oversized for s in by_policy[PPAS])} | {sum(not s.text for s in by_policy[PPAS])} |"
    ]


def _dataset_report(dataset_id: str, data_dir: Path, count_tokens: TokenCounter) -> None:
    """Print the inventory of every pair of one dataset."""
    budgets = sorted({c.token_budget for c in ABLATION_CONDITIONS
                      if dataset_id in c.datasets and c.token_budget is not None})
    for pair in load_dataset(dataset_id, data_dir):
        context = build_pair_context(pair, None, count_tokens, data_dir)
        sides: Dict[str, SideItems] = {"source": context.source, "candidate": context.candidate}
        full = {name: _selections(FULL, None, None, side, count_tokens) for name, side in sides.items()}
        print(f"\n## {dataset_id} / {pair.pair_name}\n")
        print("| side | entities | eligible items p50/p90/p99/max | full tokens p50/p90/p99/max | >512 |")
        print("|---|---|---|---|---|")
        for name, selections in full.items():
            tokens = [s.n_tokens for s in selections]
            print(f"| {name} | {len(selections)} | {_percentiles([s.n_eligible for s in selections])} | "
                  f"{_percentiles(tokens)} | {sum(t > 512 for t in tokens)} |")
        pooled = [s.n_tokens for selections in full.values() for s in selections]
        print(f"\nStress-budget rule (pooled median floored to 8): {stress_budget(pooled)}")
        print(f"Registered budgets: {budgets}\n")
        print("| budget | side | entities | sampled | PPAS != det | PPAS != rand s0 | rand s0 != s1 | oversized items | empty |")
        print("|---|---|---|---|---|---|---|---|---|")
        for budget in budgets:
            for name, side in sides.items():
                print("\n".join(_budget_rows(budget, name, side, count_tokens)))
        _controlled_report(dataset_id, sides, count_tokens)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory PPAS ablation sampling coverage.")
    parser.add_argument("--datasets", nargs="+", default=["D5", "D1"], help="Dataset IDs.")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    parser.add_argument("--results_dir", default=None,
                        help="Print the run status of every registered condition instead.")
    parser.add_argument("--revision", default=_BGE_REVISION, help="Tokenizer revision.")
    return parser


def main() -> None:
    """Load the M2 tokenizer and print the inventory of each dataset."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    if args.results_dir:
        for dataset_id in args.datasets:
            _print_status(Path(args.results_dir), dataset_id, _PAIR_NAMES[dataset_id])
        return
    model_id = get_model_config(ABLATION_MODEL_KEY).model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision)
    count_tokens = make_token_counter(tokenizer)
    for dataset_id in args.datasets:
        _dataset_report(dataset_id, Path(args.data_dir), count_tokens)


if __name__ == "__main__":
    main()
