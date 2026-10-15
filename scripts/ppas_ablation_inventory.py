"""Pre-run inventory of the controlled PPAS ablation (#122).

Uses only the M2 tokenizer and the dataset graphs; no model is loaded and no
alignment is scored.  For each dataset it reports the eligible-item and
full-representation length distributions, recomputes the outcome-blind stress
budget rule, and, for every budget registered for the dataset, how many entities
each capped policy samples and how many sampled selections differ between
policies.  Run it before the experiment to confirm that sampling is invoked
often enough for a comparison to mean anything.

Usage
-----
python scripts/ppas_ablation_inventory.py --datasets D5 D1 --data_dir data/
"""

from __future__ import annotations

import argparse
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
    stress_budget,
)
from kgsemembed.pipeline.ppas_ablation import SideItems, build_pair_context
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


def _selections(policy: str, budget: int | None, seed: int | None,
                side: SideItems, count_tokens: TokenCounter) -> List[Selection]:
    """Select every entity of a side under one policy."""
    selector = PolicySelector(policy, count_tokens, budget, seed)
    return [selector.select(items, uri, etype)
            for uri, etype, items in zip(side.uris, side.entity_types, side.items)]


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory PPAS ablation sampling coverage.")
    parser.add_argument("--datasets", nargs="+", default=["D5", "D1"], help="Dataset IDs.")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    return parser


def main() -> None:
    """Load the M2 tokenizer and print the inventory of each dataset."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(get_model_config(ABLATION_MODEL_KEY).model_id)
    count_tokens = make_token_counter(tokenizer)
    for dataset_id in args.datasets:
        _dataset_report(dataset_id, Path(args.data_dir), count_tokens)


if __name__ == "__main__":
    main()
