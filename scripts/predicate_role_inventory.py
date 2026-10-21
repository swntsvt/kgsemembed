"""Pre-run inventory and run status of the predicate-role ablation (#131).

Uses only the M2 tokenizer and the dataset graphs; no model is loaded and no
alignment is scored.  For each dataset it checks that the role map covers every
predicate, tabulates each role's entities, items and item tokens per side, and,
for every registered (budget, arm), how many entities are sampled, the longest
input, how many inputs the model would truncate at its 512-token limit, how
many entities change their text relative to A0 at the same budget, and how
often ``rdfs:label`` is dropped by the budget.

With ``--results_dir`` it instead prints the status of every registered role
condition on each requested dataset.

Usage
-----
python scripts/predicate_role_inventory.py --datasets D1 D2 --data_dir data/
python scripts/predicate_role_inventory.py --datasets D1 D2 \\
    --results_dir data/results_predicate_roles/
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppas_ablation_inventory import _PAIR_NAMES, run_status  # noqa: E402

from kgsemembed.datasets import load_dataset  # noqa: E402
from kgsemembed.embeddings.models import get_model_config  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import (  # noqa: E402
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    ROLE_MAP,
    AblationCondition,
)
from kgsemembed.pipeline.ppas_ablation import SideItems, build_pair_context, condition_selector  # noqa: E402
from kgsemembed.verbalisation.predicate_roles import ROLES, get_role_map  # noqa: E402
from kgsemembed.verbalisation.selection import ITEM_SEPARATOR, Selection, TokenCounter, make_token_counter  # noqa: E402

_BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
_MAX_SEQ_LENGTH = 512
_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def role_conditions(dataset_id: str) -> List[AblationCondition]:
    """Return the role conditions registered on a dataset, in registry order."""
    return [c for c in ABLATION_CONDITIONS if c.is_role_ablation and dataset_id in c.datasets]


def role_table(side: SideItems, count_tokens: TokenCounter) -> List[str]:
    """Tabulate each role's entities, items and item-token share on one side."""
    role_map = get_role_map(ROLE_MAP)
    entities, items, tokens = Counter(), Counter(), Counter()
    for entity_items in side.items:
        roles = [role_map.role_of(i.predicate) for i in entity_items]
        entities.update(set(roles))
        items.update(roles)
        for item, role in zip(entity_items, roles):
            tokens[role] += count_tokens(item.text) - 2
    total = max(sum(tokens.values()), 1)
    n = max(len(side.uris), 1)
    return [f"| {role} | {entities[role]} ({entities[role] / n:.1%}) | {items[role]} | {tokens[role] / total:.1%} |"
            for role in ROLES if items[role]]


def label_dropped(selection: Selection, items: list) -> bool:
    """
    Whether a budget dropped every ``rdfs:label`` item of an entity.

    Compares item texts, never ``Selection.selected``: those index the
    role-filtered item list, not ``items``.
    """
    labels = {i.text for i in items if i.predicate == _LABEL}
    kept = set(selection.text.split(ITEM_SEPARATOR)) if selection.text else set()
    return bool(labels) and selection.sampling_invoked and not labels & kept


def arm_row(condition: AblationCondition, side_name: str, side: SideItems,
            selections: List[Selection], reference: Optional[List[Selection]]) -> str:
    """Format one arm's sampling, length, truncation, change and label-loss counts on a side."""
    n = max(len(selections), 1)
    tokens = np.array([s.n_tokens for s in selections])
    sampled = sum(s.sampling_invoked for s in selections)
    changed = "-" if reference is None else str(sum(a.text != b.text for a, b in zip(selections, reference)))
    lost = sum(label_dropped(s, side.items[k]) for k, s in enumerate(selections) if s.sampling_invoked)
    return (f"| {condition.role_arm} | {condition.condition_id} | {side_name} | {sampled} ({sampled / n:.1%}) | "
            f"{int(tokens.max())} | {int((tokens > _MAX_SEQ_LENGTH).sum())} | "
            f"{sum(s.n_role_removed_items for s in selections)} | {changed} | {lost} |")


def _arm_rows(dataset_id: str, sides: Dict[str, SideItems], count_tokens: TokenCounter) -> List[str]:
    """Select every side under each role condition; A0 is the per-budget reference (registered first)."""
    rows, reference = [], {}
    for condition in role_conditions(dataset_id):
        selector = condition_selector(condition, count_tokens)
        for name, side in sides.items():
            selections = [selector.select(i, u, t) for u, t, i in zip(side.uris, side.entity_types, side.items)]
            key = (condition.token_budget, name)
            if condition.role_arm == "A0":
                reference[key] = selections
            baseline = None if condition.role_arm == "A0" else reference.get(key)
            rows.append(arm_row(condition, name, side, selections, baseline))
    return rows


def dataset_report(dataset_id: str, data_dir: Path, count_tokens: TokenCounter) -> None:
    """Print the role inventory of every pair of one dataset."""
    for pair in load_dataset(dataset_id, data_dir):
        context = build_pair_context(pair, None, count_tokens, data_dir)
        sides = {"source": context.source, "candidate": context.candidate}
        predicates = {i.predicate for side in sides.values() for items in side.items for i in items}
        print(f"\n## {dataset_id} / {pair.pair_name}\n")
        print(f"Unmapped predicates under {ROLE_MAP}: {get_role_map(ROLE_MAP).unmapped(predicates) or 'none'}")
        for name, side in sides.items():
            print(f"\n### {name}: roles\n\n| role | entities | items | item-token share |\n|---|---|---|---|")
            print("\n".join(role_table(side, count_tokens)))
        print("\n### Arms\n\n| arm | condition | side | sampled | max tokens | > 512 (truncated) | "
              "items removed | text changed vs A0 | rdfs:label dropped |\n|---|---|---|---|---|---|---|---|---|")
        print("\n".join(_arm_rows(dataset_id, sides, count_tokens)))


def print_status(results_dir: Path, dataset_id: str) -> None:
    """Print the run status of every role condition registered on a dataset."""
    rows = [(c.condition_id, c.role_arm, pair, run_status(results_dir / c.condition_id / dataset_id / f"{pair}_results.json"))
            for c in role_conditions(dataset_id) for pair in _PAIR_NAMES[dataset_id]]
    print(f"\n## Run status: {dataset_id} in {results_dir}\n\n| condition | arm | pair | status |\n|---|---|---|---|")
    print("\n".join(f"| {' | '.join(row)} |" for row in rows))
    counts = Counter(row[3] for row in rows)
    print("\n" + ", ".join(f"{k}: {counts.get(k, 0)}" for k in ("completed", "smoke", "failed", "missing")))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory the predicate-role ablation.")
    parser.add_argument("--datasets", nargs="+", default=["D1", "D2"], help="Dataset IDs.")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    parser.add_argument("--results_dir", default=None, help="Print run status instead.")
    parser.add_argument("--revision", default=_BGE_REVISION, help="Tokenizer revision.")
    return parser


def main() -> None:
    """Load the M2 tokenizer and print the role inventory of each dataset."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    if args.results_dir:
        for dataset_id in args.datasets:
            print_status(Path(args.results_dir), dataset_id)
        return
    tokenizer = AutoTokenizer.from_pretrained(get_model_config(ABLATION_MODEL_KEY).model_id, revision=args.revision)
    for dataset_id in args.datasets:
        dataset_report(dataset_id, Path(args.data_dir), make_token_counter(tokenizer))


if __name__ == "__main__":
    main()
