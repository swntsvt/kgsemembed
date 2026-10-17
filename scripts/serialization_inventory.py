"""Identity gates and text diagnostics of the serialization experiment (#134).

Tokenizer only: no model is loaded and nothing is ranked.  Two subcommands:

``gates``
    For every logical condition of the #134 matrix on D1/D2, select and render
    every source and candidate entity, then check, entity by entity:

    * fixed-set identity: equal item-set fingerprints within every fixed-set
      group (render orders of a Block S row; B1-B4 with coupled core; B6/B8;
      B7/B9);
    * shared conditions: a logical condition that shares an execution renders
      exactly the text of the execution it shares;
    * reproduction: every canonically rendered execution reproduces the
      stored #131/#128 text fingerprint of every entity, both sides;
    * budget: no rendered text exceeds its budget.

    It also writes the required diagnostics: label retention and label
    position per logical condition and side, selection changes of the coupled
    and selection-only conditions against the fixed (core-first) set, and how
    many entities each render order changes.  Exit status 1 if any gate fails.

``existing``
    Recompute every registered #122/#128/#131 ablation condition on D1/D2
    with the current code and compare it with the stored sidecar fingerprints:
    the unchanged default render order must reproduce every stored text.

Usage
-----
python scripts/serialization_inventory.py gates --datasets D1 D2 \\
    --output_dir data/results_serialization/inventory/
python scripts/serialization_inventory.py existing --datasets D1 D2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predicate_role_diagnostics import token_counter  # noqa: E402

from kgsemembed.datasets import load_dataset  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import ABLATION_CONDITIONS, ROLE_ARMS, ROLE_MAP  # noqa: E402
from kgsemembed.pipeline.ppas_ablation import (  # noqa: E402
    CONTROLLED_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
    ROLE_RESULTS_DIR,
    VALIDATION_RESULTS_DIR,
    PairContext,
    SideItems,
    _text_fingerprint,
    build_pair_context,
    condition_selector,
)
from kgsemembed.pipeline.serialization_conditions import (  # noqa: E402
    B_BUDGETS,
    FIXED_SET_GROUPS,
    LOGICAL_CONDITIONS,
    SELECTION_MODE,
    LogicalCondition,
    reproduction_target,
)
from kgsemembed.pipeline.serialization_order import item_set_fingerprint  # noqa: E402
from kgsemembed.verbalisation.predicate_roles import RoleFilter, get_role_map  # noqa: E402
from kgsemembed.verbalisation.selection import FULL, ITEM_SEPARATOR, PolicySelector, TokenCounter  # noqa: E402
from kgsemembed.verbalisation.serialization import RDFS_LABEL  # noqa: E402

SIDES = ("source", "candidate")
LOGICAL_BY_NAME = {c.name: c for c in LOGICAL_CONDITIONS}
DEFAULT_OUTPUT = Path("data/results_serialization/inventory")
_SIDECAR = "{d}/{c}/{ds}/{pair}_entities.json"


@dataclass
class Rendered:
    """
    Per-entity outcome of one logical condition on one side.

    Attributes
    ----------
    text_fp, set_fp : List[str]
        Text and item-set fingerprints, in side order.
    keys : List[frozenset]
        Selected ``(predicate, text)`` keys, in side order.
    tokens : List[int]
        Encoded length of each rendered text.
    label_offset : List[Optional[int]]
        Tokens before the label item (``None``: no label kept).
    has_label : List[bool]
        Whether the entity has a label item after role filtering.
    """

    text_fp: List[str] = field(default_factory=list)
    set_fp: List[str] = field(default_factory=list)
    keys: List[frozenset] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    label_offset: List[Optional[int]] = field(default_factory=list)
    has_label: List[bool] = field(default_factory=list)


def logical_selector(logical: LogicalCondition, count: TokenCounter) -> PolicySelector:
    """
    Build the selector of a logical condition, independently of the registry.

    Parameters
    ----------
    logical : LogicalCondition
        Logical condition.
    count : TokenCounter
        Tokenizer counter.

    Returns
    -------
    PolicySelector
        Its representation's role filter, selection order, budget and render order.
    """
    order = FULL if logical.budget is None else SELECTION_MODE[logical.selection]
    role_filter = RoleFilter(get_role_map(ROLE_MAP), frozenset(ROLE_ARMS[logical.representation]))
    return PolicySelector(order_mode=order, count_tokens=count, budget=logical.budget,
                          role_filter=role_filter, render_order=logical.render)


def label_offset(text: str, label_texts: List[str], count: TokenCounter) -> Optional[int]:
    """
    Return the number of tokens written before the first label item.

    Parameters
    ----------
    text : str
        Rendered text.
    label_texts : List[str]
        Texts of the selected ``rdfs:label`` items.
    count : TokenCounter
        Encoded length, special tokens included (subtracted here).

    Returns
    -------
    Optional[int]
        0 when the text starts with a label item; ``None`` without a label.
    """
    starts = [0 if text.startswith(t) else text.find(ITEM_SEPARATOR + t) for t in label_texts]
    starts = [s for s in starts if s >= 0]
    if not starts:
        return None
    first = min(starts)
    return 0 if first == 0 else count(text[:first + len(ITEM_SEPARATOR)]) - 2


def render_side(selector: PolicySelector, side: SideItems, count: TokenCounter, keep_keys: bool) -> Rendered:
    """Select and render every entity of a side; item keys only when ``keep_keys`` (memory)."""
    out = Rendered()
    for uri, entity_type, items in zip(side.uris, side.entity_types, side.items):
        selection = selector.select(items, uri, entity_type)
        kept = selector.kept_items(items)
        chosen = [kept[i] for i in selection.selected]
        out.text_fp.append(_text_fingerprint(selection.text))
        out.set_fp.append(item_set_fingerprint(chosen))
        out.keys.append(frozenset((i.predicate, i.text) for i in chosen) if keep_keys else frozenset())
        out.tokens.append(selection.n_tokens)
        labels = [i.text for i in chosen if i.predicate == RDFS_LABEL]
        out.label_offset.append(label_offset(selection.text, labels, count) if labels else None)
        out.has_label.append(any(i.predicate == RDFS_LABEL for i in kept))
    return out


def render_all(context: PairContext, dataset_id: str, count: TokenCounter) -> Dict[Tuple[str, str], Rendered]:
    """Render every logical condition of a dataset on both sides."""
    rendered = {}
    for logical in (c for c in LOGICAL_CONDITIONS if c.dataset == dataset_id):
        selector = logical_selector(logical, count)
        for name, side in (("source", context.source), ("candidate", context.candidate)):
            rendered[(logical.name, name)] = render_side(selector, side, count, logical.budget is not None)
        print(f"  rendered {logical.name}", flush=True)
    return rendered


def gate_row(gate: str, subject: str, side: str, n: int, mismatches: int, detail: str = "") -> Dict[str, object]:
    """Format one gate outcome."""
    return {"gate": gate, "subject": subject, "side": side, "n_entities": n, "n_mismatch": mismatches,
            "status": "pass" if mismatches == 0 else "FAIL", "detail": detail}


def fixed_set_gates(rendered: Dict[Tuple[str, str], Rendered], dataset_id: str) -> List[Dict[str, object]]:
    """Check equal item-set fingerprints within every fixed-set group of a dataset."""
    rows = []
    for group in (g for g in FIXED_SET_GROUPS if g[0].startswith(f"{dataset_id}/")):
        for side in SIDES:
            columns = [rendered[(name, side)].set_fp for name in group]
            mismatches = sum(len(set(values)) > 1 for values in zip(*columns))
            rows.append(gate_row("fixed_set_identity", " = ".join(group), side, len(columns[0]), mismatches))
    return rows


def shared_gates(rendered: Dict[Tuple[str, str], Rendered], dataset_id: str) -> List[Dict[str, object]]:
    """Check that every shared logical condition renders its execution's exact text."""
    rows = []
    for logical in (c for c in LOGICAL_CONDITIONS if c.dataset == dataset_id and c.shared_reason):
        owner = next(c for c in LOGICAL_CONDITIONS if c.condition_id == logical.condition_id
                     and c.dataset == dataset_id and not c.shared_reason)
        for side in SIDES:
            a, b = rendered[(logical.name, side)].text_fp, rendered[(owner.name, side)].text_fp
            rows.append(gate_row("shared_execution_identity", f"{logical.name} = {owner.name}", side, len(a),
                                 sum(x != y for x, y in zip(a, b)), logical.shared_reason))
    return rows


def stored_fingerprints(results_dir: str, condition_id: str, dataset_id: str, pair_name: str) -> Dict[str, Dict]:
    """Load a stored sidecar's per-entity text fingerprints, both sides."""
    path = Path(results_dir) / condition_id / dataset_id / f"{pair_name}_entities.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    return sidecar.get("selection_fingerprints") or sidecar["sampled_fingerprints"]


def compare_fingerprints(stored: Dict[str, str], uris: List[str], fps: List[str], every: bool) -> Tuple[int, int]:
    """Return (compared, mismatched) entities; ``every`` requires each URI to be stored."""
    ours = dict(zip(uris, fps))
    keys = list(uris) if every else [u for u in stored if u in ours]
    missing = sum(u not in stored for u in keys) + sum(u not in ours for u in stored)
    return len(keys), missing + sum(stored.get(u) != ours[u] for u in keys if u in stored)


def reproduction_gates(rendered: Dict[Tuple[str, str], Rendered], context: PairContext,
                       dataset_id: str) -> List[Dict[str, object]]:
    """Check every canonically rendered execution against the stored #131/#128 texts."""
    rows = []
    for logical in (c for c in LOGICAL_CONDITIONS if c.dataset == dataset_id):
        target = reproduction_target(logical)
        if target is None:
            continue
        stored = stored_fingerprints(*target, dataset_id, context.pair.pair_name)
        for side, items in (("source", context.source), ("candidate", context.candidate)):
            n, bad = compare_fingerprints(stored[side], items.uris, rendered[(logical.name, side)].text_fp, True)
            rows.append(gate_row("reproduces_stored_texts", f"{logical.name} = {target[0]}/{target[1]}", side, n, bad))
    return rows


def budget_gates(rendered: Dict[Tuple[str, str], Rendered], dataset_id: str) -> List[Dict[str, object]]:
    """Check that no rendered text of a budgeted logical condition exceeds its budget."""
    rows = []
    for logical in (c for c in LOGICAL_CONDITIONS if c.dataset == dataset_id and c.budget is not None):
        for side in SIDES:
            tokens = rendered[(logical.name, side)].tokens
            rows.append(gate_row("within_budget", logical.name, side, len(tokens),
                                 sum(t > logical.budget for t in tokens), f"max {max(tokens)} <= {logical.budget}"))
    return rows


def label_rows(rendered: Dict[Tuple[str, str], Rendered], dataset_id: str) -> List[Dict[str, object]]:
    """Label retention and label position per logical condition and side."""
    rows = []
    for logical in (c for c in LOGICAL_CONDITIONS if c.dataset == dataset_id):
        for side in SIDES:
            r = rendered[(logical.name, side)]
            offsets = [o for o in r.label_offset if o is not None]
            with_label = sum(r.has_label)
            rows.append({"logical": logical.name, "side": side, "n_entities": len(r.has_label),
                         "n_with_label": with_label, "n_label_kept": len(offsets),
                         "label_retention": round(len(offsets) / with_label, 6) if with_label else "",
                         "median_label_offset": float(np.median(offsets)) if offsets else "",
                         "share_label_first": round(sum(o == 0 for o in offsets) / len(offsets), 6) if offsets else ""})
    return rows


def change_rows(rendered: Dict[Tuple[str, str], Rendered], context: PairContext,
                dataset_id: str) -> List[Dict[str, object]]:
    """Selection changes of coupled and selection-only conditions against the fixed core-first set."""
    role_map = get_role_map(ROLE_MAP)
    rows = []
    for budget in (b for d, b in B_BUDGETS if d == dataset_id):
        reference = f"{dataset_id}/B{budget}/F-uri"
        for name in ("C-uri", "C-label", "C-role", "X-label", "X-role"):
            logical = f"{dataset_id}/B{budget}/{name}"
            for side in SIDES:
                rows.append(_change_row(rendered[(reference, side)], rendered[(logical, side)], role_map,
                                        {"logical": logical, "reference": reference, "side": side}))
    return rows


def _change_row(ref: Rendered, other: Rendered, role_map: object, row: Dict[str, object]) -> Dict[str, object]:
    """Count differing sets, items gained and lost by role, tokens and label changes."""
    gained, lost = Counter(), Counter()
    for a, b in zip(ref.keys, other.keys):
        gained.update(role_map.role_of(p) for p, _t in b - a)
        lost.update(role_map.role_of(p) for p, _t in a - b)
    ref_label = [o is not None for o in ref.label_offset]
    other_label = [o is not None for o in other.label_offset]
    row.update(n_entities=len(ref.keys), n_set_differs=sum(a != b for a, b in zip(ref.set_fp, other.set_fp)),
               items_gained=json.dumps(dict(sorted(gained.items()))), items_lost=json.dumps(dict(sorted(lost.items()))),
               mean_tokens_reference=round(float(np.mean(ref.tokens)), 4), mean_tokens=round(float(np.mean(other.tokens)), 4),
               labels_lost=sum(a and not b for a, b in zip(ref_label, other_label)),
               labels_gained=sum(b and not a for a, b in zip(ref_label, other_label)))
    return row


def text_difference_rows(rendered: Dict[Tuple[str, str], Rendered], dataset_id: str) -> List[Dict[str, object]]:
    """Entities whose text differs from the group's canonically rendered member."""
    rows = []
    for group in (g for g in FIXED_SET_GROUPS if g[0].startswith(f"{dataset_id}/")):
        anchor = next(n for n in group if LOGICAL_BY_NAME[n].render == "uri")
        for name in (n for n in group if n != anchor):
            for side in SIDES:
                a, b = rendered[(anchor, side)].text_fp, rendered[(name, side)].text_fp
                rows.append({"group_anchor": anchor, "logical": name, "side": side, "n_entities": len(a),
                             "n_text_differs": sum(x != y for x, y in zip(a, b))})
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write dict rows to CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gates(dataset_ids: List[str], output_dir: Path) -> int:
    """Run every gate and diagnostic; return the number of failed gates."""
    count = token_counter()
    tables: Dict[str, List[Dict[str, object]]] = {k: [] for k in ("identity_gates", "label_retention",
                                                                 "selection_changes", "text_differences")}
    for dataset_id in dataset_ids:
        context = build_pair_context(load_dataset(dataset_id, Path("data"))[0], None, count, Path("data"))
        rendered = render_all(context, dataset_id, count)
        tables["identity_gates"] += (fixed_set_gates(rendered, dataset_id) + shared_gates(rendered, dataset_id)
                                     + reproduction_gates(rendered, context, dataset_id)
                                     + budget_gates(rendered, dataset_id))
        tables["label_retention"] += label_rows(rendered, dataset_id)
        tables["selection_changes"] += change_rows(rendered, context, dataset_id)
        tables["text_differences"] += text_difference_rows(rendered, dataset_id)
    for name, rows in tables.items():
        write_csv(output_dir / f"{name}.csv", rows)
    failed = [r for r in tables["identity_gates"] if r["status"] != "pass"]
    print(f"{len(tables['identity_gates'])} gate checks, {len(failed)} failed; tables in {output_dir}")
    for row in failed:
        print(f"FAIL {row['gate']} {row['subject']} {row['side']}: {row['n_mismatch']} of {row['n_entities']}")
    return len(failed)


_HISTORICAL_DIRS = (DEFAULT_RESULTS_DIR, CONTROLLED_RESULTS_DIR, VALIDATION_RESULTS_DIR, ROLE_RESULTS_DIR)


def stored_runs(dataset_id: str, pair_name: str) -> Iterable[Tuple[object, str]]:
    """Yield every registered ablation condition with a stored sidecar on a dataset, and its directory."""
    for condition in (c for c in ABLATION_CONDITIONS if dataset_id in c.datasets):
        for directory in _HISTORICAL_DIRS:
            if (Path(directory) / condition.condition_id / dataset_id / f"{pair_name}_entities.json").exists():
                yield condition, directory


def existing(dataset_ids: List[str]) -> int:
    """Recompute every stored ablation run's texts; return the number of mismatching runs."""
    count, failures, checked = token_counter(), 0, 0
    for dataset_id in dataset_ids:
        context = build_pair_context(load_dataset(dataset_id, Path("data"))[0], None, count, Path("data"))
        for condition, directory in stored_runs(dataset_id, context.pair.pair_name):
            selector = condition_selector(condition, count)
            sidecar = json.loads((Path(directory) / condition.condition_id / dataset_id
                                  / f"{context.pair.pair_name}_entities.json").read_text(encoding="utf-8"))
            every = "selection_fingerprints" in sidecar
            stored = sidecar["selection_fingerprints" if every else "sampled_fingerprints"]
            bad = 0
            for side_name, side in (("source", context.source), ("candidate", context.candidate)):
                texts = [selector.select(i, u, t).text for u, t, i in zip(side.uris, side.entity_types, side.items)]
                fps = [_text_fingerprint(t) for t in texts]
                bad += compare_fingerprints(stored[side_name], side.uris, fps, every)[1]
            checked += 1
            failures += bad > 0
            print(f"{'ok  ' if not bad else 'FAIL'} {dataset_id} {directory}{condition.condition_id}: "
                  f"{'all entities' if every else 'sampled entities'}, {bad} mismatches", flush=True)
    print(f"{checked} stored runs recomputed, {failures} with mismatches")
    return failures


def main() -> None:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser(description="Identity gates and text diagnostics of #134.")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gates")
    g.add_argument("--datasets", nargs="+", default=["D1", "D2"])
    g.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    e = sub.add_parser("existing")
    e.add_argument("--datasets", nargs="+", default=["D1", "D2"])
    args = parser.parse_args()
    failed = gates(args.datasets, Path(args.output_dir)) if args.command == "gates" else existing(args.datasets)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
