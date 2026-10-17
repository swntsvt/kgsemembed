"""Diagnostics behind the #131 write-up and the #128 review.

Three subcommands; none writes into a results tree.

``label-loss`` (tokenizer only)
    Recomputes every controlled-policy (P1-P7) selection, checks it against the
    stored ``selection_fingerprints`` and counts entities whose ``rdfs:label``
    the budget dropped, per side.  Shows which #128 contrasts mix eligibility
    or vocabulary with label loss.
``examples`` (loads M2)
    Re-ranks a seeded sample of D1 *test* sources under several ``PR-*`` arms,
    exactly as the runner does, checks every recomputed gold rank against the
    stored sidecar ``best_rank``, and prints source text, gold text, the
    leading wrong candidates and ranks for sources that the second arm moves.
    Descriptive only: nothing here is used to choose a mapping or a setting.
``probe`` (loads M2)
    Renders role subsets of D1 *validation* sources (never test sources) with
    the added role moved after the core items, or with ``FMAID`` values masked,
    to separate position from content.  Exploratory; not a registered arm.

Usage
-----
python scripts/predicate_role_diagnostics.py label-loss --datasets D1 D2
python scripts/predicate_role_diagnostics.py examples --n_sample 300 \\
    --pair PR-core-full PR-core-plus-syn-full
python scripts/predicate_role_diagnostics.py probe --n_sample 400
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predicate_role_inventory import label_dropped  # noqa: E402

from kgsemembed.datasets import load_dataset  # noqa: E402
from kgsemembed.embeddings import EmbeddingEncoder, load_sentence_transformer  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import (  # noqa: E402
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    AblationCondition,
)
from kgsemembed.pipeline.ppas_ablation import (  # noqa: E402
    PairContext,
    build_pair_context,
    condition_selector,
    selection_fingerprints,
)
from kgsemembed.verbalisation.predicate_roles import NON_CORE_ROLES, ROLES, get_role_map  # noqa: E402
from kgsemembed.verbalisation.selection import ITEM_SEPARATOR, TokenCounter, make_token_counter  # noqa: E402

_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
ROLE_DIR = "data/results_predicate_roles"
POLICY_DIRS = {("D1", 48): "data/results_ppas_controlled", ("D1", 32): "data/results_ppas_validation",
               ("D1", 64): "data/results_ppas_validation", ("D2", 48): "data/results_ppas_validation"}
ALL_ROLES = frozenset(ROLES)
PROBES: Dict[str, dict] = {
    "A-core": dict(keep={"core"}),
    "C+syn": dict(keep={"core", "synonym"}),
    "C+syn/syn-last": dict(keep={"core", "synonym"}, last={"synonym"}),
    "C+id": dict(keep={"core", "identifier"}),
    "C+id/id-last": dict(keep={"core", "identifier"}, last={"identifier"}),
    "C+id/masked": dict(keep={"core", "identifier"}, mask_id=True),
    "A0": dict(keep=ALL_ROLES),
    "R-id": dict(keep=ALL_ROLES - {"identifier"}),
    "A0/id-masked": dict(keep=ALL_ROLES, mask_id=True),
    "A0/annotations-last": dict(keep=ALL_ROLES, last=set(NON_CORE_ROLES)),
    "R-id/annotations-last": dict(keep=ALL_ROLES - {"identifier"}, last=set(NON_CORE_ROLES)),
}
PROBE_CONTRASTS = (("A-core", "C+syn"), ("A-core", "C+syn/syn-last"), ("A-core", "C+id"),
                   ("A-core", "C+id/id-last"), ("A-core", "C+id/masked"), ("A0", "R-id"),
                   ("A0", "A0/id-masked"), ("A0/annotations-last", "R-id/annotations-last"),
                   ("A0", "A0/annotations-last"), ("A0", "A-core"), ("A0/annotations-last", "A-core"))


def token_counter() -> TokenCounter:
    """Return the M2 tokenizer's encoded-length counter at the pinned revision."""
    return make_token_counter(AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", revision=_REVISION))


def pair_context(dataset_id: str, encoder: Optional[EmbeddingEncoder], count: TokenCounter) -> PairContext:
    """Build the runner's pair context (candidates and items) for a dataset's only pair."""
    return build_pair_context(load_dataset(dataset_id, Path("data"))[0], encoder, count, Path("data"))


def load_encoder() -> EmbeddingEncoder:
    """Load M2 as the runner does and insist on the MPS device."""
    model, info = load_sentence_transformer(ABLATION_MODEL_KEY)
    if not info["device"].startswith("mps"):
        raise RuntimeError(f"Expected an MPS device, got {info['device']}.")
    return EmbeddingEncoder(ABLATION_MODEL_KEY, model)


def sidecar(root: str, condition_id: str, dataset_id: str) -> dict:
    """Read one run's entity sidecar."""
    return json.load(open(glob.glob(f"{root}/{condition_id}/{dataset_id}/*_entities.json")[0]))


def bootstrap(diffs: np.ndarray, n_resamples: int = 2000) -> Tuple[float, float, float]:
    """Return the mean and 95% percentile interval of paired per-source differences."""
    draws = np.random.default_rng(42).integers(0, len(diffs), (n_resamples, len(diffs)))
    means = diffs[draws].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# label-loss
# ---------------------------------------------------------------------------


def policy_conditions(dataset_id: str) -> List[AblationCondition]:
    """Return the controlled-policy conditions run on a dataset, in registry order."""
    return [c for c in ABLATION_CONDITIONS
            if c.policy_label and dataset_id in c.datasets and (dataset_id, c.token_budget) in POLICY_DIRS]


def side_label_loss(condition: AblationCondition, context: PairContext, count: TokenCounter,
                    stored: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[int, int, int]]:
    """Return (sampled, label lost, fingerprint mismatches) per side for one condition."""
    selector, out = condition_selector(condition, count), {}
    for name, side in (("source", context.source), ("candidate", context.candidate)):
        chosen = [selector.select(i, u, t) for u, t, i in zip(side.uris, side.entity_types, side.items)]
        prints = selection_fingerprints(side, chosen)
        lost = sum(label_dropped(s, side.items[k]) for k, s in enumerate(chosen))
        out[name] = (sum(s.sampling_invoked for s in chosen), lost,
                     sum(prints[u] != stored[name][u] for u in prints))
    return out


def label_loss(dataset_ids: List[str]) -> None:
    """Print label loss and fingerprint reproduction for every controlled policy."""
    count = token_counter()
    print("| Dataset | Budget | Policy | Condition | Side | Sampled | Label dropped | Fingerprint mismatches |")
    print("|---|---|---|---|---|---|---|---|")
    for dataset_id in dataset_ids:
        context = pair_context(dataset_id, None, count)
        for c in policy_conditions(dataset_id):
            stored = dict(sidecar(POLICY_DIRS[(dataset_id, c.token_budget)], c.condition_id, dataset_id)
                          ["selection_fingerprints"])
            for side, (sampled, lost, mismatched) in side_label_loss(c, context, count, stored).items():
                print(f"| {dataset_id} | {c.token_budget} | {c.policy_label} | `{c.condition_id}` | {side} | "
                      f"{sampled} | {lost} | {mismatched} |", flush=True)


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------


def rank_source(encoder: EmbeddingEncoder, source_text: str, candidates: List[str],
                candidate_texts: List[str]) -> List[Tuple[str, float, str]]:
    """Rank candidates by dot product, ties by URI, as ``run_experiment._rank_candidates``."""
    vector = encoder.encode_batch([source_text], role="source", show_progress=False)[0]
    matrix = encoder.encode_batch(candidate_texts, role="candidate", show_progress=False)
    scores = matrix @ vector
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
    return [(candidates[i], round(float(scores[i]), 4), candidate_texts[i]) for i in order]


_URI_INDEX: Dict[int, Dict[str, int]] = {}


def uri_index(side: object) -> Dict[str, int]:
    """Return (and cache) the position of every URI on a side."""
    if id(side) not in _URI_INDEX:
        _URI_INDEX[id(side)] = {uri: k for k, uri in enumerate(side.uris)}
    return _URI_INDEX[id(side)]


def arm_record(condition: AblationCondition, context: PairContext, count: TokenCounter,
               encoder: EmbeddingEncoder, source: str) -> dict:
    """Rebuild one source's texts under an arm, rank its candidates and find the gold rank."""
    selector = condition_selector(condition, count)

    def text_of(side: object, uri: str) -> str:
        k = uri_index(side)[uri]
        return selector.select(side.items[k], uri, side.entity_types[k]).text

    candidates = context.candidates[source]
    source_text = text_of(context.source, source)
    ranked = rank_source(encoder, source_text, candidates, [text_of(context.candidate, c) for c in candidates])
    return {"source_text": source_text, "ranked": ranked}


def gold_rank(ranked: List[Tuple[str, float, str]], gold: List[str]) -> Optional[int]:
    """Return the best rank of any gold target, or None when none is a candidate."""
    ranks = [k + 1 for k, (uri, _, _) in enumerate(ranked) if uri in gold]
    return min(ranks) if ranks else None


def print_example(source: str, gold: List[str], records: Dict[str, dict]) -> None:
    """Print one source's text, gold text, leading wrong candidates and rank under each arm."""
    print(f"\n#### `{source}` (gold {', '.join(g.rsplit('/', 1)[-1] for g in gold)})\n")
    for condition_id, record in records.items():
        ranked = record["ranked"]
        print(f"- **{condition_id}**: gold rank {gold_rank(ranked, gold)}\n  - source: `{record['source_text']}`")
        for uri, score, text in [r for r in ranked if r[0] in gold]:
            print(f"  - gold {uri.rsplit('/', 1)[-1]} ({score}): `{text}`")
        for uri, score, text in [r for r in ranked if r[0] not in gold][:2]:
            print(f"  - wrong {uri.rsplit('/', 1)[-1]} ({score}): `{text}`")


def examples(condition_ids: List[str], n_sample: int, n_show: int) -> None:
    """Re-rank sampled D1 test sources, verify ranks against the sidecars, print moved examples."""
    conditions = {c.condition_id: c for c in ABLATION_CONDITIONS if c.condition_id in condition_ids}
    count, encoder = token_counter(), load_encoder()
    context = pair_context("D1", encoder, count)
    stored = {cid: {t["source"]: t for t in sidecar(ROLE_DIR, cid, "D1")["test_sources"]} for cid in condition_ids}
    pool = sorted(s for s in stored[condition_ids[0]] if context.candidates.get(s))
    sample = sorted(random.Random(42).sample(pool, min(n_sample, len(pool))))
    mismatches, moved = 0, []
    for source in sample:
        gold = stored[condition_ids[0]][source]["gold_targets"]
        records = {cid: arm_record(conditions[cid], context, count, encoder, source) for cid in condition_ids}
        ranks = [gold_rank(records[cid]["ranked"], gold) for cid in condition_ids]
        mismatches += sum(r != stored[cid][source]["best_rank"] for r, cid in zip(ranks, condition_ids))
        if ranks[0] == 1 and ranks[-1] != 1:
            moved.append((source, gold, records))
    print(f"Sampled test sources: {len(sample)}; ranks recomputed: {len(sample) * len(condition_ids)}; "
          f"mismatches with stored sidecars: {mismatches}; rank 1 under {condition_ids[0]} but not "
          f"{condition_ids[-1]}: {len(moved)}")
    for source, gold, records in moved[:n_show]:
        print_example(source, gold, records)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def render(items: list, keep: set, last: frozenset | set = frozenset(), mask_id: bool = False) -> str:
    """Join the kept items' texts, ``last`` roles after the rest, optionally masking FMAID values."""
    roles = get_role_map("roles_v1")
    kept = [i for i in items if roles.role_of(i.predicate) in keep]
    ordered = [i for i in kept if roles.role_of(i.predicate) not in last] + \
              [i for i in kept if roles.role_of(i.predicate) in last]
    return ITEM_SEPARATOR.join(re.sub(r"^FMAID: .*$", "FMAID: 0", i.text) if mask_id else i.text for i in ordered)


def validation_gold(context: PairContext) -> Dict[str, set]:
    """Map each validation-population source with candidates to its gold targets."""
    test_sources = {s for s, _ in context.pair.test_refs}
    gold = defaultdict(set)
    for source, target in context.pair.val_refs:
        if source not in test_sources and context.candidates.get(source):
            gold[source].add(target)
    return gold


def probe_rr(encoder: EmbeddingEncoder, context: PairContext, variant: dict, source: str, gold: set) -> float:
    """Return a validation source's reciprocal rank under one rendering variant."""
    source_items = context.source.items[uri_index(context.source)[source]]
    index = uri_index(context.candidate)
    candidates = context.candidates[source]
    texts = [render(context.candidate.items[index[c]], **variant) for c in candidates]
    rank = gold_rank(rank_source(encoder, render(source_items, **variant), candidates, texts), list(gold))
    return 1.0 / rank if rank else 0.0


def probe(n_sample: int, variants: List[str]) -> None:
    """Print MRR per rendering variant and paired contrasts on sampled D1 validation sources."""
    count, encoder = token_counter(), load_encoder()
    context = pair_context("D1", encoder, count)
    gold = validation_gold(context)
    sample = sorted(random.Random(7).sample(sorted(gold), min(n_sample, len(gold))))
    rr = {name: np.array([probe_rr(encoder, context, v, s, gold[s]) for s in sample]) for name, v in PROBES.items()
          if name in variants}
    print(f"D1 validation sources: {len(sample)} (seed 7)\n\n| Variant | MRR |\n|---|---|")
    print("\n".join(f"| {name} | {values.mean():.4f} |" for name, values in rr.items()))
    print("\n| Contrast | Δ MRR [95% CI] |\n|---|---|")
    for a, b in [(a, b) for a, b in PROBE_CONTRASTS if a in rr and b in rr]:
        mean, low, high = bootstrap(rr[b] - rr[a])
        print(f"| {b} − {a} | {mean:+.4f} [{low:+.4f}, {high:+.4f}] |")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostics of the #128/#131 ablations.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("label-loss").add_argument("--datasets", nargs="+", default=["D1", "D2"])
    ex = sub.add_parser("examples")
    ex.add_argument("--pair", nargs="+", required=True, help="Condition IDs; first is the reference arm.")
    ex.add_argument("--n_sample", type=int, default=300)
    ex.add_argument("--n_show", type=int, default=4)
    pr = sub.add_parser("probe")
    pr.add_argument("--n_sample", type=int, default=400)
    pr.add_argument("--variants", nargs="+", default=list(PROBES), choices=list(PROBES))
    return parser


def main() -> None:
    """Dispatch to one diagnostic."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    if args.command == "label-loss":
        label_loss(args.datasets)
    elif args.command == "examples":
        examples(args.pair, args.n_sample, args.n_show)
    else:
        probe(args.n_sample, args.variants)


if __name__ == "__main__":
    main()
