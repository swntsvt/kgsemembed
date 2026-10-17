"""Write the #134 freeze record from validation results only.

Applies the frozen selection rule (:func:`kgsemembed.evaluation.order_selection.choose_order`)
to the Block S A0 cells on the D2 validation sources, records the D1 validation
consistency check on the 751 sources the #131 probe never touched (and the
400 probed ones separately), and writes ``artifacts/issue-134/serialization_freeze.json``.
It reads nothing from the test population, refuses to overwrite an existing
record, and refuses validation results from a dirty tree or mixed commits.

The intervals in the record describe validation uncertainty only: they are
not adjusted for selection, the chosen order's validation estimate is
optimistically biased by having been chosen, and the primary inferential
claims rest only on the test comparisons.

Usage
-----
python scripts/freeze_serialization_order.py \\
    --results_dir data/results_serialization/ --output artifacts/issue-134/serialization_freeze.json
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

from kgsemembed.evaluation.order_selection import (
    PRECEDENCE,
    REFERENCE_ORDER,
    SELECTION_REPLICATES,
    SELECTION_SEED,
    choose_order,
    paired_interval,
)
from kgsemembed.pipeline.serialization_conditions import so_condition_id
from kgsemembed.verbalisation.serialization import ORDER_DEFINITIONS, SERIALIZATION_ORDERS, order_definitions_sha256

PAIR_NAMES = {"D1": "d1_snomed_fma", "D2": "d2_anatomy"}
EXPECTED_D2_SOURCES = 299
PROBE_SEED, PROBE_SIZE, PROBE_SHA256_PREFIX = 7, 400, "a773ea69dc13"
DEFAULT_OUTPUT = Path("artifacts/issue-134/serialization_freeze.json")
SECONDARY = (
    "every order − uri within each Block S row (MRR, Recall@1)",
    "A-core − A0, C+syn − A-core and C+id / C+def − A-core within each order",
    "Block B: serialization at the reference set and at the order's own set, selection at uri and at the "
    "order's rendering, combined policy, interaction, and the difference from the no-budget serialization effect",
    "every contrast split into sources whose own or gold-target text differs between the two cells, and the rest",
)


def run_paths(results_dir: Path, dataset_id: str, render: str) -> Tuple[Path, Path]:
    """Return the validation result and sidecar paths of a Block S A0 cell."""
    directory = results_dir / "validation" / so_condition_id("A0", None, "all", render) / dataset_id
    return (directory / f"{PAIR_NAMES[dataset_id]}_results.json",
            directory / f"{PAIR_NAMES[dataset_id]}_entities.json")


def load_cell(results_dir: Path, dataset_id: str, render: str) -> Tuple[Dict, List[Dict]]:
    """Load a cell's result and its validation reference-source records."""
    result_path, sidecar_path = run_paths(results_dir, dataset_id, render)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["serialization"]["population"] != "validation":
        raise ValueError(f"{result_path} is not a validation-stage result.")
    records = json.loads(sidecar_path.read_text(encoding="utf-8"))["reference_sources"]["validation"]
    return result, records


def per_source(records: List[Dict]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return sources, reciprocal ranks and Recall@1 hits, sorted by source URI."""
    rows = sorted(records, key=lambda r: r["source"])
    ranks = [r["best_rank"] for r in rows]
    rr = np.array([1.0 / k if k else 0.0 for k in ranks])
    return [r["source"] for r in rows], rr, np.array([1.0 if k == 1 else 0.0 for k in ranks])


def load_orders(results_dir: Path, dataset_id: str) -> Tuple[List[str], Dict, Dict, List[Dict]]:
    """Load all four renders of a dataset's A0 row, aligned by source; also return the results."""
    sources, rr, hits, results = None, {}, {}, []
    for render in SERIALIZATION_ORDERS:
        result, records = load_cell(results_dir, dataset_id, render)
        these, rr[render], hits[render] = per_source(records)
        if sources is not None and these != sources:
            raise ValueError(f"{dataset_id}: validation sources differ between renders.")
        sources = these
        results.append(result)
    return sources, rr, hits, results


def probed_sample(ranked_sources: List[str]) -> List[str]:
    """Recreate the #131 probe's seed-7 sample of 400 D1 validation sources and check its hash."""
    sample = sorted(random.Random(PROBE_SEED).sample(sorted(ranked_sources), PROBE_SIZE))
    digest = hashlib.sha256("\n".join(sample).encode("utf-8")).hexdigest()
    if not digest.startswith(PROBE_SHA256_PREFIX):
        raise ValueError(f"Probed sample hash {digest[:12]} != {PROBE_SHA256_PREFIX}: the population differs.")
    return sample


def d1_consistency(results_dir: Path, chosen: str) -> Dict[str, object]:
    """Chosen − uri on the untouched and the probed D1 validation sources, reported separately."""
    _, records = load_cell(results_dir, "D1", REFERENCE_ORDER)
    ranked = [r["source"] for r in records if r["top"]]
    probed = set(probed_sample(ranked))
    sources, rr, hits, _ = load_orders(results_dir, "D1")
    masks = {"untouched": np.array([s not in probed for s in sources]), "probed": np.array([s in probed for s in sources])}
    out = {}
    for part, mask in masks.items():
        out[part] = {"n_sources": int(mask.sum()),
                     "mrr_minus_uri": paired_interval(rr[chosen][mask], rr[REFERENCE_ORDER][mask]),
                     "recall_at_1_minus_uri": paired_interval(hits[chosen][mask], hits[REFERENCE_ORDER][mask])}
    out["probed_sample_sha256"] = hashlib.sha256("\n".join(sorted(probed)).encode("utf-8")).hexdigest()
    out["interpretation"] = "exploratory consistency check; cannot change the choice"
    return out


def provenance(results: List[Dict], results_dir: Path) -> Dict[str, object]:
    """Check one clean commit across the validation results and fingerprint the files used."""
    commits = {r["git_commit"] for r in results}
    if len(commits) != 1 or any(r["git_dirty"] for r in results):
        raise ValueError(f"Validation results must come from one clean commit, got {sorted(commits)}.")
    files = {}
    for dataset_id in PAIR_NAMES:
        for render in SERIALIZATION_ORDERS:
            for path in run_paths(results_dir, dataset_id, render):
                files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"validation_commit": commits.pop(), "validation_files_sha256": files}


def build_record(results_dir: Path) -> Dict[str, object]:
    """Apply the rule to D2 validation and assemble the freeze record."""
    sources, rr, hits, results = load_orders(results_dir, "D2")
    if len(sources) != EXPECTED_D2_SOURCES:
        raise ValueError(f"Expected {EXPECTED_D2_SOURCES} D2 validation sources, got {len(sources)}.")
    selection = choose_order(rr)
    for order in PRECEDENCE:
        selection["candidates"][order]["recall_at_1"] = float(hits[order].mean())
    selection["uri_recall_at_1"] = float(hits[REFERENCE_ORDER].mean())
    chosen = selection["chosen"]
    _, _, _, d1_results = load_orders(results_dir, "D1")
    return {
        "issue": 134, "stage": "frozen", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chosen": chosen,
        "rule": {"population": "D2 validation reference sources, Block S A0 (no budget)", "n_sources": len(sources),
                 "candidates": list(PRECEDENCE), "reference": REFERENCE_ORDER, "precedence": list(PRECEDENCE),
                 "replicates": SELECTION_REPLICATES, "seed": SELECTION_SEED,
                 "interval": "95% percentile, shared index matrix", "eligible_if": "lower bound of MRR(best) − MRR(o) <= 0"},
        "d2_validation": selection,
        "d1_validation_exploratory": d1_consistency(results_dir, chosen),
        "status_of_intervals": ("descriptive validation uncertainty, not adjusted for selection; the chosen order's "
                                "validation estimate is optimistically biased; no inferential claim"),
        "primary_contrasts": [f"{d}/S/A0/{chosen} − {d}/S/A0/{REFERENCE_ORDER} (MRR, Recall@1; 95% and Bonferroni "
                              "97.5% intervals, 2,000 replicates, seed 42)" for d in ("D1", "D2")],
        "secondary_contrasts": list(SECONDARY),
        "order_definitions": ORDER_DEFINITIONS, "order_definitions_sha256": order_definitions_sha256(),
        **provenance(results + d1_results, results_dir),
    }


def main() -> None:
    """Parse arguments and write the record once."""
    parser = argparse.ArgumentParser(description="Freeze the #134 serialization order from validation results.")
    parser.add_argument("--results_dir", default="data/results_serialization/")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        sys.exit(f"{output} exists; a freeze record is written once and never overwritten.")
    record = build_record(Path(args.results_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen order: {record['chosen']} (E = {record['d2_validation']['eligible']}); wrote {output}")


if __name__ == "__main__":
    main()
