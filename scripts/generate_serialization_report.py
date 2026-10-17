"""Artifact bundle and reports of the serialization experiment (#134).

Subcommands
-----------
``export``
    Results (``data/results_serialization/<stage>/``) and the tokenizer-only
    inventory → the tracked bundle ``artifacts/issue-134/``: result JSONs,
    per-source outcome tables, the logical-condition map, diagnostics,
    script-chosen ranking examples, provenance and a SHA-256 manifest (which
    also lists the local-only sidecars).  Every tracked file is URI-only: no
    entity text.  Every result must belong to the exported stage (and, for the
    test stage, to the committed freeze); stage files and shared files are
    written once and never replaced by different bytes, so a test export
    cannot overwrite or mix with the validation bundle.
``report``
    The bundle alone → ``report_<stage>.md`` and ``figures/<stage>_*`` (PNG
    plus the exact series as CSV).  Reads nothing outside the bundle and writes
    no timestamp, so a rebuild is byte-identical.
``verify-rebuild``
    Copies the bundle to a temporary directory, rebuilds the report there from
    a working directory without ``data/``, and compares every output byte for byte.
``examples-local``
    Renders the readable text of every example (source, gold targets and top
    candidates under ``uri`` and the frozen order) into the local, untracked
    results directory; nothing it writes enters the bundle.
``check-text``
    Scans every tracked bundle file for dataset entity text (labels and other
    item values of D1/D2, 12 characters or longer); exit status 1 on a hit.
``gate-ranks``
    Test stage, after the freeze: for each of the 13 canonically rendered
    executions with a stored #131/#128 run (:func:`expected_rank_targets`),
    compares every stored test reference source with the new test outcome
    of the same source URI: best gold rank, gold-target set, and the text
    fingerprint of the source and of every gold target.  A missing baseline
    file, a missing or extra source, an empty comparison, or any difference
    fails the gate (exit status 1).  Writes ``diagnostics/rank_gate.json``
    with the SHA-256 of every report input; ``report --stage test`` refuses to
    run unless that record passed and every digest still matches.

The test report's primary contrasts are frozen order − ``uri`` on D1/S/A0 and
D2/S/A0 (MRR and Recall@1), with 95% and Bonferroni 97.5% intervals (two
datasets per metric).

Intervals are paired source-level bootstraps (2,000 replicates, a fresh
``default_rng(42)`` per contrast, 95% percentile), never pooled across
datasets.  D1 validation is exploratory and reported in two parts that are
never pooled: the 400 sources the #131 probe used and the 751 it did not.

Usage
-----
python scripts/generate_serialization_report.py export --stage validation
python scripts/generate_serialization_report.py report --stage validation
python scripts/generate_serialization_report.py verify-rebuild --stage validation
python scripts/generate_serialization_report.py check-text
python scripts/generate_serialization_report.py examples-local --stage validation
python scripts/generate_serialization_report.py gate-ranks
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_serialization_order import PAIR_NAMES, probed_sample  # noqa: E402

from kgsemembed.evaluation.order_selection import (  # noqa: E402
    REFERENCE_ORDER,
    REPORT_REPLICATES,
    REPORT_SEED,
    index_matrix,
    paired_interval,
)
from kgsemembed.pipeline.serialization_conditions import (  # noqa: E402
    B_BUDGETS,
    LOGICAL_CONDITIONS,
    S_REPRESENTATIONS,
)
from kgsemembed.verbalisation.serialization import SERIALIZATION_ORDERS  # noqa: E402

ARTIFACTS = Path("artifacts/issue-134")
RESULTS = Path("data/results_serialization")
INVENTORY = RESULTS / "inventory"
DIAGNOSTICS = ("identity_gates", "label_retention", "selection_changes", "text_differences")
OUTCOME_COLUMNS = ("stage", "dataset", "condition_id", "population", "source", "ranked", "n_gold", "best_rank",
                   "rr", "hit1", "hit5", "hit10", "source_text_fp", "source_set_fp", "gold_text_fps", "gold_set_fps")
CONDITION_COLUMNS = ("name", "dataset", "block", "representation", "budget", "selection", "render", "condition_id",
                     "shared_reason")
NON_URI = tuple(o for o in SERIALIZATION_ORDERS if o != "uri")
FREEZE_NAME = "serialization_freeze.json"
GATE_NAME = "rank_gate.json"
EXAMPLE_COLUMNS = ("stage", "dataset", "direction", "source", "gold_targets", "rank_uri", "rank_frozen",
                   "top_uri", "top_frozen")
EXAMPLES_PER_DIRECTION = 5
BONFERRONI_FAMILY = 2
UNENCODED_FINGERPRINTS = ("", str(None))
Cell = Dict[str, Dict[str, object]]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def outcome_rows(stage: str, dataset_id: str, condition_id: str, sidecar: Dict) -> List[Dict[str, object]]:
    """Flatten a sidecar's reference-source records into outcome rows."""
    rows = []
    for population, records in sorted(sidecar["reference_sources"].items()):
        for r in records:
            rank = r["best_rank"]
            gold = sorted(r["gold_fp"].items())
            rows.append({"stage": stage, "dataset": dataset_id, "condition_id": condition_id, "population": population,
                         "source": r["source"], "ranked": int(bool(r["top"])), "n_gold": len(r["gold_targets"]),
                         "best_rank": rank if rank else "", "rr": f"{1.0 / rank:.10f}" if rank else "0",
                         "hit1": int(rank == 1), "hit5": int(bool(rank) and rank <= 5),
                         "hit10": int(bool(rank) and rank <= 10),
                         "source_text_fp": r["source_fp"][0] or "", "source_set_fp": r["source_fp"][1] or "",
                         "gold_text_fps": ";".join(f"{t}={fp[0]}" for t, fp in gold),
                         "gold_set_fps": ";".join(f"{t}={fp[1]}" for t, fp in gold)})
    return rows


def csv_bytes(columns: Iterable[str], rows: List[Dict[str, object]]) -> bytes:
    """Return rows as plain CSV bytes."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def gzip_bytes(data: bytes) -> bytes:
    """Gzip with a zero mtime and no file name, so identical input gives identical bytes."""
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(data)
    return raw.getvalue()


def write_once(path: Path, data: bytes) -> None:
    """
    Write a bundle file that may only ever hold these bytes.

    Parameters
    ----------
    path : Path
        Target file.
    data : bytes
        Content.

    Raises
    ------
    FileExistsError
        If the file exists with different content.
    """
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"{path} exists with different content; the bundle never replaces a file.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_gzip_csv(path: Path, columns: Iterable[str], rows: List[Dict[str, object]]) -> None:
    """Write rows as a gzip CSV with a zero mtime, so identical rows give identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip_bytes(csv_bytes(columns, rows)))


def write_csv(path: Path, columns: Iterable[str], rows: List[Dict[str, object]]) -> None:
    """Write rows as a plain CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes(columns, rows))


def expected_runs(stage_dir: Path) -> List[Tuple[str, str, Path, Path]]:
    """Return (condition, dataset, result, sidecar) for every execution; raise if any is missing."""
    runs, missing = [], []
    for condition_id, dataset_id in dict.fromkeys((c.condition_id, c.dataset) for c in LOGICAL_CONDITIONS):
        directory = stage_dir / condition_id / dataset_id
        result = directory / f"{PAIR_NAMES[dataset_id]}_results.json"
        sidecar = directory / f"{PAIR_NAMES[dataset_id]}_entities.json"
        (runs if result.exists() and sidecar.exists() else missing).append((condition_id, dataset_id, result, sidecar))
    if missing:
        raise FileNotFoundError(f"{len(missing)} executions missing, e.g. {missing[0][:2]}.")
    return runs


def provenance(stage: str, results: List[Dict]) -> Dict[str, object]:
    """Summarise the commit, model, data and configuration behind a stage's results."""
    first = results[0]
    commits = sorted({r["git_commit"] for r in results})
    return {
        "stage": stage, "n_runs": len(results), "git_commits": commits,
        "any_dirty": any(r["git_dirty"] for r in results), "model_id": first["model_id"],
        "hf_revision": first["hf_revision"], "tokenizer_revision": first["ablation"]["tokenizer_revision"],
        "dataset_files_sha256": {r["dataset_id"]: r["dataset_files_sha256"] for r in results},
        "candidates_sha256": {r["dataset_id"]: r["candidates_sha256"] for r in results},
        "role_map_sha256": first["ablation"]["role_map"]["config_hash"],
        "order_definitions_sha256": first["serialization"]["order_definitions_sha256"],
        "versions": first["versions"], "evaluation_protocol": first["evaluation_protocol"],
        "run_timestamps": [min(r["run_timestamp"] for r in results), max(r["run_timestamp"] for r in results)],
        "freeze": first["serialization"]["freeze"],
        "command": f"python -m kgsemembed.pipeline.serialization_order --population {stage} --datasets D1 D2 "
                   "--data_dir data/ --results_dir data/results_serialization/",
    }


def check_stage(stage: str, results: List[Dict], artifacts: Path) -> None:
    """
    Refuse results of another stage, or test results not written under the committed freeze.

    Parameters
    ----------
    stage : str
        Stage being exported.
    results : List[Dict]
        Its result payloads.
    artifacts : Path
        Bundle root (holds the freeze record and the validation provenance).

    Raises
    ------
    ValueError
        On a population, freeze or run-signature mismatch.
    """
    if any(r["serialization"]["population"] != stage for r in results):
        raise ValueError(f"Some results are not {stage}-stage results.")
    if stage == "validation":
        return
    freeze_sha = sha256_of(artifacts / FREEZE_NAME)
    if any((r["serialization"]["freeze"] or {}).get("sha256") != freeze_sha for r in results):
        raise ValueError("Some test results were not written under the committed freeze record.")
    validation = json.loads((artifacts / "PROVENANCE_validation.json").read_text(encoding="utf-8"))
    test = provenance(stage, results)
    keys = ("model_id", "hf_revision", "tokenizer_revision", "dataset_files_sha256", "candidates_sha256",
            "role_map_sha256", "order_definitions_sha256", "versions", "evaluation_protocol")
    if any(validation[k] != test[k] for k in keys):
        raise ValueError(f"Test run signature differs from validation: {[k for k in keys if validation[k] != test[k]]}.")


def check_complete(stage: str, results: List[Dict], sidecars: Dict[Tuple[str, str], Dict]) -> None:
    """
    Refuse a partial, smoke, mixed-commit or dirty stage.

    Every run must come from one clean commit without a smoke restriction;
    each sidecar must hold a record for every reference source its result
    counts; and every execution of a dataset must cover the same sources.

    Parameters
    ----------
    stage : str
        Stage being exported.
    results : List[Dict]
        Result payloads, in the order of ``sidecars``.
    sidecars : Dict[Tuple[str, str], Dict]
        Sidecars by (dataset, condition).

    Raises
    ------
    ValueError
        On any incompleteness.
    """
    if len({r["git_commit"] for r in results}) != 1 or any(r["git_dirty"] for r in results):
        raise ValueError(f"{stage} results must come from one clean commit.")
    if any(r["ablation"].get("smoke_max_sources") is not None for r in results):
        raise ValueError(f"{stage} results include a smoke run.")
    sources: Dict[str, set] = {}
    for result, ((dataset_id, condition_id), sidecar) in zip(results, sidecars.items()):
        records = sidecar["reference_sources"][stage]
        if _expected_records(stage, result) != _counted_records(stage, records):
            raise ValueError(f"{condition_id}/{dataset_id}: sidecar does not cover every {stage} reference.")
        if sources.setdefault(dataset_id, {r["source"] for r in records}) != {r["source"] for r in records}:
            raise ValueError(f"{condition_id}/{dataset_id}: {stage} sources differ from other executions.")


def _expected_records(stage: str, result: Dict) -> int:
    """References (test) or reference sources (validation) the result counts."""
    population = result["population"]
    return population["n_test_refs"] if stage == "test" else population["n_val_reference_sources"]


def _counted_records(stage: str, records: List[Dict]) -> int:
    """The same count, from the sidecar's records."""
    return sum(len(r["gold_targets"]) for r in records) if stage == "test" else len(records)


def export(stage: str, results_dir: Path, inventory_dir: Path, artifacts: Path) -> None:
    """Write a stage's tracked bundle from its results and the inventory, never replacing a file."""
    runs = expected_runs(results_dir / stage)
    results = [json.loads(r.read_text(encoding="utf-8")) for *_, r, _s in runs]
    sidecars = {(d, c): json.loads(sc.read_text(encoding="utf-8")) for c, d, _r, sc in runs}
    check_stage(stage, results, artifacts)
    check_complete(stage, results, sidecars)
    outcomes = defaultdict(list)
    for condition_id, dataset_id, result_path, _sidecar in runs:
        write_once(artifacts / stage / "results" / condition_id / dataset_id / result_path.name,
                   result_path.read_bytes())
        outcomes[dataset_id] += outcome_rows(stage, dataset_id, condition_id, sidecars[(dataset_id, condition_id)])
    for dataset_id, rows in sorted(outcomes.items()):
        rows.sort(key=lambda r: (r["condition_id"], r["population"], r["source"]))
        write_once(artifacts / "outcomes" / f"{stage}_{dataset_id}.csv.gz", gzip_bytes(csv_bytes(OUTCOME_COLUMNS, rows)))
    write_once(artifacts / "conditions.csv", csv_bytes(CONDITION_COLUMNS, [
        {k: getattr(c, k) if getattr(c, k) is not None else "" for k in CONDITION_COLUMNS} for c in LOGICAL_CONDITIONS]))
    for name in DIAGNOSTICS:
        write_once(artifacts / "diagnostics" / f"{name}.csv", (inventory_dir / f"{name}.csv").read_bytes())
    write_once(artifacts / "examples" / f"{stage}_examples.csv",
               csv_bytes(EXAMPLE_COLUMNS, example_rows(stage, frozen_order(artifacts), sidecars)))
    write_once(artifacts / f"PROVENANCE_{stage}.json",
               (json.dumps(provenance(stage, results), indent=2) + "\n").encode("utf-8"))
    write_manifest(artifacts, [s for *_, s in runs])
    print(f"Exported {len(runs)} {stage} runs to {artifacts}")


def frozen_order(artifacts: Path) -> str:
    """Return the order chosen in the bundle's freeze record."""
    return json.loads((artifacts / FREEZE_NAME).read_text(encoding="utf-8"))["chosen"]


def a0_condition(dataset_id: str, order: str) -> str:
    """Return the execution of the Block S A0 cell rendered in ``order``."""
    return next(c.condition_id for c in LOGICAL_CONDITIONS if c.name == f"{dataset_id}/S/A0/{order}")


def example_rows(stage: str, frozen: str, sidecars: Dict[Tuple[str, str], Dict]) -> List[Dict[str, object]]:
    """
    Choose ranking examples by a fixed rule, URI-only.

    For each dataset, among the stage population's reference sources of
    S/A0, the ``EXAMPLES_PER_DIRECTION`` largest gains and losses in
    reciprocal rank of the frozen order over ``uri`` (ties by source URI).

    Parameters
    ----------
    stage : str
        Stage; its population supplies the sources.
    frozen : str
        Frozen order.
    sidecars : Dict[Tuple[str, str], Dict]
        Sidecars by (dataset, condition).

    Returns
    -------
    List[Dict[str, object]]
        Example rows (URIs, ranks and top candidates with scores).
    """
    rows = []
    for dataset_id in sorted(PAIR_NAMES):
        pair = [{r["source"]: r for r in sidecars[(dataset_id, a0_condition(dataset_id, o))]["reference_sources"][stage]}
                for o in (REFERENCE_ORDER, frozen)]
        delta = {s: _rr(pair[1][s]) - _rr(pair[0][s]) for s in pair[0]}
        gains = sorted((s for s in delta if delta[s] > 0), key=lambda s: (-delta[s], s))
        losses = sorted((s for s in delta if delta[s] < 0), key=lambda s: (delta[s], s))
        for direction, chosen in (("gain", gains), ("loss", losses)):
            rows += [_example(stage, dataset_id, direction, pair[0][s], pair[1][s])
                     for s in chosen[:EXAMPLES_PER_DIRECTION]]
    return rows


def _rr(record: Dict) -> float:
    """Reciprocal rank of a reference-source record."""
    return 1.0 / record["best_rank"] if record["best_rank"] else 0.0


def _top(record: Dict) -> str:
    """Top candidates of a record as ``uri@score`` joined by semicolons."""
    return ";".join(f"{t}@{score:.4f}" for t, score in record["top"])


def _example(stage: str, dataset_id: str, direction: str, uri: Dict, frozen: Dict) -> Dict[str, object]:
    """One URI-only example row."""
    return {"stage": stage, "dataset": dataset_id, "direction": direction, "source": uri["source"],
            "gold_targets": ";".join(uri["gold_targets"]), "rank_uri": uri["best_rank"] or "",
            "rank_frozen": frozen["best_rank"] or "", "top_uri": _top(uri), "top_frozen": _top(frozen)}


def sha256_of(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(artifacts: Path, local_sidecars: List[Path]) -> None:
    """List the SHA-256 of every bundle file, plus the local-only sidecars of this export."""
    manifest = artifacts / "MANIFEST.sha256"
    previous = [line for line in (manifest.read_text().splitlines() if manifest.exists() else [])
                if line.startswith("# local ") and not any(line.endswith(str(p)) for p in local_sidecars)]
    tracked = sorted(p for p in artifacts.rglob("*") if p.is_file() and p != manifest)
    lines = [f"{sha256_of(p)}  {p.relative_to(artifacts)}" for p in tracked]
    local = sorted(previous + [f"# local {sha256_of(p)}  {p}" for p in local_sidecars])
    manifest.write_text("\n".join(lines + local) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# report (bundle only)
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a plain or gzip CSV."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_outcomes(artifacts: Path, stage: str, population: str) -> Dict[Tuple[str, str], Cell]:
    """Map (dataset, condition) to {source: outcome} for one population of one stage."""
    cells: Dict[Tuple[str, str], Cell] = defaultdict(dict)
    for path in sorted((artifacts / "outcomes").glob(f"{stage}_*.csv.gz")):
        for row in read_csv(path):
            if row["population"] == population:
                cells[(row["dataset"], row["condition_id"])][row["source"]] = row
    return cells


def logical_map(artifacts: Path) -> Dict[str, Dict[str, str]]:
    """Read the logical-condition map from the bundle."""
    return {row["name"]: row for row in read_csv(artifacts / "conditions.csv")}


def subsets(cells: Dict[Tuple[str, str], Cell], stage: str) -> Dict[str, Tuple[str, Optional[set]]]:
    """Return the reported source subsets: D1 validation split into probed and untouched, else whole."""
    if stage != "validation":
        return {"D1": ("D1", None), "D2": ("D2", None)}
    any_d1 = next(cell for (d, _c), cell in cells.items() if d == "D1")
    probed = set(probed_sample([s for s, r in any_d1.items() if r["ranked"] == "1"]))
    return {"D1 untouched (751)": ("D1", set(any_d1) - probed), "D1 probed (400)": ("D1", probed),
            "D2": ("D2", None)}


def vectors(cell: Cell, sources: List[str], field: str) -> np.ndarray:
    """Per-source values of one field, aligned to ``sources``."""
    return np.array([float(cell[s][field]) for s in sources])


def changed(a: Cell, b: Cell, sources: List[str]) -> np.ndarray:
    """Whether a source's own text or any gold-target text differs between two cells."""
    return np.array([a[s]["source_text_fp"] != b[s]["source_text_fp"] or a[s]["gold_text_fps"] != b[s]["gold_text_fps"]
                     for s in sources])


def contrast(cells: Dict, lmap: Dict, a: str, b: str, subset: Tuple[str, Optional[set]]) -> Dict[str, object]:
    """Paired Δ MRR and Δ Recall@1 of logical ``a`` − ``b`` on a subset, with strata."""
    dataset, keep = subset
    ca, cb = cells[(dataset, lmap[a]["condition_id"])], cells[(dataset, lmap[b]["condition_id"])]
    if set(ca) != set(cb):
        raise ValueError(f"{a} and {b} cover different sources.")
    sources = sorted(s for s in ca if keep is None or s in keep)
    moved = changed(ca, cb, sources)
    out = {"n": len(sources), "n_changed": int(moved.sum())}
    for field, name in (("rr", "mrr"), ("hit1", "r1")):
        va, vb = vectors(ca, sources, field), vectors(cb, sources, field)
        out[name] = paired_interval(va, vb)
        out[f"{name}_changed"] = paired_interval(va[moved], vb[moved])
    return out


def per_source_values(cells: Dict, lmap: Dict, name: str, subset: Tuple[str, Optional[set]],
                      field: str) -> Tuple[List[str], np.ndarray]:
    """Sources and values of one field for a logical condition on a subset."""
    dataset, keep = subset
    cell = cells[(dataset, lmap[name]["condition_id"])]
    sources = sorted(s for s in cell if keep is None or s in keep)
    return sources, vectors(cell, sources, field)


def compound(cells: Dict, lmap: Dict, terms: List[Tuple[float, str]], subset: Tuple) -> Dict[str, Dict]:
    """Paired interval of a signed sum of logical conditions (interaction, difference-in-differences)."""
    out = {}
    for field, name in (("rr", "mrr"), ("hit1", "r1")):
        series = [per_source_values(cells, lmap, n, subset, field) for _w, n in terms]
        if any(s[0] != series[0][0] for s in series):
            raise ValueError("Compound contrast over different sources.")
        total = sum(w * v for (w, _n), (_s, v) in zip(terms, series))
        out[name] = paired_interval(total, np.zeros_like(total))
    return out


def fmt(interval: Dict[str, float]) -> str:
    """Format an estimate with its 95% interval."""
    if not interval.get("n"):
        return "–"
    return f"{interval['estimate']:+.4f} [{interval['low']:+.4f}, {interval['high']:+.4f}]"


def cell_table(cells: Dict, lmap: Dict, parts: Dict) -> List[str]:
    """MRR and Recall@1 of every logical condition, per reported subset."""
    lines = ["| Logical condition | Execution | " + " | ".join(f"{p} MRR / R@1 (n)" for p in parts) + " |",
             "|---|---|" + "---|" * len(parts)]
    for name, row in lmap.items():
        values = []
        for subset in parts.values():
            if subset[0] != row["dataset"]:
                values.append("")
                continue
            _s, rr = per_source_values(cells, lmap, name, subset, "rr")
            _s, hit = per_source_values(cells, lmap, name, subset, "hit1")
            values.append(f"{rr.mean():.4f} / {hit.mean():.4f} ({rr.size})")
        lines.append(f"| {name} | `{row['condition_id']}` | " + " | ".join(values) + " |")
    return lines


def s_order_table(cells: Dict, lmap: Dict, parts: Dict) -> Tuple[List[str], List[Dict[str, object]]]:
    """Every order − uri within each Block S row, with the changed-source stratum."""
    lines = ["| Subset | Row | Contrast | n (changed) | Δ MRR [95%] | Δ R@1 [95%] | Δ MRR, changed only |",
             "|---|---|---|---|---|---|---|"]
    series = []
    for part, subset in parts.items():
        for rep in S_REPRESENTATIONS[subset[0]]:
            for order in NON_URI:
                a, b = f"{subset[0]}/S/{rep}/{order}", f"{subset[0]}/S/{rep}/uri"
                c = contrast(cells, lmap, a, b, subset)
                lines.append(f"| {part} | {rep} | {order} − uri | {c['n']} ({c['n_changed']}) | {fmt(c['mrr'])} | "
                             f"{fmt(c['r1'])} | {fmt(c['mrr_changed'])} |")
                series.append({"subset": part, "row": rep, "order": order, **{k: round(c["mrr"][k], 6) for k in
                                                                              ("estimate", "low", "high")}})
    return lines, series


def representation_table(cells: Dict, lmap: Dict, parts: Dict) -> List[str]:
    """A-core − A0 and C+x − A-core within each render order."""
    lines = ["| Subset | Contrast | " + " | ".join(f"{o}: Δ MRR [95%]" for o in SERIALIZATION_ORDERS) + " |",
             "|---|---|" + "---|" * len(SERIALIZATION_ORDERS)]
    for part, subset in parts.items():
        reps = S_REPRESENTATIONS[subset[0]]
        pairs = [("A-core", "A0")] + [(r, "A-core") for r in reps if r.startswith("C+")]
        for a, b in pairs:
            row = [fmt(contrast(cells, lmap, f"{subset[0]}/S/{a}/{o}", f"{subset[0]}/S/{b}/{o}", subset)["mrr"])
                   for o in SERIALIZATION_ORDERS]
            lines.append(f"| {part} | {a} − {b} | " + " | ".join(row) + " |")
    return lines


def b_effects(dataset: str, budget: int, order: str) -> List[Tuple[str, List[Tuple[float, str]]]]:
    """Name and signed terms of every Block B effect of one order."""
    p = f"{dataset}/B{budget}/"
    if order == "uri":
        return [("selection (uri rendering) = combined", [(1, p + "C-uri"), (-1, p + "F-uri")])]
    if order == "core":
        return [("serialization (reference set) = combined", [(1, p + "F-core"), (-1, p + "F-uri")])]
    o = order
    return [("serialization, reference set", [(1, p + f"F-{o}"), (-1, p + "F-uri")]),
            ("serialization, own set", [(1, p + f"C-{o}"), (-1, p + f"X-{o}")]),
            ("selection, uri rendering", [(1, p + f"X-{o}"), (-1, p + "F-uri")]),
            ("selection, own rendering", [(1, p + f"C-{o}"), (-1, p + f"F-{o}")]),
            ("combined policy", [(1, p + f"C-{o}"), (-1, p + "F-uri")]),
            ("interaction", [(1, p + f"C-{o}"), (-1, p + f"X-{o}"), (-1, p + f"F-{o}"), (1, p + "F-uri")])]


def b_table(cells: Dict, lmap: Dict, parts: Dict) -> List[str]:
    """Every Block B effect, plus each serialization effect minus its no-budget counterpart."""
    lines = ["| Subset | Budget | Order | Effect | Δ MRR [95%] | Δ R@1 [95%] |", "|---|---|---|---|---|---|"]
    for part, subset in parts.items():
        for budget in (b for d, b in B_BUDGETS if d == subset[0]):
            for order in SERIALIZATION_ORDERS:
                for effect, terms in b_effects(subset[0], budget, order):
                    c = compound(cells, lmap, terms, subset)
                    lines.append(f"| {part} | {budget} | {order} | {effect} | {fmt(c['mrr'])} | {fmt(c['r1'])} |")
                if order != "uri":
                    terms = [(1, f"{subset[0]}/B{budget}/F-{order}"), (-1, f"{subset[0]}/B{budget}/F-uri"),
                             (-1, f"{subset[0]}/S/A0/{order}"), (1, f"{subset[0]}/S/A0/uri")]
                    c = compound(cells, lmap, terms, subset)
                    lines.append(f"| {part} | {budget} | {order} | serialization (reference set) − no-budget "
                                 f"serialization | {fmt(c['mrr'])} | {fmt(c['r1'])} |")
    return lines


def primary_interval(a: np.ndarray, b: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Paired ``a − b`` with its 95% and Bonferroni-adjusted intervals on one replicate matrix.

    The Bonferroni family is the two datasets of one metric, so the adjusted
    interval is the 97.5% percentile interval (quantiles 0.0125 and 0.9875) of
    the same 2,000 replicates (fresh seed 42) that give the 95% interval.

    Parameters
    ----------
    a, b : np.ndarray
        Per-source values aligned by source.

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``"95"`` and ``"bonferroni"``: ``estimate``, ``low``, ``high``, ``n``.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    means = diff[index_matrix(diff.size, REPORT_REPLICATES, REPORT_SEED)].mean(axis=1)
    alpha = 0.05 / BONFERRONI_FAMILY
    out = {}
    for name, tail in (("95", 0.025), ("bonferroni", alpha / 2)):
        out[name] = {"estimate": float(diff.mean()), "low": float(np.quantile(means, tail)),
                     "high": float(np.quantile(means, 1 - tail)), "n": int(diff.size)}
    return out


def primary_table(cells: Dict, lmap: Dict, frozen: str) -> List[str]:
    """The pre-specified primary contrasts: frozen − uri on D1/S/A0 and D2/S/A0, whole test population."""
    lines = ["| Dataset | n | Metric | uri | " + frozen + " | Δ [95%] | Δ [Bonferroni 97.5%] |",
             "|---|---|---|---|---|---|---|"]
    for dataset in sorted(PAIR_NAMES):
        a, b = f"{dataset}/S/A0/{frozen}", f"{dataset}/S/A0/{REFERENCE_ORDER}"
        for field, metric in (("rr", "MRR"), ("hit1", "Recall@1")):
            sa, va = per_source_values(cells, lmap, a, (dataset, None), field)
            sb, vb = per_source_values(cells, lmap, b, (dataset, None), field)
            if sa != sb:
                raise ValueError(f"{a} and {b} cover different sources.")
            c = primary_interval(va, vb)
            lines.append(f"| {dataset} | {va.size} | {metric} | {vb.mean():.4f} | {va.mean():.4f} | "
                         f"{fmt(c['95'])} | {fmt(c['bonferroni'])} |")
    return lines


def selection_section(artifacts: Path) -> List[str]:
    """How the order was chosen on D2 validation, from the committed freeze record."""
    record = json.loads((artifacts / FREEZE_NAME).read_text(encoding="utf-8"))
    d2 = record["d2_validation"]
    lines = ["| Candidate | D2 validation MRR | − uri [95%] | best − candidate [95%] | in E |", "|---|---|---|---|---|"]
    for order, c in d2["candidates"].items():
        lines.append(f"| {order} | {c['mrr']:.4f} | {fmt({**c['minus_uri'], 'n': 1})} | "
                     f"{fmt({**c['best_minus'], 'n': 1})} | {'yes' if c['in_E'] else 'no'} |")
    return [_selection_sentence(record), "", f"uri MRR {d2['uri_mrr']:.4f}; n = {record['rule']['n_sources']} D2 "
            "validation reference sources; 10,000 shared replicates, seed 20261016.", "", *lines]


def _selection_sentence(record: Dict) -> str:
    """State whether the frozen order won outright or by the tie-breaking precedence."""
    d2, chosen = record["d2_validation"], record["chosen"]
    precedence = " > ".join(record["rule"]["precedence"])
    if chosen == d2["best"]:
        return f"`{chosen}` was frozen because it had the highest D2 validation MRR."
    return (f"`{chosen}` was frozen by the pre-specified tie-breaking rule, **not** because it had the highest D2 "
            f"validation MRR: `{d2['best']}` had the highest ({d2['candidates'][d2['best']]['mrr']:.4f} vs "
            f"{d2['candidates'][chosen]['mrr']:.4f}), but no candidate's `MRR(best) − MRR(candidate)` interval "
            f"excluded 0, so E = {{{', '.join(d2['eligible'])}}} and the precedence {precedence} chose `{chosen}`. "
            "These validation intervals are descriptive and not adjusted for selection.")


def examples_section(artifacts: Path, stage: str) -> List[str]:
    """The script-chosen examples of a stage, URI-only."""
    columns = ["dataset", "direction", "source", "gold_targets", "rank_uri", "rank_frozen"]
    return [f"Per dataset, the {EXAMPLES_PER_DIRECTION} largest gains and losses in reciprocal rank of the frozen "
            "order over uri on S/A0 (ties by source URI). Top candidates with scores are in "
            f"`examples/{stage}_examples.csv`; readable text is generated locally by `examples-local`.", "",
            *csv_table(artifacts / "examples" / f"{stage}_examples.csv", columns)]


def check_cells(cells: Dict[Tuple[str, str], Cell], lmap: Dict[str, Dict[str, str]]) -> None:
    """
    Refuse to report unless every execution has outcomes over the same sources per dataset.

    Parameters
    ----------
    cells : Dict[Tuple[str, str], Cell]
        Outcomes by (dataset, condition).
    lmap : Dict[str, Dict[str, str]]
        Logical-condition map.

    Raises
    ------
    ValueError
        If an execution is missing, empty, or covers other sources.
    """
    sources: Dict[str, set] = {}
    for name, row in lmap.items():
        cell = cells.get((row["dataset"], row["condition_id"]))
        if not cell:
            raise ValueError(f"No outcomes for {name} ({row['condition_id']}); the stage is incomplete.")
        if sources.setdefault(row["dataset"], set(cell)) != set(cell):
            raise ValueError(f"{name} covers other sources than the rest of {row['dataset']}.")


def report_inputs(artifacts: Path, stage: str) -> List[Path]:
    """Every bundle file a stage's report reads (plus its provenance)."""
    fixed = [artifacts / "conditions.csv", artifacts / FREEZE_NAME, artifacts / f"PROVENANCE_{stage}.json",
             artifacts / "examples" / f"{stage}_examples.csv"]
    fixed += [artifacts / "diagnostics" / f"{n}.csv" for n in ("identity_gates", "label_retention", "selection_changes")]
    return sorted(fixed + list((artifacts / "outcomes").glob(f"{stage}_*.csv.gz")))


def report_input_digests(artifacts: Path, stage: str) -> Dict[str, str]:
    """SHA-256 of every report input, keyed by bundle-relative path; raise if one is missing."""
    paths = report_inputs(artifacts, stage)
    missing = [str(p) for p in paths if not p.exists()]
    if missing or not any(p.name.endswith(".csv.gz") for p in paths):
        raise FileNotFoundError(f"Report inputs missing: {missing or 'outcomes'}.")
    return {str(p.relative_to(artifacts)): sha256_of(p) for p in paths}


def require_rank_gate(artifacts: Path) -> None:
    """
    Refuse a test report unless ``gate-ranks`` passed on the current test outcomes.

    Parameters
    ----------
    artifacts : Path
        Bundle root.

    Raises
    ------
    RuntimeError
        If the gate record is missing, failed, or names other report-input digests.
    """
    path = artifacts / "diagnostics" / GATE_NAME
    if not path.exists():
        raise RuntimeError("No rank gate record: run gate-ranks before the test report.")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate["status"] != "pass" or gate["inputs_sha256"] != report_input_digests(artifacts, "test"):
        raise RuntimeError("The rank gate failed or was run on different test outcomes or report inputs; no test report.")


def csv_table(path: Path, columns: List[str]) -> List[str]:
    """Render selected columns of a bundle CSV as a markdown table."""
    rows = read_csv(path)
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] + \
        ["| " + " | ".join(r[c] for c in columns) + " |" for r in rows]


def plot_orders(series: List[Dict[str, object]], path: Path) -> None:
    """Forest plot of order − uri Δ MRR per Block S row and subset, with a fixed-metadata PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{s['subset']} · {s['row']} · {s['order']}" for s in series]
    fig, axis = plt.subplots(figsize=(8, 0.25 * len(series) + 1.5))
    y = np.arange(len(series))[::-1]
    est = np.array([s["estimate"] for s in series])
    err = np.array([[s["estimate"] - s["low"], s["high"] - s["estimate"]] for s in series]).T
    axis.errorbar(est, y, xerr=err, fmt="o", capsize=2, markersize=3)
    axis.set(yticks=y, yticklabels=labels, xlabel="Δ MRR vs uri (95% paired bootstrap)")
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def report(stage: str, artifacts: Path) -> Path:
    """Write a stage's report and figures from the bundle alone; return the report path."""
    if stage == "test":
        require_rank_gate(artifacts)
    population = "validation" if stage == "validation" else "test"
    cells, lmap = load_outcomes(artifacts, stage, population), logical_map(artifacts)
    check_cells(cells, lmap)
    parts = subsets(cells, stage)
    order_lines, series = s_order_table(cells, lmap, parts)
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(figures / f"{stage}_order_effects.csv", list(series[0]), series)
    plot_orders(series, figures / f"{stage}_order_effects.png")
    primary = (["## Primary contrasts (pre-specified)", "", *primary_table(cells, lmap, frozen_order(artifacts)), ""]
               if stage == "test" else [])
    sections = [f"# #134 serialization order: {stage} report", "", _preamble(stage), "",
                "## How the order was chosen (D2 validation)", "", *selection_section(artifacts), "", *primary,
                "## Cell metrics", "", *cell_table(cells, lmap, parts), "",
                "## Block S: each order − uri (selected set fixed)", "", *order_lines, "",
                f"![order effects](figures/{stage}_order_effects.png)", "",
                "## Block S: representation contrasts within each order", "", *representation_table(cells, lmap, parts),
                "", "## Block B: serialization, selection, combined and interaction", "", *b_table(cells, lmap, parts),
                "", "## Label retention and position (tokenizer-only inventory)", "",
                *csv_table(artifacts / "diagnostics" / "label_retention.csv",
                           ["logical", "side", "n_with_label", "label_retention", "median_label_offset",
                            "share_label_first"]),
                "", "## Selection changes against the fixed core-first set", "",
                *csv_table(artifacts / "diagnostics" / "selection_changes.csv",
                           ["logical", "side", "n_set_differs", "items_gained", "items_lost", "labels_lost",
                            "labels_gained"]),
                "", "## Ranking examples", "", *examples_section(artifacts, stage),
                "", "## Identity gates", "", *_gate_summary(artifacts), ""]
    path = artifacts / f"report_{stage}.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    write_manifest(artifacts, [])
    return path


def _preamble(stage: str) -> str:
    """Interpretive notes printed at the top of every report."""
    notes = ["Paired source-level bootstraps: 2,000 replicates, fresh seed 42 per contrast, 95% percentile "
             "intervals; datasets are never pooled. Two pairs (D1, D2) and one model (M2): no generalisation claim."]
    if stage == "test":
        notes.append("Primary contrasts: frozen order − uri on D1/S/A0 and D2/S/A0 (MRR, Recall@1), with 95% and "
                     "Bonferroni-adjusted 97.5% intervals (family: the two datasets of one metric). Every other "
                     "table is secondary and unadjusted. Written only after `gate-ranks` passed.")
    if stage == "validation":
        notes.append("Validation stage: descriptive only. D1 validation is exploratory and split into the 400 "
                     "sources the #131 probe used and the 751 it did not; the order is chosen on D2 validation by "
                     "the rule in `serialization_freeze.json`, whose intervals are not adjusted for selection.")
    return "\n".join(f"- {n}" for n in notes)


def _gate_summary(artifacts: Path) -> List[str]:
    """Count passed and failed identity gates by gate type."""
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in read_csv(artifacts / "diagnostics" / "identity_gates.csv"):
        counts[(row["gate"], row["status"])] += 1
    return ["| Gate | Checks passed | Failed |", "|---|---|---|"] + \
        [f"| {g} | {counts[(g, 'pass')]} | {counts[(g, 'FAIL')]} |" for g in sorted({g for g, _s in counts})]


# ---------------------------------------------------------------------------
# verify-rebuild and check-text
# ---------------------------------------------------------------------------


def verify_rebuild(stage: str, artifacts: Path) -> int:
    """Rebuild the report from a copy of the bundle, outside the repository; return mismatches."""
    outputs = [f"report_{stage}.md", f"figures/{stage}_order_effects.png", f"figures/{stage}_order_effects.csv"]
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "bundle"
        shutil.copytree(artifacts, copy, ignore=shutil.ignore_patterns(*outputs, "figures"))
        script = Path(__file__).resolve()
        subprocess.run([sys.executable, str(script), "report", "--stage", stage, "--artifacts_dir", str(copy)],
                       cwd=tmp, check=True)
        bad = [o for o in outputs if (copy / o).read_bytes() != (artifacts / o).read_bytes()]
    print(f"Rebuilt {len(outputs)} outputs from the bundle alone; {len(bad)} differ: {bad}")
    return len(bad)


def entity_texts(min_length: int = 12) -> set:
    """Collect every D1/D2 source and candidate item value of at least ``min_length`` characters (local data)."""
    from kgsemembed.datasets import load_dataset
    from kgsemembed.pipeline.ppas_ablation import build_pair_context
    values = set()
    for dataset_id in PAIR_NAMES:
        context = build_pair_context(load_dataset(dataset_id, Path("data"))[0], None, len, Path("data"))
        for side in (context.source, context.candidate):
            values |= {item.text.split(": ", 1)[-1] for items in side.items for item in items}
    return {v for v in values if len(v) >= min_length and not v.startswith("http")}


def files_with_text(artifacts: Path, values: set) -> List[Path]:
    """
    Return the bundle files (PNG excluded) that contain any of ``values``.

    Parameters
    ----------
    artifacts : Path
        Bundle root.
    values : set
        Entity texts to look for.

    Returns
    -------
    List[Path]
        Offending files, sorted.
    """
    hits = []
    for path in sorted(p for p in artifacts.rglob("*") if p.is_file() and p.suffix != ".png"):
        content = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        if any(v in content for v in values):
            hits.append(path)
    return hits


def check_text(artifacts: Path) -> int:
    """Return the number of bundle files containing D1/D2 entity text."""
    values = entity_texts()
    hits = files_with_text(artifacts, values)
    print(f"Scanned bundle files for {len(values)} entity texts: {len(hits)} files contain one {hits}")
    return len(hits)


def expected_rank_targets() -> Dict[Tuple[str, str], Tuple[str, str]]:
    """(dataset, logical name) → (results_dir, condition_id) of every stored run the gate must compare."""
    from kgsemembed.pipeline.serialization_conditions import reproduction_target
    targets = {(c.dataset, c.name): reproduction_target(c) for c in LOGICAL_CONDITIONS}
    return {key: target for key, target in sorted(targets.items()) if target is not None}


def _fps(field: str) -> Dict[str, Optional[str]]:
    """
    Parse an exported ``uri=fingerprint;...`` field.

    :func:`outcome_rows` writes a gold target that was never encoded as
    ``uri=None`` (``str(None)``); an empty fingerprint is the same marker in
    the ``source_text_fp`` form. Only these two exact values mean "not
    encoded"; every other value is kept as written and compared verbatim.

    Parameters
    ----------
    field : str
        The exported ``gold_text_fps`` (or ``gold_set_fps``) value.

    Returns
    -------
    Dict[str, Optional[str]]
        Fingerprint by target URI; ``None`` for an unencoded target.
    """
    pairs = (entry.rpartition("=") for entry in field.split(";") if entry)
    return {uri: None if fp in UNENCODED_FINGERPRINTS else fp for uri, _sep, fp in pairs}


def compare_source(ours: Optional[Dict[str, str]], stored: Dict, fingerprints: Dict[str, Dict[str, str]]) -> Dict[str, bool]:
    """
    Compare one source's new test outcome with its stored record.

    Parameters
    ----------
    ours : Optional[Dict[str, str]]
        The new outcome row of the same source URI, ``None`` if absent.
    stored : Dict
        The stored ``test_sources`` record.
    fingerprints : Dict[str, Dict[str, str]]
        The stored run's ``selection_fingerprints`` by side.

    Returns
    -------
    Dict[str, bool]
        Which of ``missing``, ``rank``, ``gold`` and ``text`` differ.
    """
    if ours is None:
        return {"missing": True, "rank": True, "gold": True, "text": True}
    gold = _fps(ours["gold_text_fps"])
    rank = int(ours["best_rank"]) if ours["best_rank"] else None
    text = (ours["source_text_fp"] or None) != fingerprints["source"].get(stored["source"]) or any(
        gold[t] != fingerprints["candidate"].get(t) for t in gold)
    return {"missing": False, "rank": rank != stored["best_rank"], "gold": sorted(gold) != sorted(stored["gold_targets"]),
            "text": text}


def rank_gate(cells: Dict[Tuple[str, str], Cell], stored: Dict[Tuple[str, str], Optional[Dict]],
              lmap: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    """
    Compare each canonically rendered execution with its stored run, source by source.

    Parameters
    ----------
    cells : Dict[Tuple[str, str], Cell]
        Test-population outcomes by (dataset, condition).
    stored : Dict[Tuple[str, str], Optional[Dict]]
        Stored sidecar (``test_sources`` and ``selection_fingerprints``) by
        (dataset, logical name); ``None`` when the baseline file is missing.
    lmap : Dict[str, Dict[str, str]]
        Logical-condition map.

    Returns
    -------
    List[Dict[str, object]]
        Per logical condition: sources compared, missing and extra sources,
        rank, gold-set and text mismatches, MRR of both, and the status.
    """
    rows = []
    for (dataset, name), baseline in sorted(stored.items()):
        cell = cells.get((dataset, lmap[name]["condition_id"]), {}) if name in lmap else {}
        records = baseline["test_sources"] if baseline else []
        diffs = [compare_source(cell.get(r["source"]), r, baseline["selection_fingerprints"]) for r in records]
        counts = {k: sum(d[k] for d in diffs) for k in ("missing", "rank", "gold", "text")}
        extra = len(set(cell) - {r["source"] for r in records})
        ours = [1 / int(cell[r["source"]]["best_rank"]) if r["source"] in cell and cell[r["source"]]["best_rank"] else 0
                for r in records]
        stored_rr = [1 / r["best_rank"] if r["best_rank"] else 0 for r in records]
        ok = baseline is not None and records and extra == 0 and not any(counts.values())
        rows.append({"logical": name, "baseline_found": int(baseline is not None), "n_sources": len(records),
                     "n_missing": counts["missing"], "n_extra": extra, "n_rank_mismatch": counts["rank"],
                     "n_gold_mismatch": counts["gold"], "n_text_mismatch": counts["text"],
                     "mrr": round(float(np.mean(ours)), 6) if ours else "", "stored_mrr":
                     round(float(np.mean(stored_rr)), 6) if records else "", "status": "pass" if ok else "FAIL"})
    return rows


def stored_baselines() -> Dict[Tuple[str, str], Optional[Dict]]:
    """Load the stored #131/#128 sidecar behind every expected target (local); ``None`` if absent."""
    stored = {}
    for (dataset, name), (results_dir, condition_id) in expected_rank_targets().items():
        path = Path(results_dir) / condition_id / dataset / f"{PAIR_NAMES[dataset]}_entities.json"
        stored[(dataset, name)] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return stored


def gate_ranks(artifacts: Path) -> int:
    """Write the entity-level reproduction table and the gate record into the bundle; return failures."""
    lmap = logical_map(artifacts)
    inputs = report_input_digests(artifacts, "test")
    rows = rank_gate(load_outcomes(artifacts, "test", "test"), stored_baselines(), lmap)
    write_csv(artifacts / "diagnostics" / "rank_reproduction.csv", list(rows[0]), rows)
    failed = [r for r in rows if r["status"] != "pass"]
    if len(rows) != len(expected_rank_targets()):
        failed.append({"logical": "expected targets"})
    gate = {"status": "FAIL" if failed else "pass", "n_executions": len(rows), "n_failed": len(failed),
            "inputs_sha256": inputs}
    (artifacts / "diagnostics" / GATE_NAME).write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} executions compared source by source with stored test runs; {len(failed)} failed")
    return len(failed)


def readable_examples(stage: str, artifacts: Path, results_dir: Path) -> Path:
    """
    Render the readable text of every example locally (never into the bundle).

    Parameters
    ----------
    stage : str
        Stage whose examples are rendered.
    artifacts : Path
        Bundle root (examples and freeze record).
    results_dir : Path
        Local results root; the output goes to ``<results_dir>/<stage>/examples_readable.md``.

    Returns
    -------
    Path
        The written file.
    """
    from serialization_inventory import logical_selector, token_counter
    from kgsemembed.datasets import load_dataset
    from kgsemembed.pipeline.ppas_ablation import build_pair_context
    from kgsemembed.pipeline.serialization_conditions import get_logical
    frozen, count = frozen_order(artifacts), token_counter()
    rows = read_csv(artifacts / "examples" / f"{stage}_examples.csv")
    lines = [f"# #134 {stage} examples (local only: contains dataset text; do not commit)", ""]
    for dataset_id in sorted({r["dataset"] for r in rows}):
        context = build_pair_context(load_dataset(dataset_id, Path("data"))[0], None, count, Path("data"))
        texts = {o: _texts(context, logical_selector(get_logical(f"{dataset_id}/S/A0/{o}"), count))
                 for o in (REFERENCE_ORDER, frozen)}
        lines += [line for r in rows if r["dataset"] == dataset_id for line in _example_lines(r, texts, frozen)]
    path = results_dir / stage / "examples_readable.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _texts(context: object, selector: object) -> Dict[str, str]:
    """Rendered text of every source and candidate entity under one selector."""
    out = {}
    for side in (context.source, context.candidate):
        for uri, entity_type, items in zip(side.uris, side.entity_types, side.items):
            out[uri] = selector.select(items, uri, entity_type).text
    return out


def _example_lines(row: Dict[str, str], texts: Dict[str, Dict[str, str]], frozen: str) -> List[str]:
    """Markdown block of one example: source, gold targets and top candidates under both orders."""
    lines = [f"## {row['dataset']} {row['direction']}: rank {row['rank_uri'] or '–'} → {row['rank_frozen'] or '–'}",
             "", f"Source `{row['source']}`"]
    gold = row["gold_targets"].split(";")
    for order, column in ((REFERENCE_ORDER, "top_uri"), (frozen, "top_frozen")):
        lines += ["", f"**{order}**", "", f"- source: {texts[order].get(row['source'], '')}"]
        lines += [f"- gold `{g}`: {texts[order].get(g, '(not a candidate)')}" for g in gold]
        for entry in filter(None, row[column].split(";")):
            uri, score = entry.rsplit("@", 1)
            lines.append(f"- top {score}{' (gold)' if uri in gold else ''} `{uri}`: {texts[order].get(uri, '')}")
    return lines + [""]


def main() -> None:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser(description="#134 artifact bundle and reports.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "report", "verify-rebuild", "check-text", "gate-ranks", "examples-local"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts_dir", default=str(ARTIFACTS))
        if name not in ("check-text", "gate-ranks"):
            p.add_argument("--stage", required=True, choices=("validation", "test"))
        if name in ("export", "examples-local"):
            p.add_argument("--results_dir", default=str(RESULTS))
            p.add_argument("--inventory_dir", default=str(INVENTORY))
    args = parser.parse_args()
    artifacts = Path(args.artifacts_dir)
    if args.command == "export":
        export(args.stage, Path(args.results_dir), Path(args.inventory_dir), artifacts)
    elif args.command == "report":
        print(f"Wrote {report(args.stage, artifacts)}")
    elif args.command == "verify-rebuild":
        sys.exit(1 if verify_rebuild(args.stage, artifacts) else 0)
    elif args.command == "gate-ranks":
        sys.exit(1 if gate_ranks(artifacts) else 0)
    elif args.command == "examples-local":
        print(f"Wrote {readable_examples(args.stage, artifacts, Path(args.results_dir))}")
    else:
        sys.exit(1 if check_text(artifacts) else 0)


if __name__ == "__main__":
    main()
