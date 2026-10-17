"""Artifact bundle and reports of the serialization experiment (#134).

Subcommands
-----------
``export``
    Results (``data/results_serialization/<stage>/``) and the tokenizer-only
    inventory → the tracked bundle ``artifacts/issue-134/``: result JSONs,
    per-source outcome tables, the logical-condition map, diagnostics,
    provenance and a SHA-256 manifest (which also lists the local-only
    sidecars).  Every tracked file is URI-only: no entity text.
``report``
    The bundle alone → ``report_<stage>.md`` and ``figures/<stage>_*`` (PNG
    plus the exact series as CSV).  Reads nothing outside the bundle and writes
    no timestamp, so a rebuild is byte-identical.
``verify-rebuild``
    Copies the bundle to a temporary directory, rebuilds the report there from
    a working directory without ``data/``, and compares every output byte for byte.
``check-text``
    Scans every tracked bundle file for dataset entity text (labels and other
    item values of D1/D2, 12 characters or longer); exit status 1 on a hit.
``gate-ranks``
    Test stage, after the freeze: compares every canonically rendered
    execution's test ranks with the stored #131/#128 sidecars source by
    source (not only aggregate MRR); exit status 1 on any mismatch.

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

from kgsemembed.evaluation.order_selection import paired_interval  # noqa: E402
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


def write_gzip_csv(path: Path, columns: Iterable[str], rows: List[Dict[str, object]]) -> None:
    """Write rows as a gzip CSV with a zero mtime, so identical rows give identical bytes."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(buffer.getvalue().encode("utf-8"))


def write_csv(path: Path, columns: Iterable[str], rows: List[Dict[str, object]]) -> None:
    """Write rows as a plain CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def export(stage: str, results_dir: Path, inventory_dir: Path, artifacts: Path) -> None:
    """Write a stage's tracked bundle from its results and the inventory."""
    runs = expected_runs(results_dir / stage)
    results, outcomes = [], defaultdict(list)
    for condition_id, dataset_id, result_path, sidecar_path in runs:
        target = artifacts / stage / "results" / condition_id / dataset_id / result_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_path, target)
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        outcomes[dataset_id] += outcome_rows(stage, dataset_id, condition_id, sidecar)
    for dataset_id, rows in sorted(outcomes.items()):
        rows.sort(key=lambda r: (r["condition_id"], r["population"], r["source"]))
        write_gzip_csv(artifacts / "outcomes" / f"{stage}_{dataset_id}.csv.gz", OUTCOME_COLUMNS, rows)
    write_csv(artifacts / "conditions.csv", CONDITION_COLUMNS,
              [{k: getattr(c, k) if getattr(c, k) is not None else "" for k in CONDITION_COLUMNS}
               for c in LOGICAL_CONDITIONS])
    (artifacts / "diagnostics").mkdir(parents=True, exist_ok=True)
    for name in DIAGNOSTICS:
        shutil.copyfile(inventory_dir / f"{name}.csv", artifacts / "diagnostics" / f"{name}.csv")
    (artifacts / f"PROVENANCE_{stage}.json").write_text(json.dumps(provenance(stage, results), indent=2) + "\n",
                                                        encoding="utf-8")
    write_manifest(artifacts, [s for *_, s in runs])
    print(f"Exported {len(runs)} {stage} runs to {artifacts}")


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
    population = "validation" if stage == "validation" else "test"
    cells, lmap = load_outcomes(artifacts, stage, population), logical_map(artifacts)
    parts = subsets(cells, stage)
    order_lines, series = s_order_table(cells, lmap, parts)
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(figures / f"{stage}_order_effects.csv", list(series[0]), series)
    plot_orders(series, figures / f"{stage}_order_effects.png")
    sections = [f"# #134 serialization order: {stage} report", "", _preamble(stage), "",
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
                "", "## Identity gates", "", *_gate_summary(artifacts), ""]
    path = artifacts / f"report_{stage}.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def _preamble(stage: str) -> str:
    """Interpretive notes printed at the top of every report."""
    notes = ["Paired source-level bootstraps: 2,000 replicates, fresh seed 42 per contrast, 95% percentile "
             "intervals; datasets are never pooled. Two pairs (D1, D2) and one model (M2): no generalisation claim."]
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


def rank_gate(cells: Dict[Tuple[str, str], Cell], stored: Dict[Tuple[str, str], Dict[str, Optional[int]]],
              lmap: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    """
    Compare each canonically rendered execution's test ranks with the stored run, source by source.

    Parameters
    ----------
    cells : Dict[Tuple[str, str], Cell]
        Test-population outcomes by (dataset, condition).
    stored : Dict[Tuple[str, str], Dict[str, Optional[int]]]
        Stored best ranks by (dataset, logical name), from the #131/#128 sidecars.
    lmap : Dict[str, Dict[str, str]]
        Logical-condition map.

    Returns
    -------
    List[Dict[str, object]]
        Per logical condition: sources compared, rank mismatches, MRR of both.
    """
    rows = []
    for (dataset, name), ranks in sorted(stored.items()):
        cell = cells[(dataset, lmap[name]["condition_id"])]
        ours = {s: int(r["best_rank"]) if r["best_rank"] else None for s, r in cell.items()}
        mismatched = sum(ours.get(s, "missing") != rank for s, rank in ranks.items()) + len(set(ours) - set(ranks))
        mrr = [float(np.mean([1 / k if k else 0 for k in values.values()])) for values in (ours, ranks)]
        rows.append({"logical": name, "n_sources": len(ranks), "n_rank_mismatch": mismatched,
                     "mrr": round(mrr[0], 6), "stored_mrr": round(mrr[1], 6),
                     "status": "pass" if mismatched == 0 else "FAIL"})
    return rows


def stored_test_ranks(lmap: Dict[str, Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, Optional[int]]]:
    """Load the stored #131/#128 test ranks behind every canonically rendered execution (local sidecars)."""
    from kgsemembed.pipeline.serialization_conditions import get_logical, reproduction_target
    stored = {}
    for name in lmap:
        logical = get_logical(name)
        target = reproduction_target(logical)
        if target is None:
            continue
        path = Path(target[0]) / target[1] / logical.dataset / f"{PAIR_NAMES[logical.dataset]}_entities.json"
        records = json.loads(path.read_text(encoding="utf-8"))["test_sources"]
        stored[(logical.dataset, name)] = {r["source"]: r["best_rank"] for r in records}
    return stored


def gate_ranks(artifacts: Path) -> int:
    """Write the entity-level test-rank reproduction table into the bundle; return failures."""
    lmap = logical_map(artifacts)
    rows = rank_gate(load_outcomes(artifacts, "test", "test"), stored_test_ranks(lmap), lmap)
    write_csv(artifacts / "diagnostics" / "rank_reproduction.csv", list(rows[0]), rows)
    failed = [r for r in rows if r["status"] != "pass"]
    print(f"{len(rows)} executions compared source by source with stored test ranks; {len(failed)} failed")
    return len(failed)


def main() -> None:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser(description="#134 artifact bundle and reports.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "report", "verify-rebuild", "check-text", "gate-ranks"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts_dir", default=str(ARTIFACTS))
        if name not in ("check-text", "gate-ranks"):
            p.add_argument("--stage", required=True, choices=("validation", "test"))
        if name == "export":
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
    else:
        sys.exit(1 if check_text(artifacts) else 0)


if __name__ == "__main__":
    main()
