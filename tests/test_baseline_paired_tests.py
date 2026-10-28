"""Tests for the condition-against-B0 paired-test artifact script (#165)."""

import json
import sys
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.baseline_paired_tests as script  # noqa: E402
from kgsemembed.evaluation.aggregator import baseline_section_body  # noqa: E402
from kgsemembed.evaluation.metrics import compute_all_metrics  # noqa: E402
from kgsemembed.evaluation.per_source import link_to_result, per_source_record  # noqa: E402


def _sidecar(n: int, hit_every: int) -> dict:
    """``n`` sources; every ``hit_every``-th ranks its gold first, the rest second."""
    lists, refs = [], []
    for i in range(n):
        first, second = (f"g{i}", f"x{i}") if i % hit_every == 0 else (f"x{i}", f"g{i}")
        lists.append([(f"s{i}", first, 0.9), (f"s{i}", second, 0.4)])
        refs.append((f"s{i}", f"g{i}"))
    return per_source_record(lists, refs, compute_all_metrics(lists, refs, threshold=0.5))


def _write(root: Path, condition: str, sidecar: dict) -> None:
    predictions = sidecar["predictions"]
    payload = {
        "run_timestamp": "2026-10-20T00:00:00Z", "candidates_sha256": "c" * 64, "git_commit": "abc",
        "metrics": {"threshold": sidecar["threshold"]},
        "counts": {"tp": sum(r["correct"] for r in predictions),
                   "fp": sum(r["predicted"] and not r["correct"] for r in predictions)},
        "population": {"n_test_refs": sidecar["n_test_refs"]},
    }
    directory = root / condition / "D1"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "p_results.json").write_text(json.dumps(payload), encoding="utf-8")
    (directory / "p_entities.json").write_text(json.dumps(link_to_result(sidecar, payload)), encoding="utf-8")


def _seed(tmp_path: Path) -> list:
    results, baseline = tmp_path / "results", tmp_path / "baseline"
    _write(baseline, "B0", _sidecar(30, 3))
    _write(results, "C1", _sidecar(30, 1))
    return ["--results_dir", str(results), "--baseline_dir", str(baseline)]


def _run(tmp_path: Path, name: str) -> Tuple[str, list]:
    script.main([*_seed(tmp_path), "--output_dir", str(tmp_path / name)])
    out = tmp_path / name
    return ((out / "baseline_paired_tests.md").read_text(encoding="utf-8"),
            json.loads((out / "baseline_paired_tests.json").read_text(encoding="utf-8")))


def test_artifacts_regenerate_byte_identically(tmp_path: Path) -> None:
    outputs = []
    for run in ("first", "second"):
        script.main([*_seed(tmp_path), "--output_dir", str(tmp_path / run)])
        outputs.append([(tmp_path / run / f"baseline_paired_tests.{ext}").read_bytes() for ext in ("json", "md")])
    assert outputs[0] == outputs[1]


def test_markdown_lists_tested_cells_counts_and_untested_reasons(tmp_path: Path) -> None:
    text, rows = _run(tmp_path, "out")
    c1 = next(r for r in rows if r["condition_b"] == "C1" and r["dataset_id"] == "D1" and r["metric"] == "mrr")
    assert c1["tested"] and c1["delta"] > 0 and c1["family_size"] == 15
    assert "| D1 | 15 | 1 | 1 | 0 | 1 | 0 |" in text
    assert text.count("| C1 | V1 | M1 | D1 |") == 3
    assert "- C2 on D1: No results for C2/D1." in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_artifact_body_is_the_report_section_body(tmp_path: Path) -> None:
    """One renderer: the artifact and report.md never disagree."""
    text, rows = _run(tmp_path, "out")
    assert text.endswith(baseline_section_body(rows) + "\n")
