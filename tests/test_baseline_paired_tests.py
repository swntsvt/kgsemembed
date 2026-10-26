"""Tests for the condition-against-B0 paired-test artifact script (#165)."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.baseline_paired_tests as script  # noqa: E402
from tests.test_paired_tests import _lists, _sidecar, _write  # noqa: E402


def _seed(tmp_path: Path) -> list:
    results, baseline = tmp_path / "results", tmp_path / "baseline"
    _write(baseline, "B0", "D1", "p", _sidecar(*_lists(30, 3)))
    _write(results, "C1", "D1", "p", _sidecar(*_lists(30, 1)))
    return ["--results_dir", str(results), "--baseline_dir", str(baseline)]


def test_artifacts_regenerate_byte_identically(tmp_path: Path) -> None:
    args = _seed(tmp_path)
    outputs = []
    for run in ("first", "second"):
        script.main([*args, "--output_dir", str(tmp_path / run)])
        outputs.append([(tmp_path / run / f"baseline_paired_tests.{ext}").read_bytes() for ext in ("json", "md")])
    assert outputs[0] == outputs[1]


def test_markdown_lists_tested_cells_counts_and_untested_reasons(tmp_path: Path) -> None:
    script.main([*_seed(tmp_path), "--output_dir", str(tmp_path / "out")])
    text = (tmp_path / "out" / "baseline_paired_tests.md").read_text(encoding="utf-8")
    rows = json.loads((tmp_path / "out" / "baseline_paired_tests.json").read_text(encoding="utf-8"))
    c1 = next(r for r in rows if r["condition_b"] == "C1" and r["dataset_id"] == "D1" and r["metric"] == "mrr")
    assert c1["tested"] and c1["delta"] > 0 and c1["family_size"] == 14
    assert "| D1 | 14 | 1 | 1 | 0 | 1 | 0 |" in text
    assert text.count("| C1 | V1 | M1 | D1 |") == 3
    assert "- C2 on D1: No results for C2/D1." in text
    assert "## Not tested" in text and text.endswith("\n") and not text.endswith("\n\n")
