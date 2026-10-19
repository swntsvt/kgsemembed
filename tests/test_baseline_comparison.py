"""Tests for the B0 comparison report and the dataset provenance script (#137)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.dataset_provenance as provenance  # noqa: E402
import scripts.generate_baseline_comparison as comparison  # noqa: E402

_POPULATION = {"n_test_refs": 2, "reference_complete": True}


def _payload(condition_id: str, mrr: float, **overrides) -> dict:
    payload = {
        "condition_id": condition_id, "dataset_id": "D5", "pair_name": "p",
        "candidates_sha256": "abc", "population": dict(_POPULATION), "evaluation_protocol": 2,
        "metrics": {"mrr": mrr, "recall_at_1": mrr, "recall_at_10": mrr, "f1": mrr},
        "candidate_recall_at_20": 0.5,
    }
    return {**payload, **overrides}


def _write(root: Path, payload: dict) -> None:
    directory = root / payload["condition_id"] / payload["dataset_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{payload['pair_name']}_results.json").write_text(json.dumps(payload))


def _write_b0(root: Path, payload: dict, records: list) -> None:
    _write(root, payload)
    sidecar = root / "B0" / payload["dataset_id"] / f"{payload['pair_name']}_entities.json"
    sidecar.write_text(json.dumps({"test_sources": records}))


def test_mismatch_reasons() -> None:
    base = _payload("B0", 0.2)
    assert comparison.mismatch_reasons(_payload("C1", 0.1), base) == []
    assert comparison.mismatch_reasons(_payload("C1", 0.1, candidates_sha256="x"), base) == ["candidates_sha256"]
    assert comparison.mismatch_reasons(_payload("C1", 0.1), None) == ["no B0 result"]


def test_partial_reference_ignores_judgeable_counts() -> None:
    population = {"n_test_refs": 2, "n_test_ranked_sources": 5, "reference_complete": False}
    base = _payload("B0", 0.2, population=population)
    other = _payload("C1", 0.1, population={**population, "n_test_ranked_sources": 3})
    assert comparison.mismatch_reasons(other, base) == []
    complete = {**population, "reference_complete": True}
    base = _payload("B0", 0.2, population=complete)
    other = _payload("C1", 0.1, population={**complete, "n_test_ranked_sources": 3})
    assert comparison.mismatch_reasons(other, base) == ["population"]


def test_rank_distribution_bins() -> None:
    records = [
        {"ranked": True, "best_gold_rank": 1}, {"ranked": True, "best_gold_rank": 4},
        {"ranked": True, "best_gold_rank": 20}, {"ranked": True, "best_gold_rank": None},
        {"ranked": False, "best_gold_rank": None},
    ]
    counts = comparison.rank_distribution(records)
    assert counts == {"1": 1, "2–5": 1, "6–10": 0, "11–20": 1, "> 20": 1, "unranked": 1, "n": 5}


def test_report_excludes_mismatched_results_and_shows_deltas(tmp_path: Path) -> None:
    _write_b0(tmp_path / "b", _payload("B0", 0.2), [{"ranked": True, "best_gold_rank": 1}])
    _write(tmp_path / "r", _payload("C14", 0.15))
    _write(tmp_path / "r", _payload("C1", 0.9, population={"n_test_refs": 3}))
    report = "\n".join(comparison.build_report(str(tmp_path / "r"), str(tmp_path / "b")))
    assert "1 C-results share" in report and "| C1 | D5 | p | population |" in report
    assert "| C14 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | -0.0500 |" in report
    assert "0.9000" not in report


def test_unregistered_and_old_protocol_results_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path, _payload("X9", 0.1))
    _write(tmp_path, _payload("C1", 0.1, evaluation_protocol=1))
    assert comparison.load_condition_payloads(str(tmp_path)) == []


def test_correspondence_count(tmp_path: Path) -> None:
    rdf = tmp_path / "reference.rdf"
    rdf.write_text("<map><Cell/></map>\n<map><Cell/></map>")
    links = tmp_path / "test_links"
    links.write_text("a\tb\nc\td\n\n")
    assert provenance.correspondence_count(rdf) == 2
    assert provenance.correspondence_count(links) == 2


def test_release_status(tmp_path: Path) -> None:
    local, same, other = tmp_path / "l", tmp_path / "s", tmp_path / "o"
    local.write_text("x")
    same.write_text("x")
    other.write_text("y")
    assert provenance.release_status(local, same) == "identical"
    assert provenance.release_status(local, other) == "DIFFERENT"
    assert provenance.release_status(local, tmp_path / "missing") == "release file not in cache"
    assert provenance.release_status(local, None) == "not in the MELT repository"
