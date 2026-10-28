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
        "metrics": {"mrr": mrr, "hits_at_1": mrr, "recall_at_1": mrr, "recall_at_10": mrr, "f1": mrr},
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
    assert "| C14 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | -0.0500 |" in report
    assert "0.9000" not in report


def test_unregistered_and_old_protocol_results_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path, _payload("X9", 0.1))
    _write(tmp_path, _payload("C1", 0.1, evaluation_protocol=1))
    assert comparison.load_condition_payloads(str(tmp_path)) == []


def test_the_sanity_check_is_not_counted_against_b0(tmp_path: Path) -> None:
    """C19 reproduces C10, so counting both would count one result twice."""
    _write(tmp_path, _payload("C19", 0.1))
    _write(tmp_path, _payload("C10", 0.1))
    assert [p["condition_id"] for p in comparison.load_condition_payloads(str(tmp_path))] == ["C10"]


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


def test_column_mean_is_missing_when_any_pair_lacks_the_metric() -> None:
    complete = _payload("C1", 0.4, pair_name="a")
    partial = _payload("C1", 0.6, pair_name="b")
    del partial["metrics"]["hits_at_1"]
    means = comparison.column_means([complete, partial])
    assert comparison._cell(means, "C1", "D5", "mrr") == 0.5
    assert comparison._cell(means, "C1", "D5", "hits_at_1") is None


def test_column_mean_derives_hits_at_1_from_reference_counts() -> None:
    older = _payload("C1", 0.5)
    del older["metrics"]["hits_at_1"]
    counts = {("D5", "p"): {"n_test_refs": 2, "n_test_reference_sources": 1}}
    means = comparison.column_means([older], counts)
    assert comparison._cell(means, "C1", "D5", "hits_at_1") == 1.0


def test_column_mean_is_missing_when_counts_disagree_with_the_result() -> None:
    older = _payload("C1", 0.5)
    del older["metrics"]["hits_at_1"]
    counts = {("D5", "p"): {"n_test_refs": 3, "n_test_reference_sources": 3}}
    means = comparison.column_means([older], counts)
    assert comparison._cell(means, "C1", "D5", "hits_at_1") is None
    assert comparison._cell(means, "C1", "D5", "mrr") == 0.5


def test_best_names_every_condition_tied_at_four_decimals() -> None:
    payloads = [_payload("C10", 0.70001), _payload("C2", 0.6), _payload("C3", 0.7), _payload("C8", 0.69999)]
    means = comparison.column_means(payloads)
    conditions = comparison._conditions(means)
    assert comparison._best(means, conditions, "D5", "mrr") == "C3, C10, C8 (0.7000)"


def test_best_names_a_single_condition_without_a_tie() -> None:
    means = comparison.column_means([_payload("C1", 0.5), _payload("C2", 0.6)])
    assert comparison._best(means, comparison._conditions(means), "D5", "mrr") == "C2 (0.6000)"
    assert comparison._best(means, comparison._conditions(means), "D1", "mrr") == "—"
