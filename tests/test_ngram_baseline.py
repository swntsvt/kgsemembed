"""Tests for the stage-1 n-gram ranking baseline (B0, #137)."""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.candidates.ngram import dump_candidates
from kgsemembed.datasets import AlignmentPair
from kgsemembed.pipeline import ngram_baseline as baseline
from kgsemembed.utils.errors import DataError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.generate_candidates as candidate_script  # noqa: E402
from scripts.generate_results_table import build_table  # noqa: E402

_NS = "http://example.org/"
_SOURCE_LABELS = {"s1": "heart", "s2": "left lung", "s3": "kidney", "s4": "spleen"}
_TARGET_LABELS = {"t1": "heart", "t2": "lung left", "t3": "kidney", "t4": "liver", "t5": "kidneys"}


def _graph(labels: Dict[str, str]) -> Graph:
    graph = Graph()
    for name, label in labels.items():
        graph.add((URIRef(_NS + name), RDF.type, OWL.Class))
        graph.add((URIRef(_NS + name), RDFS.label, Literal(label, lang="en")))
    return graph


def _pair() -> AlignmentPair:
    return AlignmentPair(
        dataset_id="D2",
        pair_name="toy",
        source_graph=_graph(_SOURCE_LABELS),
        target_graph=_graph(_TARGET_LABELS),
        source_entities=sorted(_NS + name for name in _SOURCE_LABELS),
        val_refs=[(_NS + "s1", _NS + "t1")],
        test_refs=[(_NS + "s2", _NS + "t2"), (_NS + "s3", _NS + "t5")],
    )


def _stored(pair: AlignmentPair) -> Dict[str, List[str]]:
    return candidate_script._candidate_map(pair, baseline.NGRAM_CONFIG)


def _write_stored(data_dir: Path, candidates: Dict[str, List[str]]) -> None:
    dump_candidates(candidates, data_dir / "candidates" / "D2" / "toy_candidates.json")


def test_recomputed_order_reproduces_the_candidate_script() -> None:
    """B0 and the candidate script share one scoring path."""
    pair = _pair()
    assert baseline.count_order_mismatches(baseline.scored_candidates(pair), _stored(pair)) == 0


def test_perturbed_candidate_order_raises() -> None:
    """A stored list in a different order fails the identity gate."""
    pair = _pair()
    stored = _stored(pair)
    source = _NS + "s2"
    stored[source] = list(reversed(stored[source]))
    with pytest.raises(baseline.CandidateOrderMismatchError, match="1 of 4"):
        baseline.check_stored_order(baseline.scored_candidates(pair), stored, "D2/toy")


def test_missing_source_counts_as_mismatch() -> None:
    pair = _pair()
    stored = _stored(pair)
    del stored[_NS + "s4"]
    assert baseline.count_order_mismatches(baseline.scored_candidates(pair), stored) == 1


def test_ranked_lists_keep_stored_order_with_descending_scores() -> None:
    pair = _pair()
    ranked = baseline.build_ranked_lists(pair, baseline.scored_candidates(pair))
    stored = _stored(pair)
    for ranked_list in ranked:
        source = ranked_list[0][0]
        assert [target for _, target, _ in ranked_list] == stored[source]
        scores = [score for _, _, score in ranked_list]
        assert scores == sorted(scores, reverse=True)
    assert [lst[0][0] for lst in ranked] == pair.source_entities


def test_gold_rank_records() -> None:
    ranked = [
        [("a", "x", 0.9), ("a", "g", 0.5)],
        [("b", "y", 0.9)],
    ]
    records = baseline.gold_rank_records(ranked, [("a", "g"), ("b", "g"), ("c", "g")])
    assert [r["best_gold_rank"] for r in records] == [2, None, None]
    assert [r["gold_ranks"] for r in records] == [[2], [None], [None]]
    assert [r["ranked"] for r in records] == [True, True, False]
    assert [r["source"] for r in records] == ["a", "b", "c"]


def test_sidecar_ranks_skip_known_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A known pair ranked above the gold target does not count against it."""
    pair = _pair()
    _write_stored(tmp_path, _stored(pair))
    assert _stored(pair)[_NS + "s3"][:2] == [_NS + "t3", _NS + "t5"]
    pair = dataclasses.replace(pair, train_refs=[(_NS + "s3", _NS + "t3")])
    _payload, records = baseline.evaluate_pair(pair, tmp_path)
    assert {r["source"]: r["best_gold_rank"] for r in records}[_NS + "s3"] == 1


def test_refuses_historical_results_dir() -> None:
    with pytest.raises(ValueError, match="must not be written"):
        baseline.check_results_dir("data/results/")


def test_run_writes_result_and_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = _pair()
    _write_stored(tmp_path, _stored(pair))
    monkeypatch.setattr(baseline, "load_dataset", lambda _d, _dir: [pair])
    baseline.run_baseline(["D2"], tmp_path, tmp_path / "out")
    directory = tmp_path / "out" / "B0" / "D2"
    payload = json.loads((directory / "toy_results.json").read_text())
    assert payload["condition_id"] == "B0" and payload["strategy"] == "ngram"
    assert payload["model_key"] is None and payload["n_order_mismatches"] == 0
    assert payload["ngram_config"] == {"n": 3, "metric": "cosine", "top_k": 20, "strip_punctuation": True}
    assert payload["evaluation_protocol"] == 2
    assert set(payload["metrics"]) >= {"f1", "mrr", "recall_at_1", "threshold"}
    assert payload["population"]["n_test_refs"] == 2
    assert 0.0 <= payload["candidate_recall_at_20"] <= 1.0
    sidecar = json.loads((directory / "toy_entities.json").read_text())
    assert [r["source"] for r in sidecar["test_sources"]] == [_NS + "s2", _NS + "s3"]


def test_run_refuses_a_mismatching_candidate_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _pair()
    stored = _stored(pair)
    stored[_NS + "s1"] = stored[_NS + "s1"][1:] + stored[_NS + "s1"][:1]
    _write_stored(tmp_path, stored)
    monkeypatch.setattr(baseline, "load_dataset", lambda _d, _dir: [pair])
    with pytest.raises(baseline.CandidateOrderMismatchError):
        baseline.run_baseline(["D2"], tmp_path, tmp_path / "out")
    assert not (tmp_path / "out" / "B0").exists()


def _write_b0(root: Path, dataset_id: str, metrics: Dict[str, float], pair_name: str = "p") -> None:
    directory = root / "B0" / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"condition_id": "B0", "dataset_id": dataset_id, "metrics": metrics,
               "counts": {"tp": 1, "fp": 1, "fn": 1}}
    (directory / f"{pair_name}_results.json").write_text(json.dumps(payload))


def _b0_line(lines: List[str]) -> str:
    return next(line for line in lines if line.startswith("B0 &"))


def test_table_puts_b0_first_above_a_rule(tmp_path: Path) -> None:
    _write_b0(tmp_path / "b", "D5", {"f1": 0.2, "mrr": 0.3, "recall_at_1": 0.1})
    lines = build_table(str(tmp_path / "r"), "f1", str(tmp_path / "b"))
    index = lines.index(_b0_line(lines))
    assert lines[index - 1] == r"\hline" and lines[index + 1] == r"\hline"
    assert _b0_line(lines).startswith("B0 & n-gram & --- & --- &")
    assert r"\textbf{0.2000}" in _b0_line(lines)
    assert "B0: stage-1" in "\n".join(lines)


def test_table_metric_selects_the_column(tmp_path: Path) -> None:
    _write_b0(tmp_path / "b", "D2", {"f1": 0.2, "mrr": 0.3, "recall_at_1": 0.1})
    lines = build_table(str(tmp_path / "r"), "mrr", str(tmp_path / "b"))
    assert "0.3000" in _b0_line(lines)
    assert r"\label{tab:main_results}" in lines


def test_table_b0_d3_cell_is_pair_mean(tmp_path: Path) -> None:
    _write_b0(tmp_path / "b", "D3", {"f1": 0.2}, "p1")
    _write_b0(tmp_path / "b", "D3", {"f1": 0.4}, "p2")
    assert "0.3000" in _b0_line(build_table(str(tmp_path / "r"), "f1", str(tmp_path / "b")))


def test_table_refuses_b0_scored_with_other_known_pairs(tmp_path: Path) -> None:
    _write_b0(tmp_path / "b", "D1", {"f1": 0.2}, "d1")
    stale = tmp_path / "b" / "B0" / "D1" / "d1_results.json"
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "pair_name": "d1",
                                 "population": {"n_train_refs": 0}}))
    fresh = tmp_path / "r" / "C1" / "D1" / "d1_results.json"
    fresh.parent.mkdir(parents=True)
    fresh.write_text(json.dumps({"dataset_id": "D1", "pair_name": "d1", "population": {"n_train_refs": 726}}))
    with pytest.raises(DataError, match="mix known-pair counts"):
        build_table(str(tmp_path / "r"), "f1", str(tmp_path / "b"))


def test_table_without_baseline_has_no_b0_row(tmp_path: Path) -> None:
    lines = build_table(str(tmp_path / "r"))
    assert not any(line.startswith("B0 &") for line in lines)
    assert "B0: stage-1" not in "\n".join(lines)
