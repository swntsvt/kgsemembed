"""Mocked orchestration tests for the Phase 2 experiment runner.

These tests exercise the full ``run_condition`` / ``run_all_conditions``
pipeline without loading real embedding models or datasets: datasets,
candidates, verbalisers, and the encoder are replaced with deterministic
synthetic doubles.
"""

import json

import numpy as np
import pytest
from rdflib import Graph

import kgsemembed.pipeline.run_experiment as runner
from kgsemembed.datasets import AlignmentPair
from kgsemembed.datasets.loader import _split_80_10_10, _split_first_fraction
from kgsemembed.embeddings.models import MODEL_REGISTRY
from kgsemembed.utils.errors import DataError

_VECTORS = {
    "s1": [1.0, 0.0, 0.0],
    "s2": [0.0, 1.0, 0.0],
    "t1": [1.0, 0.0, 0.0],
    "t2": [0.0, 1.0, 0.0],
}
_CANDIDATES = {"s1": ["t1", "t2"], "s2": ["t1", "t2"]}


class _FakeVerbaliser:
    def verbalise(self, _graph, entity_uri, _entity_type):
        return str(entity_uri)


class _FakeEncoder:
    def __init__(self, model_key, model):
        self.model_key = model_key
        self.model = model

    def encode_batch(self, texts, role="candidate", show_progress=True):
        return np.array([_VECTORS[text] for text in texts], dtype=np.float32)


def _make_pair(dataset_id="D2", pair_name="d2_pair"):
    return AlignmentPair(
        dataset_id=dataset_id,
        pair_name=pair_name,
        source_graph=Graph(),
        target_graph=Graph(),
        source_entities=["s1", "s2"],
        val_refs=[("s1", "t1")],
        test_refs=[("s2", "t2")],
    )


@pytest.fixture
def fakes(monkeypatch):
    """Install deterministic doubles and return call recorders."""
    calls = {"models": [], "encoders": 0, "candidate_loads": 0, "datasets": []}

    def fake_load_model(model_key):
        calls["models"].append(model_key)
        return object()

    def fake_encoder(model_key, model):
        calls["encoders"] += 1
        return _FakeEncoder(model_key, model)

    def fake_load_dataset(dataset_id, _data_dir):
        calls["datasets"].append(dataset_id)
        return [_make_pair(dataset_id=dataset_id, pair_name=f"{dataset_id.lower()}_pair")]

    def fake_load_candidates(_dataset_id, _pair_name, _data_dir):
        calls["candidate_loads"] += 1
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_sentence_transformer", fake_load_model)
    monkeypatch.setattr(runner, "EmbeddingEncoder", fake_encoder)
    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "load_candidates", fake_load_candidates)
    monkeypatch.setattr(runner, "build_verbaliser", lambda _s, _m: _FakeVerbaliser())
    return calls


# ---------------------------------------------------------------------------
# run_condition
# ---------------------------------------------------------------------------


def test_run_condition_returns_metrics_and_writes_json(fakes, tmp_path):
    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )

    assert set(results) == {"D2"}
    metrics = results["D2"]
    assert metrics["recall_at_1"] == 1.0
    assert metrics["threshold"] == 0.1

    result_file = tmp_path / "results" / "C1" / "D2" / "d2_pair_results.json"
    assert result_file.exists()


def test_result_json_matches_required_schema(fakes, tmp_path):
    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    payload = json.loads(
        (tmp_path / "results" / "C1" / "D2" / "d2_pair_results.json").read_text()
    )

    assert set(payload) == {
        "condition_id",
        "dataset_id",
        "pair_name",
        "strategy",
        "model_key",
        "model_id",
        "metrics",
        "n_source_entities",
        "n_candidates_per_entity",
    }
    assert set(payload["metrics"]) == {
        "f1",
        "precision",
        "recall",
        "threshold",
        "mrr",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
    }
    assert payload["condition_id"] == "C1"
    assert payload["model_id"] == MODEL_REGISTRY["M1"].model_id
    assert payload["n_candidates_per_entity"] == 20
    assert payload["n_source_entities"] == 2


def test_dataset_filter_restricts_execution(fakes, tmp_path):
    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert fakes["datasets"] == ["D2"]


def test_dataset_filter_excluding_all_runs_nothing(fakes, tmp_path):
    results = runner.run_condition(
        "C6", dataset_ids=["D5"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert results == {}
    assert fakes["datasets"] == []


def test_existing_result_is_skipped_and_read_back(fakes, tmp_path):
    results_dir = tmp_path / "results"
    runner.run_condition("C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=results_dir)
    loads_after_first = fakes["candidate_loads"]

    result_file = results_dir / "C1" / "D2" / "d2_pair_results.json"
    payload = json.loads(result_file.read_text())
    payload["metrics"]["f1"] = 0.123
    result_file.write_text(json.dumps(payload))

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=results_dir
    )
    assert results["D2"]["f1"] == 0.123
    assert fakes["candidate_loads"] == loads_after_first


def test_force_recompute_overwrites_existing_result(fakes, tmp_path):
    results_dir = tmp_path / "results"
    runner.run_condition("C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=results_dir)

    result_file = results_dir / "C1" / "D2" / "d2_pair_results.json"
    payload = json.loads(result_file.read_text())
    payload["metrics"]["f1"] = 0.123
    result_file.write_text(json.dumps(payload))

    results = runner.run_condition(
        "C1",
        dataset_ids=["D2"],
        data_dir=tmp_path,
        results_dir=results_dir,
        force_recompute=True,
    )
    assert results["D2"]["f1"] != 0.123
    assert json.loads(result_file.read_text())["metrics"]["f1"] != 0.123


def test_unknown_condition_raises(fakes, tmp_path):
    with pytest.raises(KeyError):
        runner.run_condition("C999", data_dir=tmp_path, results_dir=tmp_path / "results")


def test_validation_used_for_tuning_and_test_for_evaluation(fakes, monkeypatch, tmp_path):
    recorded = {}

    def fake_tune(_scored_pairs, references):
        recorded["tune_refs"] = references
        return {"best_threshold": 0.42}

    def fake_metrics(_ranked_lists, references, threshold=None):
        recorded["eval_refs"] = references
        recorded["eval_threshold"] = threshold
        return {key: 0.0 for key in runner._METRIC_KEYS}

    monkeypatch.setattr(runner, "tune_threshold", fake_tune)
    monkeypatch.setattr(runner, "compute_all_metrics", fake_metrics)

    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert recorded["tune_refs"] == [("s1", "t1")]
    assert recorded["eval_refs"] == [("s2", "t2")]
    assert recorded["eval_threshold"] == 0.42


def test_sources_and_candidates_encoded_with_correct_roles(fakes, monkeypatch, tmp_path):
    roles = []
    original_encode = _FakeEncoder.encode_batch

    def recording_encode(self, texts, role="candidate", show_progress=True):
        roles.append(role)
        return original_encode(self, texts, role=role, show_progress=show_progress)

    monkeypatch.setattr(_FakeEncoder, "encode_batch", recording_encode)

    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert roles == ["source", "candidate"]


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


def test_failing_pair_does_not_stop_other_pairs(fakes, monkeypatch, tmp_path):
    good = _make_pair(dataset_id="D2", pair_name="good_pair")
    bad = _make_pair(dataset_id="D2", pair_name="bad_pair")
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [bad, good])

    def selective_candidates(_dataset_id, pair_name, _data_dir):
        if pair_name == "bad_pair":
            raise DataError("missing candidates")
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_candidates", selective_candidates)

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert "D2" in results
    assert (tmp_path / "results" / "C1" / "D2" / "good_pair_results.json").exists()
    assert not (tmp_path / "results" / "C1" / "D2" / "bad_pair_results.json").exists()


def test_corrupt_existing_result_is_isolated_per_pair(fakes, monkeypatch, tmp_path):
    good = _make_pair(dataset_id="D2", pair_name="good_pair")
    corrupt = _make_pair(dataset_id="D2", pair_name="corrupt_pair")
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [corrupt, good])

    results_dir = tmp_path / "results"
    corrupt_file = results_dir / "C1" / "D2" / "corrupt_pair_results.json"
    corrupt_file.parent.mkdir(parents=True)
    corrupt_file.write_text("{ not valid json")

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=results_dir
    )
    assert "D2" in results
    assert (results_dir / "C1" / "D2" / "good_pair_results.json").exists()


def test_multiple_pairs_write_separate_results(fakes, monkeypatch, tmp_path):
    pair_a = _make_pair(dataset_id="D3", pair_name="pair_a")
    pair_b = _make_pair(dataset_id="D3", pair_name="pair_b")
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [pair_a, pair_b])

    results_dir = tmp_path / "results"
    results = runner.run_condition(
        "C1", dataset_ids=["D3"], data_dir=tmp_path, results_dir=results_dir
    )
    assert set(results) == {"D3"}
    assert (results_dir / "C1" / "D3" / "pair_a_results.json").exists()
    assert (results_dir / "C1" / "D3" / "pair_b_results.json").exists()


def test_failing_dataset_does_not_stop_other_datasets(fakes, monkeypatch, tmp_path):
    def selective_dataset(dataset_id, _data_dir):
        if dataset_id == "D1":
            raise DataError("dataset missing")
        return [_make_pair(dataset_id=dataset_id, pair_name=f"{dataset_id.lower()}_pair")]

    monkeypatch.setattr(runner, "load_dataset", selective_dataset)

    results = runner.run_condition(
        "C6",
        dataset_ids=["D1", "D2"],
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert set(results) == {"D2"}


# ---------------------------------------------------------------------------
# run_all_conditions / model caching
# ---------------------------------------------------------------------------


def test_model_loaded_once_per_model_group(fakes, tmp_path):
    runner.run_all_conditions(
        condition_ids=["C3", "C9", "C10"],
        dataset_ids=["D1"],
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert fakes["models"] == ["M2"]
    assert fakes["encoders"] == 1


def test_distinct_models_loaded_once_each(fakes, tmp_path):
    runner.run_all_conditions(
        condition_ids=["C1", "C3"],
        dataset_ids=["D1"],
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert fakes["models"] == ["M1", "M2"]
    assert fakes["encoders"] == 2


def test_unknown_condition_is_skipped_in_run_all(fakes, tmp_path):
    summary = runner.run_all_conditions(
        condition_ids=["C1", "C999"],
        dataset_ids=["D2"],
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert set(summary) == {"C1"}
    assert fakes["models"] == ["M1"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_runs_selected_condition_and_dataset(fakes, tmp_path):
    runner.cli_main(
        [
            "--conditions",
            "C1",
            "--datasets",
            "D2",
            "--data_dir",
            str(tmp_path),
            "--results_dir",
            str(tmp_path / "results"),
        ]
    )
    assert (tmp_path / "results" / "C1" / "D2" / "d2_pair_results.json").exists()


# ---------------------------------------------------------------------------
# Loader unit tests (split logic + candidate loading + end-to-end parsing)
# ---------------------------------------------------------------------------


def test_split_80_10_10_slices_sorted_by_source():
    refs = [(f"s{i:02d}", f"t{i:02d}") for i in range(10)]
    val, test = _split_80_10_10(list(reversed(refs)))
    assert val == [("s08", "t08")]
    assert test == [("s09", "t09")]


def test_split_first_fraction_takes_leading_portion():
    refs = [(f"s{i:02d}", f"t{i:02d}") for i in range(10)]
    val, test = _split_first_fraction(list(reversed(refs)), 0.2)
    assert val == [("s00", "t00"), ("s01", "t01")]
    assert len(test) == 8


def test_load_candidates_roundtrip_and_missing(tmp_path):
    from kgsemembed.candidates import load_candidates

    path = tmp_path / "candidates" / "D1" / "d1_pair_candidates.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"candidates": {"s1": ["t1", "t2"]}}))

    loaded = load_candidates("D1", "d1_pair", tmp_path)
    assert loaded == {"s1": ["t1", "t2"]}

    with pytest.raises(DataError):
        load_candidates("D1", "missing", tmp_path)


_TURTLE = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
<http://ex/{a}> a owl:Class .
<http://ex/{b}> a owl:Class .
"""


def test_load_dataset_d1_reads_direct_val_test_splits(tmp_path):
    from kgsemembed.datasets import load_dataset

    pair_dir = tmp_path / "datasets" / "D1" / "d1_pair"
    (pair_dir / "refs_equiv").mkdir(parents=True)
    (pair_dir / "source.ttl").write_text(_TURTLE.format(a="s1", b="s2"))
    (pair_dir / "target.ttl").write_text(_TURTLE.format(a="t1", b="t2"))
    (pair_dir / "refs_equiv" / "val.tsv").write_text("http://ex/s1\thttp://ex/t1\n")
    (pair_dir / "refs_equiv" / "test.tsv").write_text("http://ex/s2\thttp://ex/t2\n")

    pairs = load_dataset("D1", tmp_path)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.pair_name == "d1_pair"
    assert pair.source_entities == ["http://ex/s1", "http://ex/s2"]
    assert pair.val_refs == [("http://ex/s1", "http://ex/t1")]
    assert pair.test_refs == [("http://ex/s2", "http://ex/t2")]


def test_load_dataset_rejects_unknown_dataset(tmp_path):
    from kgsemembed.datasets import load_dataset

    with pytest.raises(DataError):
        load_dataset("D9", tmp_path)


# ---------------------------------------------------------------------------
# Regression: existing entry points remain intact
# ---------------------------------------------------------------------------


def test_hydra_entrypoints_still_importable():
    from kgsemembed.pipeline.run_experiment import main, run_experiment

    assert callable(run_experiment)
    assert callable(main)


def test_pipeline_package_exports_runner():
    import kgsemembed.pipeline as pipeline

    assert "run_condition" in pipeline.__all__
    assert "run_all_conditions" in pipeline.__all__
