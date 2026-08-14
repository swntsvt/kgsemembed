"""Mocked orchestration tests for the Phase 2 experiment runner.

These tests exercise the full ``run_condition`` / ``run_all_conditions``
pipeline without loading real embedding models or datasets: datasets,
candidates, verbalisers, and the encoder are replaced with deterministic
synthetic doubles.
"""

import dataclasses
import json
import logging
import platform
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph

import kgsemembed.pipeline.run_experiment as runner
from kgsemembed.datasets import AlignmentPair, load_pair_from_dir
from kgsemembed.embeddings.models import MODEL_REGISTRY
from kgsemembed.utils.errors import DataError

_VECTORS = {
    "s1": [1.0, 0.0, 0.0],
    "s2": [0.0, 1.0, 0.0],
    "t1": [1.0, 0.0, 0.0],
    "t2": [0.0, 1.0, 0.0],
}
_CANDIDATES = {"s1": ["t1", "t2"], "s2": ["t1", "t2"]}

_SAMPLE_SOURCE = "http://mouse.owl#MA_0002307"
_SAMPLE_TARGET = "http://human.owl#NCI_C52928"


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
        return object(), {
            "model_key": model_key,
            "model_id": f"fake/{model_key}",
            "device": "cpu",
            "hf_revision": "unknown",
        }

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


@pytest.fixture
def runner_logs(monkeypatch, caplog):
    """
    Capture runner warnings regardless of earlier ``init_logging`` calls.

    ``init_logging`` disables propagation on the ``kgsemembed`` logger, which
    would otherwise keep every record away from the root handler ``caplog``
    installs once any test in the session has initialised logging.
    """
    monkeypatch.setattr(logging.getLogger("kgsemembed"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger=runner._LOGGER.name):
        yield caplog


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
        "apply_ppas",
        "ppas_effective",
        "metrics",
        "n_source_entities",
        "n_candidates_per_entity",
        "versions",
    }
    assert payload["apply_ppas"] is True
    assert payload["ppas_effective"] is True
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
    assert payload["n_candidates_per_entity"] == 2
    assert payload["n_source_entities"] == 2
    assert set(payload["versions"]) == {
        "python",
        "torch",
        "transformers",
        "sentence_transformers",
    }
    assert all(isinstance(value, str) for value in payload["versions"].values())


def test_library_versions_reports_installed_versions():
    versions = runner._library_versions()
    assert versions["python"] == platform.python_version()
    assert versions["sentence_transformers"] != "unknown"


def test_library_versions_marks_missing_distribution_unknown(monkeypatch):
    def raise_not_found(_name):
        raise runner.metadata.PackageNotFoundError

    monkeypatch.setattr(runner.metadata, "version", raise_not_found)
    versions = runner._library_versions()
    assert versions["torch"] == "unknown"
    assert versions["transformers"] == "unknown"
    assert versions["sentence_transformers"] == "unknown"
    assert versions["python"] == platform.python_version()


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


def test_failed_pairs_are_counted_for_the_dataset(fakes, monkeypatch, tmp_path):
    pairs = [_make_pair(dataset_id="D2", pair_name=f"pair_{index}") for index in range(3)]
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: pairs)

    def selective_candidates(_dataset_id, pair_name, _data_dir):
        if pair_name in {"pair_0", "pair_2"}:
            raise DataError("missing candidates")
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_candidates", selective_candidates)

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert results["D2"]["n_failed_pairs"] == 2


def test_successful_metrics_survive_a_failing_pair(fakes, monkeypatch, tmp_path):
    good = _make_pair(dataset_id="D2", pair_name="good_pair")
    bad = _make_pair(dataset_id="D2", pair_name="bad_pair")
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [good, bad])

    def selective_candidates(_dataset_id, pair_name, _data_dir):
        if pair_name == "bad_pair":
            raise DataError("missing candidates")
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_candidates", selective_candidates)

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert results["D2"]["recall_at_1"] == 1.0
    assert results["D2"]["n_failed_pairs"] == 1


def test_clean_dataset_reports_zero_failures(fakes, tmp_path):
    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert results["D2"]["n_failed_pairs"] == 0


def test_fully_failed_dataset_is_still_reported(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner, "load_dataset", lambda _d, _dir: [_make_pair(pair_name="only_pair")]
    )

    def failing_candidates(_dataset_id, _pair_name, _data_dir):
        raise DataError("missing candidates")

    monkeypatch.setattr(runner, "load_candidates", failing_candidates)

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert results["D2"] == {"n_failed_pairs": 1}


def test_dataset_without_pairs_is_absent_from_results(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [])
    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert results == {}


def test_skipped_existing_results_are_not_counted_as_failures(fakes, tmp_path):
    results_dir = tmp_path / "results"
    runner.run_condition("C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=results_dir)

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=results_dir
    )
    assert results["D2"]["n_failed_pairs"] == 0


def test_failure_counts_are_tracked_per_dataset(fakes, monkeypatch, tmp_path):
    def two_pairs(dataset_id, _data_dir):
        return [
            _make_pair(dataset_id=dataset_id, pair_name=f"{dataset_id}_p{index}")
            for index in range(2)
        ]

    def selective_candidates(_dataset_id, pair_name, _data_dir):
        if pair_name.startswith("D1"):
            raise DataError("missing candidates")
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_dataset", two_pairs)
    monkeypatch.setattr(runner, "load_candidates", selective_candidates)

    results = runner.run_condition(
        "C6",
        dataset_ids=["D1", "D2"],
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert results["D1"]["n_failed_pairs"] == 2
    assert results["D2"]["n_failed_pairs"] == 0


def test_dataset_failures_are_logged(fakes, monkeypatch, tmp_path, runner_logs):
    pairs = [_make_pair(dataset_id="D2", pair_name=f"pair_{index}") for index in range(3)]
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: pairs)

    def selective_candidates(_dataset_id, pair_name, _data_dir):
        if pair_name != "pair_1":
            raise DataError("missing candidates")
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_candidates", selective_candidates)

    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert "Dataset D2: 2 of 3 alignment pairs failed" in runner_logs.text


def test_clean_dataset_logs_no_failure_warning(fakes, tmp_path, runner_logs):
    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert "alignment pairs failed" not in runner_logs.text


def test_summary_shows_failure_count_prominently(capsys):
    summary = {"C1": {"D3": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "n_failed_pairs": 7}}}
    runner._print_summary(summary)
    output = capsys.readouterr().out
    assert "failures" in output
    assert "7 failed" in output


def test_summary_shows_zero_failures_for_clean_dataset(capsys):
    summary = {"C1": {"D3": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "n_failed_pairs": 0}}}
    runner._print_summary(summary)
    lines = capsys.readouterr().out.strip().splitlines()
    assert "failed" not in lines[-1]
    assert lines[-1].split()[-1] == "0"


def test_summary_marks_metrics_of_a_fully_failed_dataset(capsys):
    runner._print_summary({"C1": {"D3": {"n_failed_pairs": 21}}})
    output = capsys.readouterr().out
    assert "n/a" in output
    assert "21 failed" in output


def test_summary_distinguishes_failed_from_clean_dataset(capsys):
    metrics = {"f1": 0.5, "precision": 0.5, "recall": 0.5}
    runner._print_summary(
        {
            "C1": {
                "D3": {**metrics, "n_failed_pairs": 7},
                "D4": {**metrics, "n_failed_pairs": 0},
            }
        }
    )
    failed_row, clean_row = capsys.readouterr().out.strip().splitlines()[-2:]
    assert failed_row != clean_row
    assert "7 failed" in failed_row
    assert "failed" not in clean_row


def test_partially_failed_dataset_reported_end_to_end(fakes, monkeypatch, tmp_path, capsys):
    pairs = [_make_pair(dataset_id="D2", pair_name=f"pair_{index}") for index in range(21)]
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: pairs)

    def selective_candidates(_dataset_id, pair_name, _data_dir):
        if int(pair_name.removeprefix("pair_")) < 7:
            raise DataError("missing candidates")
        return dict(_CANDIDATES)

    monkeypatch.setattr(runner, "load_candidates", selective_candidates)

    summary = runner.run_all_conditions(
        condition_ids=["C1"],
        dataset_ids=["D2"],
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert summary["C1"]["D2"]["n_failed_pairs"] == 7
    assert "7 failed" in capsys.readouterr().out


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


def test_load_candidates_roundtrip_and_missing(tmp_path):
    from kgsemembed.candidates import load_candidates

    path = tmp_path / "candidates" / "D1" / "d1_pair_candidates.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"candidates": {"s1": ["t1", "t2"]}}))

    loaded = load_candidates("D1", "d1_pair", tmp_path)
    assert loaded == {"s1": ["t1", "t2"]}

    with pytest.raises(DataError):
        load_candidates("D1", "missing", tmp_path)


def test_load_dataset_rejects_unknown_dataset(tmp_path):
    from kgsemembed.datasets import load_dataset

    with pytest.raises(DataError):
        load_dataset("D9", tmp_path)


# ---------------------------------------------------------------------------
# End-to-end smoke test over the real data/sample/ graphs
# ---------------------------------------------------------------------------


def _sample_pair() -> AlignmentPair:
    """Load data/sample/ and pin one ref pair so the mocked run is deterministic."""
    pair = load_pair_from_dir("data/sample")
    return dataclasses.replace(
        pair,
        dataset_id="D2",
        pair_name="sample_pair",
        val_refs=[(_SAMPLE_SOURCE, _SAMPLE_TARGET)],
        test_refs=[(_SAMPLE_SOURCE, _SAMPLE_TARGET)],
    )


@pytest.mark.integration
def test_sample_pair_loads_real_graphs_from_disk():
    if not Path("data/sample").exists():
        pytest.skip("dataset directory missing: data/sample")
    pair = load_pair_from_dir("data/sample")

    assert pair.source_graph is not None
    assert len(pair.source_graph) > 0
    assert _SAMPLE_SOURCE in pair.source_entities


@pytest.mark.integration
def test_run_condition_over_sample_pair_with_mocked_encoder(monkeypatch, tmp_path):
    if not Path("data/sample").exists():
        pytest.skip("dataset directory missing: data/sample")
    pair = _sample_pair()
    dimension = 4

    class _SampleEncoder:
        def __init__(self, model_key, model):
            self.model_key = model_key

        def encode_batch(self, texts, role="candidate", show_progress=True):
            return np.ones((len(texts), dimension), dtype=np.float32)

    monkeypatch.setattr(
        runner,
        "load_sentence_transformer",
        lambda _key: (object(), {"device": "cpu"}),
    )
    monkeypatch.setattr(runner, "EmbeddingEncoder", _SampleEncoder)
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [pair])
    monkeypatch.setattr(
        runner, "load_candidates", lambda _d, _p, _dir: {_SAMPLE_SOURCE: [_SAMPLE_TARGET]}
    )

    results = runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )

    assert set(results) == {"D2"}
    assert (tmp_path / "results" / "C1" / "D2" / "sample_pair_results.json").exists()


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


# ---------------------------------------------------------------------------
# Candidate count reporting
# ---------------------------------------------------------------------------


def test_candidate_count_is_the_maximum_across_sources():
    candidates = {"s1": ["t1"], "s2": ["t1", "t2", "t3"], "s3": ["t1", "t2"]}
    assert runner._candidate_count(candidates) == 3


def test_candidate_count_ignores_first_source_length():
    candidates = {"s1": [], "s2": ["t1", "t2"]}
    assert runner._candidate_count(candidates) == 2


def test_candidate_count_of_empty_mapping_is_zero():
    assert runner._candidate_count({}) == 0


def test_candidate_count_of_single_source():
    assert runner._candidate_count({"s1": ["t1", "t2", "t3"]}) == 3


def test_candidate_count_of_uniform_lists():
    assert runner._candidate_count({"s1": ["t1", "t2"], "s2": ["t3", "t4"]}) == 2


def test_candidate_count_of_empty_lists_is_zero():
    assert runner._candidate_count({"s1": [], "s2": []}) == 0


def test_candidate_count_reports_the_widest_source():
    candidates = {"a": ["t"] * 20, "b": ["t"] * 18, "c": ["t"] * 25}
    assert runner._candidate_count(candidates) == 25


def test_result_json_records_maximum_candidate_count(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "load_candidates",
        lambda _d, _p, _dir: {"s1": ["t1"], "s2": ["t1", "t2"]},
    )
    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    payload = json.loads(
        (tmp_path / "results" / "C1" / "D2" / "d2_pair_results.json").read_text()
    )
    assert payload["n_candidates_per_entity"] == 2


# ---------------------------------------------------------------------------
# PPAS configuration reporting
# ---------------------------------------------------------------------------


def test_effective_ppas_false_for_model_without_budget():
    assert runner._effective_ppas("M3") is False


def test_effective_ppas_true_for_budgeted_models():
    for model_key in ("M1", "M2", "M4", "M5"):
        assert runner._effective_ppas(model_key) is True


def test_effective_ppas_matches_ppas_disabled_conditions():
    from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS

    for condition in EXPERIMENT_CONDITIONS:
        assert runner._effective_ppas(condition.model_key) is condition.apply_ppas
