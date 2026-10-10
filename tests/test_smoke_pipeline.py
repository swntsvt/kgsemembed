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
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

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

# A synthetic mixed-type pair: four classes and four predicates, each source
# sharing a one-hot direction with its gold target and one distractor.
_MIXED_NS = "http://example.org/mixed#"
_MIXED_WIDTH = 8
_MIXED_INDICES = range(1, 5)


def _mixed_uri(prefix: str, index: int) -> str:
    return f"{_MIXED_NS}{prefix}{index}"


def _one_hot(position: int) -> list:
    vector = [0.0] * _MIXED_WIDTH
    vector[position] = 1.0
    return vector


_MIXED_VECTORS = {
    **{
        _mixed_uri(prefix, index): _one_hot(index - 1)
        for prefix in ("c", "tc")
        for index in _MIXED_INDICES
    },
    **{
        _mixed_uri(prefix, index): _one_hot(3 + index)
        for prefix in ("p", "tp")
        for index in _MIXED_INDICES
    },
}
_MIXED_CANDIDATES = {
    _mixed_uri(source, index): [
        _mixed_uri(target, index),
        _mixed_uri(target, index % 4 + 1),
    ]
    for source, target in (("c", "tc"), ("p", "tp"))
    for index in _MIXED_INDICES
}
_ALL_VECTORS = {**_VECTORS, **_MIXED_VECTORS}


class _FakeVerbaliser:
    def verbalise(self, _graph, entity_uri, _entity_type):
        return str(entity_uri)


class _FakeEncoder:
    def __init__(self, model_key, model):
        self.model_key = model_key
        self.model = model

    def encode_batch(self, texts, role="candidate", show_progress=True):
        return np.array([_ALL_VECTORS[text] for text in texts], dtype=np.float32)


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


def _mixed_graph():
    """Type four sources as classes and four as predicates."""
    graph = Graph()
    for index in _MIXED_INDICES:
        graph.add((URIRef(_mixed_uri("c", index)), RDF.type, OWL.Class))
        graph.add((URIRef(_mixed_uri("p", index)), RDF.type, OWL.ObjectProperty))
    return graph


def _mixed_refs(prefix, indices):
    return [(_mixed_uri(prefix, index), _mixed_uri(f"t{prefix}", index)) for index in indices]


def _mixed_pair(test_refs=None, dataset_id="D3", pair_name="mixed_pair"):
    """Build a mixed-type pair whose classes and predicates both hold refs."""
    return AlignmentPair(
        dataset_id=dataset_id,
        pair_name=pair_name,
        source_graph=_mixed_graph(),
        target_graph=Graph(),
        source_entities=sorted(_MIXED_CANDIDATES),
        val_refs=_mixed_refs("c", [4]) + _mixed_refs("p", [4]),
        test_refs=test_refs or _mixed_refs("c", [1, 2, 3]) + _mixed_refs("p", [1, 2, 3]),
        entity_type="mixed",
    )


def _run_mixed_pair(monkeypatch, tmp_path, pair):
    """Run C1 over a mixed pair and return its written result payload."""
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [pair])
    monkeypatch.setattr(runner, "load_candidates", lambda _d, _p, _dir: dict(_MIXED_CANDIDATES))
    runner.run_condition(
        "C1", dataset_ids=["D3"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    return json.loads(
        (tmp_path / "results" / "C1" / "D3" / f"{pair.pair_name}_results.json").read_text()
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
            "hf_revision": f"revision-{model_key}",
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
    assert metrics["threshold"] == 0.54

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
        "token_budget",
        "metrics",
        "counts",
        "per_entity_type",
        "population",
        "n_source_entities",
        "n_candidates_per_entity",
        "candidates_sha256",
        "hf_revision",
        "kgsemembed_version",
        "python_version",
        "run_timestamp",
        "versions",
        "random_seed",
        "split_seed",
        "evaluation_protocol",
        "git_commit",
        "git_dirty",
    }
    assert payload["apply_ppas"] is True
    assert payload["ppas_effective"] is False
    assert payload["token_budget"] == 200
    assert payload["evaluation_protocol"] == runner.EVALUATION_PROTOCOL_VERSION
    assert set(payload["counts"]) == {"tp", "fp", "fn"}
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
    assert payload["per_entity_type"] == {}
    assert payload["condition_id"] == "C1"
    assert payload["model_id"] == MODEL_REGISTRY["M1"].model_id
    assert payload["n_candidates_per_entity"] == 2
    assert payload["n_source_entities"] == 2
    assert set(payload["versions"]) == {
        "python",
        "torch",
        "transformers",
        "sentence_transformers",
        "numpy",
        "scipy",
        "rdflib",
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


# ---------------------------------------------------------------------------
# Result provenance
# ---------------------------------------------------------------------------


def _run_and_read_payload(tmp_path):
    """Run C1 over D2 with the installed doubles and return the result payload."""
    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    return json.loads(
        (tmp_path / "results" / "C1" / "D2" / "d2_pair_results.json").read_text()
    )


def test_hf_revision_comes_from_the_loaded_model(fakes, tmp_path):
    payload = _run_and_read_payload(tmp_path)
    assert payload["hf_revision"] == "revision-M1"


def test_hf_revision_is_unknown_when_the_loader_reports_none(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner, "load_sentence_transformer", lambda _key: (object(), {"device": "cpu"})
    )
    payload = _run_and_read_payload(tmp_path)
    assert payload["hf_revision"] == "unknown"


def test_package_version_queries_the_kgsemembed_distribution(monkeypatch):
    requested = []

    def record_name(name):
        requested.append(name)
        return "9.9.9"

    monkeypatch.setattr(runner.metadata, "version", record_name)
    assert runner._package_version() == "9.9.9"
    assert requested == ["kgsemembed"]


def test_package_version_reports_the_installed_distribution():
    try:
        installed = runner.metadata.version("kgsemembed")
    except runner.metadata.PackageNotFoundError:
        pytest.skip("kgsemembed is not installed as a distribution")
    assert runner._package_version() == installed


def test_package_version_falls_back_to_unknown(monkeypatch):
    def raise_not_found(_name):
        raise runner.metadata.PackageNotFoundError

    monkeypatch.setattr(runner.metadata, "version", raise_not_found)
    assert runner._package_version() == "unknown"


def test_result_records_package_and_runtime_versions(fakes, tmp_path):
    payload = _run_and_read_payload(tmp_path)
    assert payload["kgsemembed_version"] == runner._package_version()
    assert payload["python_version"] == sys.version
    assert payload["versions"]["python"] == platform.python_version()
    assert payload["versions"]["python"] in payload["python_version"]


def test_run_timestamp_is_utc_iso8601(fakes, tmp_path):
    """
    Bound the timestamp on both sides to reject a local clock labelled ``Z``.

    A one-sided check passes on any machine whose local time runs ahead of UTC,
    so the written instant must fall inside the UTC window spanning the run.
    """
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    payload = _run_and_read_payload(tmp_path)
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    timestamp = payload["run_timestamp"]
    assert timestamp.endswith("Z")
    assert before <= datetime.fromisoformat(timestamp.removesuffix("Z")) <= after


def test_provenance_is_internally_consistent(fakes, tmp_path):
    payload = _run_and_read_payload(tmp_path)
    assert payload["model_key"] == "M1"
    assert payload["model_id"] == MODEL_REGISTRY["M1"].model_id
    assert payload["hf_revision"] == "revision-M1"


def test_each_model_group_records_its_own_revision(fakes, tmp_path):
    """Grouped execution must not attribute one model's revision to another."""
    results_dir = tmp_path / "results"
    runner.run_all_conditions(
        condition_ids=["C1", "C17"],
        dataset_ids=["D2"],
        data_dir=tmp_path,
        results_dir=results_dir,
    )

    assert fakes["models"] == ["M1", "M4"]
    revisions = {
        condition_id: json.loads(
            (results_dir / condition_id / "D2" / "d2_pair_results.json").read_text()
        )["hf_revision"]
        for condition_id in ("C1", "C17")
    }
    assert revisions == {"C1": "revision-M1", "C17": "revision-M4"}


def test_run_timestamp_is_regenerated_on_recompute(fakes, monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    first = _run_and_read_payload(tmp_path)["run_timestamp"]

    recomputed = "2030-01-01T00:00:00Z"
    monkeypatch.setattr(runner, "_utc_timestamp", lambda: recomputed)
    runner.run_condition(
        "C1",
        dataset_ids=["D2"],
        data_dir=tmp_path,
        results_dir=results_dir,
        force_recompute=True,
    )

    payload = json.loads(
        (results_dir / "C1" / "D2" / "d2_pair_results.json").read_text()
    )
    assert payload["run_timestamp"] == recomputed != first


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
# Per-entity-type metrics for mixed pairs
# ---------------------------------------------------------------------------


def test_split_refs_by_entity_type_separates_classes_and_predicates():
    refs = _mixed_refs("c", [1, 2]) + _mixed_refs("p", [1])
    buckets = runner._split_refs_by_entity_type(refs, _mixed_graph())

    assert buckets == {
        "class": _mixed_refs("c", [1, 2]),
        "predicate": _mixed_refs("p", [1]),
    }


def test_split_refs_by_entity_type_omits_empty_buckets():
    buckets = runner._split_refs_by_entity_type(_mixed_refs("c", [1]), _mixed_graph())
    assert list(buckets) == ["class"]


def test_split_refs_by_entity_type_orders_buckets_deterministically():
    refs = _mixed_refs("p", [1]) + _mixed_refs("c", [1])
    buckets = runner._split_refs_by_entity_type(refs, _mixed_graph())
    assert list(buckets) == ["class", "predicate"]


def test_split_refs_by_entity_type_treats_untyped_sources_as_instances():
    refs = [("http://example.org/mixed#untyped", "http://example.org/mixed#t")]
    buckets = runner._split_refs_by_entity_type(refs, _mixed_graph())
    assert list(buckets) == ["instance"]


def test_split_refs_by_entity_type_of_empty_input_is_empty():
    assert runner._split_refs_by_entity_type([], _mixed_graph()) == {}


def _ranked(source_uri, target_uri, score=1.0):
    return [(source_uri, target_uri, score)]


def test_split_ranked_lists_by_entity_type_groups_by_source_type():
    class_list = _ranked(_mixed_uri("c", 1), _mixed_uri("tc", 1))
    predicate_list = _ranked(_mixed_uri("p", 1), _mixed_uri("tp", 1))

    buckets = runner._split_ranked_lists_by_entity_type(
        [class_list, predicate_list], _mixed_graph()
    )
    assert buckets == {"class": [class_list], "predicate": [predicate_list]}


def test_split_ranked_lists_keeps_sources_carrying_no_reference():
    """A ranked source without a gold target still belongs to its own bucket."""
    lists = [_ranked(_mixed_uri("c", index), _mixed_uri("tc", index)) for index in (1, 2)]
    buckets = runner._split_ranked_lists_by_entity_type(lists, _mixed_graph())

    assert len(buckets["class"]) == 2
    assert "predicate" not in buckets


def test_split_ranked_lists_ignores_empty_ranked_lists():
    assert runner._split_ranked_lists_by_entity_type([[]], _mixed_graph()) == {}


def test_buckets_do_not_borrow_each_others_predictions():
    """A predicate false positive must not be charged against class precision."""
    ranked_lists = [
        _ranked(_mixed_uri("c", index), _mixed_uri("tc", index)) for index in (1, 2, 3)
    ] + [_ranked(_mixed_uri("p", index), _mixed_uri("tp", 9)) for index in (1, 2, 3)]
    pair = dataclasses.replace(_mixed_pair(), val_refs=[])

    breakdown = runner._per_entity_type_metrics(pair, {"val": [], "test": ranked_lists}, 0.5)

    assert breakdown["class"]["precision"] == 1.0
    assert breakdown["class"]["f1"] == 1.0
    assert breakdown["predicate"]["precision"] == 0.0
    assert breakdown["predicate"]["recall"] == 0.0


def test_buckets_without_validation_refs_use_the_supplied_threshold():
    pair = dataclasses.replace(_mixed_pair(), val_refs=[])
    ranked_lists = [
        _ranked(_mixed_uri("c", index), _mixed_uri("tc", index)) for index in (1, 2, 3)
    ] + [_ranked(_mixed_uri("p", index), _mixed_uri("tp", index)) for index in (1, 2, 3)]

    breakdown = runner._per_entity_type_metrics(pair, {"val": [], "test": ranked_lists}, 0.75)
    assert [metrics["threshold"] for metrics in breakdown.values()] == [0.75, 0.75]


def test_n_refs_counts_each_bucket_separately():
    """Buckets of different sizes must each report their own reference count."""
    test_refs = _mixed_refs("c", [1, 2, 3, 4]) + _mixed_refs("p", [1, 2, 3])
    pair = dataclasses.replace(_mixed_pair(), val_refs=[], test_refs=test_refs)
    ranked_lists = [
        _ranked(_mixed_uri(prefix, index), _mixed_uri(f"t{prefix}", index))
        for prefix in ("c", "p")
        for index in _MIXED_INDICES
    ]

    breakdown = runner._per_entity_type_metrics(pair, {"val": [], "test": ranked_lists}, 0.5)
    assert breakdown["class"]["n_refs"] == 4
    assert breakdown["predicate"]["n_refs"] == 3
    assert sum(metrics["n_refs"] for metrics in breakdown.values()) == len(test_refs)


def test_buckets_decompose_the_pair_level_recall():
    """Under one shared threshold the buckets must reconstruct the pair recall."""
    pair = dataclasses.replace(_mixed_pair(), val_refs=[])
    hits = [
        _ranked(_mixed_uri(prefix, index), _mixed_uri(f"t{prefix}", index))
        for prefix, indices in (("c", (1, 2)), ("p", (1,)))
        for index in indices
    ]
    misses = [
        _ranked(_mixed_uri(prefix, index), _mixed_uri(f"t{prefix}", 9))
        for prefix, indices in (("c", (3,)), ("p", (2, 3)))
        for index in indices
    ]
    ranked_lists = hits + misses

    overall = runner.compute_all_metrics(ranked_lists, pair.test_refs, threshold=0.5)
    breakdown = runner._per_entity_type_metrics(pair, {"val": [], "test": ranked_lists}, 0.5)

    recovered = sum(
        metrics["recall"] * metrics["n_refs"] for metrics in breakdown.values()
    )
    assert recovered == pytest.approx(overall["recall"] * len(pair.test_refs))
    assert overall["recall"] == pytest.approx(0.5)


def test_non_mixed_pair_is_never_split():
    """A class-typed pair must not be split even when its graph is mixed."""
    pair = dataclasses.replace(_mixed_pair(), entity_type="class")
    ranked_lists = [_ranked(_mixed_uri("c", 1), _mixed_uri("tc", 1))]
    assert runner._per_entity_type_metrics(pair, {"val": [], "test": ranked_lists}, 0.5) == {}


def test_mixed_pair_records_metrics_per_entity_type(fakes, monkeypatch, tmp_path):
    payload = _run_mixed_pair(monkeypatch, tmp_path, _mixed_pair())

    breakdown = payload["per_entity_type"]
    assert set(breakdown) == {"class", "predicate"}
    for metrics in breakdown.values():
        assert set(metrics) == {*runner._METRIC_KEYS, *runner._COUNT_KEYS, "n_refs"}
        assert metrics["n_refs"] == 3
        assert metrics["recall_at_1"] == 1.0


def test_mixed_breakdown_leaves_top_level_metrics_unchanged(fakes, monkeypatch, tmp_path):
    pair = _mixed_pair()
    payload = _run_mixed_pair(monkeypatch, tmp_path, pair)

    monkeypatch.setattr(runner, "_per_entity_type_metrics", lambda *_args: {})
    baseline = _run_mixed_pair(
        monkeypatch, tmp_path / "baseline", dataclasses.replace(pair)
    )
    assert payload["metrics"] == baseline["metrics"]
    assert baseline["per_entity_type"] == {}


def test_non_mixed_pair_records_an_empty_breakdown(fakes, tmp_path):
    runner.run_condition(
        "C1", dataset_ids=["D2"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    payload = json.loads(
        (tmp_path / "results" / "C1" / "D2" / "d2_pair_results.json").read_text()
    )
    assert payload["per_entity_type"] == {}


def test_small_entity_type_bucket_is_skipped(fakes, monkeypatch, tmp_path):
    pair = _mixed_pair(test_refs=_mixed_refs("c", [1, 2, 3]) + _mixed_refs("p", [1, 2]))
    payload = _run_mixed_pair(monkeypatch, tmp_path, pair)
    assert set(payload["per_entity_type"]) == {"class"}


def test_small_entity_type_bucket_logs_a_warning(fakes, monkeypatch, tmp_path, runner_logs):
    pair = _mixed_pair(test_refs=_mixed_refs("c", [1, 2, 3]) + _mixed_refs("p", [1, 2]))
    _run_mixed_pair(monkeypatch, tmp_path, pair)
    assert "Skipping predicate entity-type metrics: 2 reference pairs" in runner_logs.text


def test_entity_type_buckets_are_tuned_independently(fakes, monkeypatch, tmp_path):
    """Each bucket must tune on its own validation refs, not the pair's."""
    tuned = []

    def recording_tune(_scored_pairs, references):
        tuned.append(sorted(source for source, _ in references))
        return {"best_threshold": 0.1}

    monkeypatch.setattr(runner, "tune_threshold", recording_tune)
    _run_mixed_pair(monkeypatch, tmp_path, _mixed_pair())

    assert tuned[0] == sorted([_mixed_uri("c", 4), _mixed_uri("p", 4)])
    assert tuned[1:] == [[_mixed_uri("c", 4)], [_mixed_uri("p", 4)]]


def test_tuning_never_observes_a_test_source(fakes, monkeypatch, tmp_path):
    """No scored pair of a test source may reach the threshold grid search."""
    scored_sources = set()

    def recording_tune(scored_pairs, _references):
        scored_sources.update(source for source, _, _ in scored_pairs)
        return {"best_threshold": 0.1}

    monkeypatch.setattr(runner, "tune_threshold", recording_tune)
    pair = _mixed_pair()
    _run_mixed_pair(monkeypatch, tmp_path, pair)

    assert scored_sources == {source for source, _ in pair.val_refs}
    assert scored_sources.isdisjoint({source for source, _ in pair.test_refs})


def test_bucket_without_validation_refs_reuses_the_pair_threshold(fakes, monkeypatch, tmp_path):
    pair = dataclasses.replace(_mixed_pair(), val_refs=_mixed_refs("c", [4]))
    payload = _run_mixed_pair(monkeypatch, tmp_path, pair)

    overall = payload["metrics"]["threshold"]
    assert payload["per_entity_type"]["predicate"]["threshold"] == overall
    assert overall != runner._DEFAULT_THRESHOLD


def test_bucket_with_validation_refs_may_diverge_from_the_pair_threshold(
    fakes, monkeypatch, tmp_path
):
    """A bucket tuning on its own refs is free to pick another threshold."""
    thresholds = iter([0.2, 0.4, 0.6])
    monkeypatch.setattr(
        runner, "tune_threshold", lambda *_args: {"best_threshold": next(thresholds)}
    )
    payload = _run_mixed_pair(monkeypatch, tmp_path, _mixed_pair())

    assert payload["metrics"]["threshold"] == 0.2
    assert payload["per_entity_type"]["class"]["threshold"] == 0.4
    assert payload["per_entity_type"]["predicate"]["threshold"] == 0.6


def test_per_entity_type_is_absent_from_the_returned_summary(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_dataset", lambda _d, _dir: [_mixed_pair()])
    monkeypatch.setattr(runner, "load_candidates", lambda _d, _p, _dir: dict(_MIXED_CANDIDATES))

    results = runner.run_condition(
        "C1", dataset_ids=["D3"], data_dir=tmp_path, results_dir=tmp_path / "results"
    )
    assert "per_entity_type" not in results["D3"]
    assert results["D3"]["recall_at_1"] == 1.0


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


_SHARED_C20_C10_FIELDS = (
    "model_key",
    "model_id",
    "hf_revision",
    "apply_ppas",
    "ppas_effective",
    "n_candidates_per_entity",
    "n_source_entities",
    "versions",
)


def _run_c20_and_c10(results_dir: Path, data_dir: Path) -> dict:
    """Run C20 and C10 on D1 together and return each condition's payload."""
    runner.run_all_conditions(
        condition_ids=["C20", "C10"],
        dataset_ids=["D1"],
        data_dir=data_dir,
        results_dir=results_dir,
    )
    return {
        condition_id: json.loads(
            (results_dir / condition_id / "D1" / "d1_pair_results.json").read_text()
        )
        for condition_id in ("C20", "C10")
    }


def test_c20_and_c10_share_model_and_provenance(fakes: dict, tmp_path: Path) -> None:
    payloads = _run_c20_and_c10(tmp_path / "results", tmp_path)
    assert fakes["models"] == ["M2"]
    assert (payloads["C20"]["strategy"], payloads["C10"]["strategy"]) == ("V2", "V2+V8")
    for field in _SHARED_C20_C10_FIELDS:
        assert payloads["C20"][field] == payloads["C10"][field]


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
        lambda _key: (object(), {"device": "cpu", "hf_revision": "sample-revision"}),
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


def _condition_with(strategy_name, model_key):
    from kgsemembed.pipeline.conditions import get_condition

    return dataclasses.replace(
        get_condition("C3"), strategy_name=strategy_name, model_key=model_key
    )


def test_effective_ppas_false_for_model_without_budget():
    assert runner._effective_ppas(_condition_with("V2+V6", "M3")) is False


def test_effective_ppas_true_for_ppas_strategies_on_budgeted_models():
    for strategy in ("V3", "V4", "V6", "V2+V6", "V6+V3", "V4+V6", "V8+V6"):
        for model_key in ("M1", "M2", "M4", "M5"):
            assert runner._effective_ppas(_condition_with(strategy, model_key)) is True


def test_effective_ppas_false_for_strategies_without_ppas():
    for strategy in ("V1", "V2", "V5", "V7", "V8", "V2+V8", "V2+V7", "V2+V8+V7"):
        assert runner._effective_ppas(_condition_with(strategy, "M2")) is False


def test_effective_ppas_of_registered_conditions():
    from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS

    expected = {"C3", "C4", "C5", "C7"}
    effective = {
        condition.condition_id
        for condition in EXPERIMENT_CONDITIONS
        if runner._effective_ppas(condition)
    }
    assert effective == expected


def test_c19_and_c10_report_identical_ppas_execution():
    from kgsemembed.pipeline.conditions import get_condition

    assert runner._effective_ppas(get_condition("C19")) is False
    assert runner._effective_ppas(get_condition("C10")) is False


# ---------------------------------------------------------------------------
# Evaluation populations
# ---------------------------------------------------------------------------


def _population_pair(source_entities, train_refs=()):
    return AlignmentPair(
        dataset_id="D1",
        pair_name="population_pair",
        source_graph=Graph(),
        target_graph=Graph(),
        source_entities=list(source_entities),
        val_refs=[("v", "x")],
        test_refs=[("t", "y")],
        train_refs=list(train_refs),
    )


def _unmatched_uri(in_validation):
    from kgsemembed.evaluation.population import draws_validation

    return next(
        uri
        for uri in (f"http://example.org/u{index}" for index in range(1000))
        if draws_validation(uri, 0.5) is in_validation
    )


_VAL_AND_TEST_LISTS = [
    [("v", "x", 0.9), ("v", "n", 0.2)],
    [("t", "y", 0.9), ("t", "n", 0.2)],
]


def test_validation_predictions_are_not_test_false_positives():
    metrics = runner._evaluate_pair(_population_pair(["v", "t"]), _VAL_AND_TEST_LISTS)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 0)
    assert metrics["precision"] == 1.0
    assert metrics["mrr"] == 1.0


def test_training_source_predictions_are_not_counted():
    pair = _population_pair(["v", "t", "r"], train_refs=[("r", "z")])
    ranked = _VAL_AND_TEST_LISTS + [[("r", "z", 0.99)]]
    metrics = runner._evaluate_pair(pair, ranked)
    assert metrics["fp"] == 0
    assert metrics["population"]["n_test_ranked_sources"] == 1


def test_unmatched_test_source_prediction_is_a_false_positive():
    uri = _unmatched_uri(in_validation=False)
    ranked = _VAL_AND_TEST_LISTS + [[(uri, "w", 0.95)]]
    metrics = runner._evaluate_pair(_population_pair(["v", "t", uri]), ranked)
    assert metrics["fp"] == 1
    assert metrics["population"]["n_test_unmatched_sources"] == 1


def test_unmatched_validation_source_raises_the_tuned_threshold():
    uri = _unmatched_uri(in_validation=True)
    ranked = _VAL_AND_TEST_LISTS + [[(uri, "w", 0.7)]]
    with_unmatched = runner._evaluate_pair(_population_pair(["v", "t", uri]), ranked)
    without = runner._evaluate_pair(_population_pair(["v", "t"]), _VAL_AND_TEST_LISTS)
    assert with_unmatched["threshold"] > 0.7
    assert without["threshold"] < 0.7
    assert with_unmatched["fp"] == 0


def test_mrr_is_averaged_over_test_reference_sources_only():
    uri = _unmatched_uri(in_validation=False)
    ranked = [
        [("v", "x", 0.9)],
        [("t", "n", 0.9), ("t", "y", 0.8)],
        [(uri, "w", 0.95)],
    ]
    metrics = runner._evaluate_pair(_population_pair(["v", "t", uri]), ranked)
    assert metrics["mrr"] == 0.5


def test_population_record_reports_every_count():
    uri = _unmatched_uri(in_validation=False)
    pair = _population_pair(["v", "t", "r", uri], train_refs=[("r", "z")])
    ranked = _VAL_AND_TEST_LISTS + [[("r", "z", 0.9)], [(uri, "w", 0.3)]]
    assert runner._evaluate_pair(pair, ranked)["population"] == {
        "n_train_refs": 1,
        "n_val_refs": 1,
        "n_test_refs": 1,
        "n_val_ranked_sources": 1,
        "n_test_ranked_sources": 2,
        "n_val_unmatched_sources": 0,
        "n_test_unmatched_sources": 1,
        "reference_complete": True,
    }


def test_partial_reference_ignores_unjudgeable_unmatched_predictions():
    uri = _unmatched_uri(in_validation=False)
    pair = dataclasses.replace(_population_pair(["v", "t", uri]), reference_complete=False)
    ranked = _VAL_AND_TEST_LISTS + [[(uri, "w", 0.95)]]
    metrics = runner._evaluate_pair(pair, ranked)
    assert metrics["fp"] == 0
    assert metrics["population"]["n_test_ranked_sources"] == 1


def test_partial_reference_counts_predictions_that_contradict_gold():
    uri = _unmatched_uri(in_validation=False)
    pair = dataclasses.replace(_population_pair(["v", "t", uri]), reference_complete=False)
    ranked = _VAL_AND_TEST_LISTS + [[(uri, "y", 0.95)]]
    metrics = runner._evaluate_pair(pair, ranked)
    assert metrics["fp"] == 1


def test_buckets_score_only_the_test_population():
    pair = _mixed_pair()
    ranked_lists = [
        _ranked(_mixed_uri(prefix, index), _mixed_uri(f"t{prefix}", index))
        for prefix in ("c", "p")
        for index in _MIXED_INDICES
    ]
    metrics = runner._evaluate_pair(pair, ranked_lists)
    for bucket in metrics["per_entity_type"].values():
        assert bucket["fp"] == 0
        assert bucket["tp"] == 3


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_candidate_digest_is_order_independent_and_content_sensitive():
    first = runner._candidate_digest({"a": ["x", "y"], "b": ["z"]})
    reordered = runner._candidate_digest({"b": ["z"], "a": ["x", "y"]})
    changed = runner._candidate_digest({"a": ["y", "x"], "b": ["z"]})
    assert first == reordered
    assert first != changed


def test_git_state_reports_unknown_outside_a_work_tree(monkeypatch):
    monkeypatch.setattr(runner, "_git_output", lambda *_args: None)
    assert runner._git_state() == {"git_commit": "unknown", "git_dirty": None}


def test_git_state_reads_the_repository():
    state = runner._git_state()
    assert len(state["git_commit"]) == 40
    assert isinstance(state["git_dirty"], bool)


def test_result_records_seeds_and_candidate_digest(fakes, tmp_path):
    payload = _run_and_read_payload(tmp_path)
    assert payload["random_seed"] == 42
    assert payload["split_seed"] == 42
    assert payload["candidates_sha256"] == runner._candidate_digest(dict(_CANDIDATES))
    assert payload["counts"]["tp"] + payload["counts"]["fn"] == 1


def test_stale_protocol_result_is_read_with_a_warning(tmp_path, runner_logs):
    path = tmp_path / "stale_results.json"
    path.write_text(json.dumps({"metrics": {"f1": 0.5}}), encoding="utf-8")
    assert runner._read_metrics(path) == {"f1": 0.5}
    assert "evaluation protocol" in runner_logs.text


def test_current_protocol_result_is_read_silently(tmp_path, runner_logs):
    path = tmp_path / "current_results.json"
    payload = {"metrics": {"f1": 0.5}, "evaluation_protocol": runner.EVALUATION_PROTOCOL_VERSION}
    path.write_text(json.dumps(payload), encoding="utf-8")
    runner._read_metrics(path)
    assert "evaluation protocol" not in runner_logs.text
