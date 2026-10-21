"""Tests for the controlled PPAS ablation registry and runner (#122, #128)."""

import hashlib
from dataclasses import replace
import sys
import json
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

import kgsemembed.pipeline.ppas_ablation as ablation
from kgsemembed.datasets import AlignmentPair
from kgsemembed.embeddings.models import get_model_config
from kgsemembed.utils.errors import ConfigurationError, DataError
from kgsemembed.pipeline.ablation_conditions import (
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    BASE_POLICIES,
    CONTROLLED_BUDGET,
    CONTROLLED_POLICIES,
    EXTENDED_VOCABULARIES,
    PRIMARY_BUDGET,
    RANDOM_SEEDS,
    STRESS_BUDGETS,
    VALIDATION_BUDGETS,
    _validate_registry,
    extended_policies,
    get_ablation_condition,
    stress_budget,
)
from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS
from kgsemembed.verbalisation.selection import DETERMINISTIC, FULL, PPAS, RANDOM, Selection

EX = "http://example.org/"
_DIM = 32
_N_ATTRIBUTES = 30
_CONDITIONS = ["PA-full", "PA-ppas-80", "PA-det-80", "PA-rand-80-s0", "PA-rand-80-s1"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_ids_are_unique_and_share_one_model():
    ids = [c.condition_id for c in ABLATION_CONDITIONS]
    assert len(ids) == len(set(ids)) == 22 + len(CONTROLLED_POLICIES) + 3 + 2 * len(BASE_POLICIES) + 24 == 64
    assert {c.model_key for c in ABLATION_CONDITIONS} == {ABLATION_MODEL_KEY} == {"M2"}


def test_each_budget_has_ppas_deterministic_and_every_random_seed():
    historical = [c for c in ABLATION_CONDITIONS if not c.is_controlled and not c.is_role_ablation]
    budgets = {c.token_budget for c in historical} - {None}
    assert budgets == {PRIMARY_BUDGET, *STRESS_BUDGETS.values()}
    for budget in budgets:
        family = [c for c in historical if c.token_budget == budget]
        assert sorted(c.policy for c in family if c.policy != RANDOM) == [DETERMINISTIC, PPAS]
        assert sorted(c.seed for c in family if c.policy == RANDOM) == list(RANDOM_SEEDS)
        assert len({c.datasets for c in family}) == 1


def _controlled_on(dataset_id: str, budget: int) -> list:
    return [c for c in ABLATION_CONDITIONS
            if c.is_controlled and dataset_id in c.datasets and c.token_budget == budget]


def test_controlled_policies_are_p1_to_p7_on_d1_at_48():
    controlled = _controlled_on("D1", CONTROLLED_BUDGET)
    assert [c.policy_label for c in controlled] == [f"P{i}" for i in range(1, 8)]
    assert CONTROLLED_BUDGET == 48
    dimensions = {(c.filter_mode, c.predicate_config, c.order_mode) for c in controlled}
    assert len(dimensions) == len(controlled)
    assert ("all", "fma_extended", DETERMINISTIC) not in dimensions


def test_validation_budgets_register_only_the_base_policies_on_d1():
    assert VALIDATION_BUDGETS == (32, 64)
    for budget in VALIDATION_BUDGETS:
        family = _controlled_on("D1", budget)
        assert [(c.policy_label, c.filter_mode, c.predicate_config, c.order_mode) for c in family] == list(BASE_POLICIES)
        assert all(c.condition_id.endswith(f"-{budget}") and c.datasets == ("D1",) for c in family)
    assert {c.token_budget for c in ABLATION_CONDITIONS if c.is_controlled} == {32, 48, 64}


def test_d2_runs_base_policies_and_its_own_extended_vocabulary_at_48():
    family = _controlled_on("D2", 48)
    assert [c.policy_label for c in family] == [f"P{i}" for i in range(1, 8)]
    vocabularies = {c.policy_label: c.predicate_config for c in family}
    assert {vocabularies[f"P{i}"] for i in range(1, 5)} == {"base"}
    assert {vocabularies[f"P{i}"] for i in range(5, 8)} == {"obo_extended"} == {EXTENDED_VOCABULARIES["D2"]}
    assert not any(c.predicate_config == "fma_extended" for c in family)
    assert not _controlled_on("D2", 32) and not _controlled_on("D2", 64)


def test_extended_policies_keep_the_p5_to_p7_design():
    designs = [(label, f, o) for label, f, _v, o in extended_policies("obo_extended")]
    assert designs == [(label, f, o) for label, f, v, o in CONTROLLED_POLICIES if v == "fma_extended"]


def test_a_label_repeated_within_one_dataset_and_budget_raises():
    duplicate = get_ablation_condition("PA-all-base-det-48")
    with pytest.raises(ValueError, match="Duplicate controlled policy labels"):
        _validate_registry(ABLATION_CONDITIONS + [replace(duplicate, condition_id="PA-x")])


def test_controlled_condition_ids_name_every_dimension():
    ids = {c.policy_label: c.condition_id for c in _controlled_on("D1", 48)}
    assert ids == {
        "P1": "PA-all-base-det-48", "P2": "PA-all-base-ppas-48", "P3": "PA-cov-base-det-48",
        "P4": "PA-cov-base-ppas-48", "P5": "PA-all-fmax-ppas-48", "P6": "PA-cov-fmax-det-48",
        "P7": "PA-cov-fmax-ppas-48",
    }


def test_legacy_conditions_default_to_all_base_and_their_policy_order():
    for condition in ABLATION_CONDITIONS:
        assert condition.order_mode == condition.policy
        if not condition.is_controlled:
            assert (condition.filter_mode, condition.predicate_config) == ("all", "base")


def test_random_seeds_are_at_least_five():
    assert len(RANDOM_SEEDS) >= 5


def test_primary_budget_runs_on_d5_only_and_d1_uses_its_stress_budget():
    primary = [c for c in ABLATION_CONDITIONS if c.token_budget == PRIMARY_BUDGET]
    assert {c.datasets for c in primary} == {("D5",)}
    d1_budgets = {c.token_budget for c in ABLATION_CONDITIONS if "D1" in c.datasets and not c.is_controlled}
    assert d1_budgets == {None, STRESS_BUDGETS["D1"]}


def test_budgets_fit_the_model_input_limit():
    limit = get_model_config(ABLATION_MODEL_KEY).max_tokens
    assert all(c.token_budget <= limit for c in ABLATION_CONDITIONS if c.token_budget)


def test_only_full_is_uncapped_and_only_random_is_seeded():
    for condition in ABLATION_CONDITIONS:
        assert (condition.policy == FULL) == (condition.token_budget is None)
        assert (condition.policy == RANDOM) == (condition.seed is not None)


def test_historical_registry_is_unchanged():
    assert len(EXPERIMENT_CONDITIONS) == 19
    assert not any(c.condition_id.startswith("PA-") for c in EXPERIMENT_CONDITIONS)


def test_unknown_ablation_condition_raises():
    with pytest.raises(KeyError):
        get_ablation_condition("PA-nope")


@pytest.mark.parametrize(
    "lengths,expected", [([10, 90, 100], 88), ([40, 50, 60, 70], 48), ([7], 0), ([95, 97], 96)]
)
def test_stress_budget_floors_the_median_to_a_multiple_of_eight(lengths, expected):
    assert stress_budget(lengths) == expected


# ---------------------------------------------------------------------------
# Runner fixtures: a synthetic instance pair, whitespace tokenizer, hashed model
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    name_or_path = "fake/tokenizer"

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}


class _FakeModel:
    """Embed text as a normalised hashed bag of words."""

    tokenizer = _FakeTokenizer()
    max_seq_length = 512

    def __init__(self):
        self.encoded = []

    def encode(self, texts, **_kwargs):
        self.encoded.append(list(texts))
        return np.stack([_embed(text) for text in texts])


def _embed(text: str) -> np.ndarray:
    vector = np.zeros(_DIM, dtype=np.float32)
    vector[0] = 0.1
    for word in text.split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM] += 1.0
    return vector / np.linalg.norm(vector)


def _entity_graph(prefix: str, n: int) -> Graph:
    """Entities sharing their index words, each with varied-length attributes."""
    graph = Graph()
    for i in range(n):
        entity = URIRef(f"{EX}{prefix}{i}")
        graph.add((entity, RDF.type, URIRef(f"{EX}Film")))
        for j in range(_N_ATTRIBUTES):
            words = " ".join(f"k{i}w{j}x{r}" for r in range(j % 3 + 1))
            graph.add((entity, URIRef(f"{EX}attr{j}"), Literal(words)))
    return graph


def _pair() -> AlignmentPair:
    sources = [f"{EX}s{i}" for i in range(8)]
    refs = [(f"{EX}s{i}", f"{EX}t{i}") for i in range(8)]
    return AlignmentPair(
        dataset_id="D5", pair_name="toy", source_graph=_entity_graph("s", 8),
        target_graph=_entity_graph("t", 8), source_entities=sources,
        val_refs=refs[:3], test_refs=refs[3:], entity_type="instance",
    )


@pytest.fixture
def toy_run(monkeypatch, tmp_path):
    """Run the given conditions on the toy pair and return payloads and the model."""
    model = _FakeModel()
    info = {"model_key": "M2", "model_id": "fake/M2", "device": "mps", "hf_revision": "rev-1"}
    candidates = {f"{EX}s{i}": [f"{EX}t{j}" for j in range(8)] for i in range(8)}
    monkeypatch.setattr(ablation, "load_sentence_transformer", lambda _key: (model, info))
    monkeypatch.setattr(ablation, "load_dataset", lambda _d, _dir: [_pair()])
    monkeypatch.setattr(ablation, "load_candidates", lambda _d, _p, _dir: dict(candidates))
    results_dir = tmp_path / "ablation"

    def run(condition_ids=_CONDITIONS, force=False, max_sources=None):
        kwargs = {"data_dir": tmp_path, "results_dir": results_dir,
                  "force_recompute": force, "max_sources": max_sources}
        return ablation.run_ablation(["D5"], condition_ids, kwargs)

    return run, model, results_dir


def _read(results_dir: Path, condition_id: str, suffix: str = "results") -> dict:
    return json.loads((results_dir / condition_id / "D5" / f"toy_{suffix}.json").read_text())


# ---------------------------------------------------------------------------
# Runner behaviour
# ---------------------------------------------------------------------------


def test_ablation_result_records_the_device(toy_run):
    run, _model, results_dir = toy_run
    run([_CONDITIONS[0]])
    payload = _read(results_dir, _CONDITIONS[0])
    assert (payload["device"], payload["device_override"]) == ("mps", False)


def test_ablation_refuses_a_non_mps_device(monkeypatch, tmp_path):
    info = {"model_key": "M2", "model_id": "fake/M2", "device": "cpu", "hf_revision": "r"}
    monkeypatch.setattr(ablation, "load_sentence_transformer", lambda _key: (_FakeModel(), info))
    kwargs = {"data_dir": tmp_path, "results_dir": tmp_path / "ablation",
              "force_recompute": False, "max_sources": None}
    with pytest.raises(ConfigurationError, match="allow_device"):
        ablation.run_ablation(["D5"], [_CONDITIONS[0]], kwargs)


def test_run_writes_result_and_sidecar_for_every_condition(toy_run):
    run, _model, results_dir = toy_run
    payloads = run()
    assert [p["condition_id"] for p in payloads] == _CONDITIONS
    for condition_id in _CONDITIONS:
        assert _read(results_dir, condition_id)["ablation"]["status"] == "completed"
        assert _read(results_dir, condition_id, "entities")["condition_id"] == condition_id


def test_result_keeps_the_main_schema_and_records_provenance(toy_run):
    run, _model, results_dir = toy_run
    run()
    payload = _read(results_dir, "PA-rand-80-s1")
    for key in ("metrics", "counts", "population", "candidates_sha256", "hf_revision",
                "git_commit", "versions", "evaluation_protocol", "random_seed"):
        assert key in payload
    block = payload["ablation"]
    assert (block["policy"], block["seed"], block["nominal_token_budget"]) == (RANDOM, 1, 80)
    assert block["tokenizer"] == "fake/tokenizer" and block["tokenizer_revision"] == "rev-1"
    assert block["budget_includes_special_tokens"] is True
    for key in ("tier_config_sha256", "selection_sha256", "runtime", "encoder", "threshold_procedure"):
        assert key in block


def test_controlled_settings_are_identical_across_conditions(toy_run):
    run, _model, _results_dir = toy_run
    payloads = run()
    shared = ("candidates_sha256", "model_id", "hf_revision", "n_candidates_per_entity", "dataset_id")
    for key in shared:
        assert len({p[key] for p in payloads}) == 1, key
    for key in ("tokenizer", "encoder", "threshold_procedure", "representation", "tier_config_sha256"):
        assert len({json.dumps(p["ablation"][key], sort_keys=True) for p in payloads}) == 1, key
    assert len({json.dumps(p["population"], sort_keys=True) for p in payloads}) == 1


def test_capped_conditions_respect_the_budget_and_record_sampling(toy_run):
    run, _model, _results_dir = toy_run
    for payload in run():
        for side in ("source", "candidate"):
            summary = payload["ablation"][side]
            if payload["token_budget"] is not None:
                assert summary["tokens"]["max"] <= payload["token_budget"]
                assert summary["n_sampling_invoked"] == summary["n_entities"]
            else:
                assert summary["n_sampling_invoked"] == 0


def test_ppas_effective_reflects_executed_ppas_only(toy_run):
    run, _model, _results_dir = toy_run
    effective = {p["condition_id"]: p["ppas_effective"] for p in run()}
    assert effective == {"PA-full": False, "PA-ppas-80": True, "PA-det-80": False,
                         "PA-rand-80-s0": False, "PA-rand-80-s1": False}


def test_random_seeds_produce_different_selections(toy_run):
    run, _model, _results_dir = toy_run
    digests = {p["condition_id"]: p["ablation"]["selection_sha256"] for p in run()}
    assert len(set(digests.values())) == len(digests)


def test_unsampled_entities_reuse_the_full_reference_embeddings(toy_run):
    run, _model, _results_dir = toy_run
    full, ppas = run(["PA-full", "PA-ppas-420"])
    assert ppas["ablation"]["source"]["n_sampling_invoked"] == 0
    assert ppas["ablation"]["selection_sha256"] == full["ablation"]["selection_sha256"]
    for side in ("source", "candidate"):
        assert ppas["ablation"]["runtime"][side]["n_texts_encoded"] == 0
    assert ppas["metrics"] == full["metrics"]


def test_ppas_and_deterministic_differ_when_tiered_predicates_exist(toy_run):
    run, _model, _results_dir = toy_run
    ppas, det = run(["PA-ppas-80", "PA-det-80"])
    assert ppas["ablation"]["selection_sha256"] != det["ablation"]["selection_sha256"]


def test_sidecar_ranks_reproduce_the_reported_mrr(toy_run):
    run, _model, results_dir = toy_run
    run()
    for condition_id in _CONDITIONS:
        sources = _read(results_dir, condition_id, "entities")["test_sources"]
        rr = [1.0 / s["best_rank"] if s["best_rank"] else 0.0 for s in sources]
        assert np.mean(rr) == pytest.approx(_read(results_dir, condition_id)["metrics"]["mrr"])


def _pair_with_known_pair_above_gold() -> AlignmentPair:
    """Give test source s3 a known target that ties t3's text and sorts before it."""
    pair = _pair()
    known = URIRef(f"{EX}t2dup")
    for _s, predicate, obj in list(pair.target_graph.triples((URIRef(f"{EX}t3"), None, None))):
        pair.target_graph.add((known, predicate, obj))
    return replace(pair, train_refs=[(f"{EX}s3", str(known))])


def test_sidecar_ranks_skip_known_pairs_like_the_mrr(toy_run, monkeypatch):
    run, _model, results_dir = toy_run
    pair = _pair_with_known_pair_above_gold()
    candidates = {f"{EX}s{i}": [f"{EX}t{j}" for j in range(8)] + [f"{EX}t2dup"] for i in range(8)}
    monkeypatch.setattr(ablation, "load_dataset", lambda _d, _dir: [pair])
    monkeypatch.setattr(ablation, "load_candidates", lambda _d, _p, _dir: dict(candidates))
    run(["PA-full"])
    sources = _read(results_dir, "PA-full", "entities")["test_sources"]
    assert {s["source"]: s["best_rank"] for s in sources}[f"{EX}s3"] == 1
    rr = [1.0 / s["best_rank"] if s["best_rank"] else 0.0 for s in sources]
    assert np.mean(rr) == pytest.approx(_read(results_dir, "PA-full")["metrics"]["mrr"])


def test_results_scored_with_other_known_pairs_are_refused(toy_run, monkeypatch):
    run, _model, results_dir = toy_run
    run(["PA-full"])
    pair = replace(_pair(), train_refs=[(f"{EX}s0", f"{EX}t1")])
    monkeypatch.setattr(ablation, "load_dataset", lambda _d, _dir: [pair])
    with pytest.raises(DataError, match="known pairs"):
        run(["PA-ppas-80"])
    assert not (results_dir / "PA-ppas-80").exists()
    run(["PA-full", "PA-ppas-80"], force=True)
    assert _read(results_dir, "PA-full")["population"]["n_train_refs"] == 1


def test_signature_separates_runs_with_different_known_pairs():
    run = {"model_id": "m", "hf_revision": "r", "candidates_sha256": "c", "evaluation_protocol": 2,
           "token_budget": 48, "ablation": {"tokenizer_revision": "t", "smoke_max_sources": None},
           "_entities": {"test_sources": [{"source": "a"}]}, "population": {"n_train_refs": 0}}
    fixed = {**run, "population": {"n_train_refs": 726}}
    assert report.run_signature(run) != report.run_signature(fixed)


def test_sidecar_fingerprints_only_sampled_entities(toy_run):
    run, _model, results_dir = toy_run
    run()
    assert _read(results_dir, "PA-full", "entities")["sampled_fingerprints"] == {"source": {}, "candidate": {}}
    assert len(_read(results_dir, "PA-ppas-80", "entities")["sampled_fingerprints"]["source"]) == 8


def test_existing_results_are_not_overwritten_without_force(toy_run):
    run, _model, results_dir = toy_run
    path = results_dir / "PA-full" / "D5" / "toy_results.json"
    path.parent.mkdir(parents=True)
    path.write_text("sentinel")
    run(["PA-full"])
    assert path.read_text() == "sentinel"


def test_historical_results_directory_is_refused(toy_run):
    with pytest.raises(ValueError):
        ablation.run_ablation(["D5"], ["PA-full"], {
            "data_dir": Path("data"), "results_dir": Path("data/results"),
            "force_recompute": False, "max_sources": None,
        })


def test_smoke_restriction_keeps_a_stable_subset_with_its_references():
    pair = _pair()
    smaller = ablation.restrict_sources(pair, 3)
    assert len(smaller.source_entities) == 3
    assert smaller.source_entities == ablation.restrict_sources(pair, 3).source_entities
    kept = set(smaller.source_entities)
    assert all(s in kept for s, _t in smaller.val_refs + smaller.test_refs)
    assert ablation.restrict_sources(pair, None) is pair


# ---------------------------------------------------------------------------
# Report script
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.generate_ppas_ablation_report as report  # noqa: E402


def test_t_interval_matches_the_t_distribution():
    mean, sd, low, high = report.t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    half = 2.7764451 * np.sqrt(2.5) / np.sqrt(5)
    assert (mean, sd) == pytest.approx((3.0, np.sqrt(2.5)))
    assert (low, high) == pytest.approx((3.0 - half, 3.0 + half), abs=1e-6)


def test_paired_bootstrap_of_a_constant_difference_is_degenerate():
    rng = np.random.default_rng(0)
    diff, low, high = report.paired_bootstrap(np.full(50, 0.7), np.full(50, 0.5), rng)
    assert (diff, low, high) == pytest.approx((0.2, 0.2, 0.2))


def test_report_renders_every_section_for_a_toy_run(toy_run):
    run, _model, results_dir = toy_run
    run()
    text = report.build_report(results_dir)
    for heading in ("### Budget 80 tokens", "Coverage and cost", "Effectiveness",
                    "PPAS vs deterministic", "Paired differences", "MRR by source degree",
                    "Random 95% t-interval", "PPAS − Deterministic"):
        assert heading in text


def test_report_refuses_an_empty_results_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.build_report(tmp_path)


# ---------------------------------------------------------------------------
# Controlled policies P1-P7 on a synthetic FMA-style class pair (#128)
# ---------------------------------------------------------------------------

FMA = "http://purl.org/sig/ont/fma/"
_CONTROLLED_IDS = [c.condition_id for c in ABLATION_CONDITIONS if ablation.is_reference_run(c, "D1")]
_LABEL_OF = {cid: get_ablation_condition(cid).policy_label for cid in _CONTROLLED_IDS}


def _fma_class_graph(prefix: str, n: int) -> Graph:
    """Classes whose FMA annotations sort before rdfs:label and exceed 48 tokens."""
    graph = Graph()
    for i in range(n):
        entity = URIRef(f"{EX}{prefix}{i}")
        graph.add((entity, RDF.type, OWL.Class))
        graph.add((entity, RDFS.label, Literal(f"organ {i}")))
        graph.add((entity, URIRef(f"{FMA}preferred_name"), Literal(f"organ {i} proper")))
        graph.add((entity, URIRef(f"{FMA}FMAID"), Literal(str(7000 + i))))
        for j in range(6):
            graph.add((entity, URIRef(f"{FMA}synonym"), Literal(f"organ {i} alias {j} term")))
            graph.add((entity, URIRef(f"{FMA}author"), Literal(f"curator {j} of the record {i}")))
    return graph


def _class_pair() -> AlignmentPair:
    sources = [f"{EX}s{i}" for i in range(8)]
    refs = [(f"{EX}s{i}", f"{EX}t{i}") for i in range(8)]
    return AlignmentPair(
        dataset_id="D1", pair_name="toy", source_graph=_fma_class_graph("s", 8),
        target_graph=_fma_class_graph("t", 8), source_entities=sources,
        val_refs=refs[:3], test_refs=refs[3:], entity_type="class",
    )


@pytest.fixture
def controlled_run(monkeypatch, tmp_path):
    """Run every controlled condition on the toy class pair."""
    model = _FakeModel()
    info = {"model_key": "M2", "model_id": "fake/M2", "device": "mps", "hf_revision": "rev-1"}
    candidates = {f"{EX}s{i}": [f"{EX}t{j}" for j in range(8)] for i in range(8)}
    monkeypatch.setattr(ablation, "load_sentence_transformer", lambda _key: (model, info))
    monkeypatch.setattr(ablation, "load_dataset", lambda _d, _dir: [_class_pair()])
    monkeypatch.setattr(ablation, "load_candidates", lambda _d, _p, _dir: dict(candidates))
    results_dir = tmp_path / "controlled"
    kwargs = {"data_dir": tmp_path, "results_dir": results_dir, "force_recompute": False, "max_sources": None}
    payloads = ablation.run_ablation(["D1"], _CONTROLLED_IDS, kwargs)
    return {_LABEL_OF[p["condition_id"]]: p for p in payloads}, results_dir


def test_controlled_run_writes_every_policy(controlled_run):
    payloads, _ = controlled_run
    assert sorted(payloads) == [f"P{i}" for i in range(1, 8)]


def test_controlled_results_record_the_policy_and_its_vocabulary(controlled_run):
    payloads, _ = controlled_run
    for label, payload in payloads.items():
        block = payload["ablation"]
        filter_mode, vocabulary, order_mode = next(
            (f, v, o) for lab, f, v, o in CONTROLLED_POLICIES if lab == label
        )
        assert (block["policy_label"], block["filter_mode"], block["order_mode"]) == (label, filter_mode, order_mode)
        assert block["predicate_config"]["name"] == vocabulary
        assert block["tier_config_sha256"] == block["predicate_config"]["tiers_sha256"]
        assert block["tokenizer_revision"] == payload["hf_revision"] == "rev-1"
        for key in ("candidates_sha256", "evaluation_protocol", "git_commit", "git_dirty"):
            assert key in payload


def test_extended_vocabulary_hash_differs_from_base(controlled_run):
    payloads, _ = controlled_run
    hashes = {label: p["ablation"]["predicate_config"]["config_hash"] for label, p in payloads.items()}
    assert hashes["P1"] == hashes["P2"] == hashes["P3"] == hashes["P4"]
    assert hashes["P5"] == hashes["P6"] == hashes["P7"] != hashes["P2"]


def test_p5_selects_differently_from_p2(controlled_run):
    payloads, _ = controlled_run
    assert payloads["P5"]["ablation"]["selection_sha256"] != payloads["P2"]["ablation"]["selection_sha256"]


def test_policies_differ_except_where_filtering_cannot_bind(controlled_run):
    """P3 = P4: the covered base text fits.  P5 = P7: tier 0 fills the budget
    before the filtered fma:author items would be visited."""
    payloads, _ = controlled_run
    digest = {label: p["ablation"]["selection_sha256"] for label, p in payloads.items()}
    assert digest["P3"] == digest["P4"]
    assert payloads["P3"]["ablation"]["source"]["n_sampling_invoked"] == 0
    assert digest["P5"] == digest["P7"]
    others = [digest[label] for label in ("P1", "P2", "P3", "P5", "P6")]
    assert len(set(others)) == len(others)


def test_controlled_policies_share_population_candidates_and_budget(controlled_run):
    payloads, _ = controlled_run
    for key in ("candidates_sha256", "token_budget", "hf_revision", "evaluation_protocol"):
        assert len({p[key] for p in payloads.values()}) == 1, key
    assert len({json.dumps(p["population"], sort_keys=True) for p in payloads.values()}) == 1


def test_controlled_policies_never_exceed_the_budget(controlled_run):
    payloads, _ = controlled_run
    for payload in payloads.values():
        for side in ("source", "candidate"):
            summary = payload["ablation"][side]
            assert summary["tokens"]["max"] <= 48
            assert summary["budget"]["n_over_budget"] == 0


def test_filtering_accounting_matches_between_paired_policies(controlled_run):
    payloads, _ = controlled_run
    side = {label: p["ablation"]["source"] for label, p in payloads.items()}
    for a, b in (("P3", "P4"), ("P6", "P7"), ("P1", "P2")):
        for key in ("n_items_before_filter", "n_filtered_items", "eligible_items"):
            assert side[a][key] == side[b][key], (a, b, key)
    assert side["P1"]["n_filtered_items"] == 0
    assert side["P3"]["n_filtered_items"] > side["P6"]["n_filtered_items"] > 0
    assert side["P5"]["n_covered_items"] > side["P2"]["n_covered_items"]


def test_sidecar_fingerprints_every_entity(controlled_run):
    _, results_dir = controlled_run
    sidecar = json.loads((results_dir / "PA-cov-base-det-48" / "D1" / "toy_entities.json").read_text())
    assert len(sidecar["selection_fingerprints"]["source"]) == 8
    assert len(sidecar["selection_fingerprints"]["candidate"]) == 8


def test_ppas_effective_requires_ppas_order_and_invoked_sampling(controlled_run):
    payloads, _ = controlled_run
    effective = {label: p["ppas_effective"] for label, p in payloads.items()}
    assert effective == {"P1": False, "P2": True, "P3": False, "P4": False,
                         "P5": True, "P6": False, "P7": True}


def test_controlled_conditions_refuse_the_issue_122_directory():
    with pytest.raises(ValueError):
        ablation._guard_results_dir(Path(ablation.DEFAULT_RESULTS_DIR), [get_ablation_condition("PA-all-base-det-48")], ["D1"])
    ablation._guard_results_dir(Path(ablation.DEFAULT_RESULTS_DIR), [get_ablation_condition("PA-det-48")], ["D1"])


# ---------------------------------------------------------------------------
# Controlled-policy report (#128)
# ---------------------------------------------------------------------------


def _controlled_runs(results_dir: Path) -> list:
    return [r for runs in report.load_runs(results_dir).values() for r in runs if report.is_controlled(r)]


def test_report_renders_the_controlled_section_keyed_by_label(controlled_run):
    _, results_dir = controlled_run
    text = report.build_report(results_dir)
    for heading in ("### Controlled policies (#128)", "##### Execution matrix", "##### Matching quality",
                    "##### Representation cost", "##### Selection changes", "unit = test source",
                    "Missing policies: none", "P3 − P1", "P4 − P3", "P2 − P1", "P5 − P2", "P6 − P3",
                    "P7 − P6", "P7 − P5"):
        assert heading in text, heading
    assert "### Policy ablation (#122)" not in text


def test_controlled_runs_with_the_same_policy_do_not_collide(controlled_run):
    _, results_dir = controlled_run
    [(_signature, group)] = report.signature_groups(_controlled_runs(results_dir))
    assert sorted(group) == [f"P{i}" for i in range(1, 8)]
    assert group["P1"]["ablation"]["policy"] == group["P3"]["ablation"]["policy"] == DETERMINISTIC


def test_runs_with_different_signatures_are_never_combined(controlled_run):
    _, results_dir = controlled_run
    runs = _controlled_runs(results_dir)
    smoke = json.loads(json.dumps([r for r in runs if report._label(r) == "P1"][0]))
    smoke["ablation"]["smoke_max_sources"] = 3
    groups = report.signature_groups(runs + [smoke])
    assert [sorted(g) for _s, g in groups] == [[f"P{i}" for i in range(1, 8)], ["P1"]]
    text = "\n\n".join(report.controlled_parts(runs + [smoke]))
    assert "Run group 2" in text and "SMOKE, 3 sources" in text
    assert "Missing policies: P2, P3, P4, P5, P6, P7" in text


def test_duplicate_labels_within_one_signature_raise(controlled_run):
    _, results_dir = controlled_run
    runs = _controlled_runs(results_dir)
    with pytest.raises(ValueError):
        report.signature_groups(runs + [runs[0]])


def test_selection_change_counts_match_the_fingerprints(controlled_run):
    _, results_dir = controlled_run
    [(_signature, group)] = report.signature_groups(_controlled_runs(results_dir))
    assert report.changed_entities(group["P3"], group["P4"], "source") == set()
    assert len(report.changed_entities(group["P5"], group["P2"], "source")) == 8


def test_controlled_bootstrap_pairs_per_source_differences(controlled_run):
    _, results_dir = controlled_run
    [(_signature, group)] = report.signature_groups(_controlled_runs(results_dir))
    a, b = report._per_source(group["P5"]), report._per_source(group["P2"])
    sources = sorted(a)
    expected = np.mean([a[s][0] - b[s][0] for s in sources])
    row = report._contrast_row("x", a, b, set(sources), np.random.default_rng(0))
    assert row[2].startswith(report._fmt(expected))


def test_changed_and_unchanged_strata_partition_the_test_sources(controlled_run):
    _, results_dir = controlled_run
    [(_signature, group)] = report.signature_groups(_controlled_runs(results_dir))
    table = report.controlled_contrast_section(group)
    for a, b, _effect in report._available_contrasts(group):
        counts = {
            cells[2]: int(cells[3])
            for cells in (line.split(" | ") for line in table.splitlines())
            if len(cells) > 3 and cells[0].lstrip("| ") == f"{a} − {b}"
        }
        assert counts[report._CHANGED] + counts[report._UNCHANGED] == counts[report._ALL]


def test_inventory_compares_selected_text_not_filtered_indices():
    import scripts.ppas_ablation_inventory as inventory

    base = Selection("a: 1", (0,), 1, 4, False, 0)
    assert inventory._differ([base], [replace(base, selected=(3,))]) == 0
    assert inventory._differ([base], [replace(base, text="b: 2")]) == 1


@pytest.mark.parametrize("results_dir,condition_id,datasets", [
    (ablation.CONTROLLED_RESULTS_DIR, "PA-all-base-det-32", ["D1"]),
    (ablation.CONTROLLED_RESULTS_DIR, "PA-all-base-det-48", ["D2"]),
    (ablation.CONTROLLED_RESULTS_DIR, "PA-cov-obox-ppas-48", ["D2"]),
    (ablation.VALIDATION_RESULTS_DIR, "PA-all-base-det-48", ["D1"]),
    (ablation.VALIDATION_RESULTS_DIR, "PA-cov-fmax-ppas-48", ["D1", "D2"]),
])
def test_validation_and_reference_runs_refuse_each_others_directory(results_dir, condition_id, datasets):
    with pytest.raises(ValueError):
        ablation._guard_results_dir(Path(results_dir), [get_ablation_condition(condition_id)], datasets)


@pytest.mark.parametrize("results_dir,condition_ids,datasets", [
    (ablation.CONTROLLED_RESULTS_DIR, ["PA-all-base-det-48", "PA-cov-fmax-ppas-48"], ["D1"]),
    (ablation.VALIDATION_RESULTS_DIR, ["PA-all-base-det-32", "PA-cov-base-ppas-64"], ["D1"]),
    (ablation.VALIDATION_RESULTS_DIR, ["PA-all-base-det-48", "PA-cov-obox-ppas-48"], ["D2"]),
])
def test_validation_and_reference_runs_accept_their_own_directory(results_dir, condition_ids, datasets):
    ablation._guard_results_dir(Path(results_dir), [get_ablation_condition(c) for c in condition_ids], datasets)


def test_dataset_file_digests_fingerprint_the_registered_files(tmp_path):
    (tmp_path / "d2_anatomy").mkdir()
    (tmp_path / "d2_anatomy" / "source.rdf").write_bytes(b"abc")
    digests = ablation.dataset_file_digests("D2", tmp_path)
    assert digests["d2_anatomy/source.rdf"] == hashlib.sha256(b"abc").hexdigest()
    assert digests["d2_anatomy/target.rdf"] is None
    assert ablation.dataset_file_digests("D5", tmp_path) == {}


def test_results_record_dataset_file_digests(controlled_run):
    payloads, _ = controlled_run
    assert all("dataset_files_sha256" in p for p in payloads.values())


@pytest.mark.parametrize("dataset_id,budget,labels", [
    ("D1", 48, [f"P{i}" for i in range(1, 8)]),
    ("D1", 32, ["P1", "P2", "P3", "P4"]),
    ("D2", 48, [f"P{i}" for i in range(1, 8)]),
    ("D2", 64, []),
])
def test_report_expects_the_registered_labels_per_dataset_and_budget(dataset_id, budget, labels):
    assert report.expected_labels(dataset_id, budget) == labels


def test_contrast_effects_name_the_groups_extended_vocabulary(controlled_run):
    _, results_dir = controlled_run
    [(_signature, group)] = report.signature_groups(_controlled_runs(results_dir))
    effects = {(a, b): effect for a, b, effect in report._available_contrasts(group)}
    assert effects[("P7", "P6")] == "ordering, eligibility fixed (fma_extended)"
    assert ("P4", "P2") in effects


# ---------------------------------------------------------------------------
# Validation report (#128 follow-up)
# ---------------------------------------------------------------------------

import scripts.generate_ppas_validation_report as validation  # noqa: E402

_VALIDATION_IDS = {
    dataset_id: [c.condition_id for c in ABLATION_CONDITIONS if c.is_controlled
                 and dataset_id in c.datasets and not ablation.is_reference_run(c, dataset_id)]
    for dataset_id in ("D1", "D2")
}


def _run_into(monkeypatch, tmp_path, dataset_id, condition_ids, results_dir):
    model = _FakeModel()
    info = {"model_key": "M2", "model_id": "fake/M2", "device": "mps", "hf_revision": "rev-1"}
    candidates = {f"{EX}s{i}": [f"{EX}t{j}" for j in range(8)] for i in range(8)}
    pair = replace(_class_pair(), dataset_id=dataset_id)
    monkeypatch.setattr(ablation, "load_sentence_transformer", lambda _key: (model, info))
    monkeypatch.setattr(ablation, "load_dataset", lambda _d, _dir: [pair])
    monkeypatch.setattr(ablation, "load_candidates", lambda _d, _p, _dir: dict(candidates))
    kwargs = {"data_dir": tmp_path, "results_dir": results_dir, "force_recompute": False, "max_sources": None}
    return ablation.run_ablation([dataset_id], condition_ids, kwargs)


@pytest.fixture
def validation_dirs(monkeypatch, tmp_path):
    """Write a toy #128 reference and toy D1 (32, 64) and D2 (48) validation runs."""
    reference, results = tmp_path / "controlled", tmp_path / "validation"
    _run_into(monkeypatch, tmp_path, "D1", _CONTROLLED_IDS, reference)
    _run_into(monkeypatch, tmp_path, "D1", _VALIDATION_IDS["D1"], results)
    _run_into(monkeypatch, tmp_path, "D2", _VALIDATION_IDS["D2"], results)
    return reference, results


def test_validation_runs_cover_the_expected_matrix(validation_dirs):
    runs = validation.collect_runs(*validation_dirs)
    present = sorted((r["dataset_id"], r["token_budget"], report._label(r)) for r in runs)
    assert present == sorted(validation.expected_runs())
    assert len(present) == 3 * 4 + 7


def test_validation_report_renders_both_analyses_separately(validation_dirs, tmp_path):
    reference, results = validation_dirs
    text = validation.build_report(reference, results, results / "figures", tmp_path)
    assert "Expected full-population runs: 19; complete: 19." in text
    assert "## D1 budget sensitivity" in text and "## D2 replication at 48 tokens" in text
    for name in ("d1_metrics_by_budget.png", "d1_contrasts_by_budget.png", "d2_contrasts.png"):
        assert (results / "figures" / name).stat().st_size > 0


def test_validation_groups_never_mix_budgets_or_datasets(validation_dirs):
    runs = validation.collect_runs(*validation_dirs)
    d1 = validation.full_groups(validation.groups_of(runs, "D1"))
    assert sorted(d1) == [32, 48, 64]
    for budget, group in d1.items():
        assert sorted(group) == list(validation.BASE_LABELS)
        assert {r["token_budget"] for r in group.values()} == {budget}
        assert {r["dataset_id"] for r in group.values()} == {"D1"}
    d2 = validation.full_groups(validation.groups_of(runs, "D2"))
    assert list(d2) == [48] and len(d2[48]) == 7


def test_smoke_groups_are_excluded_from_the_analysis(validation_dirs):
    runs = validation.collect_runs(*validation_dirs)
    smoke = json.loads(json.dumps([r for r in runs if r["dataset_id"] == "D2"]))
    for run in smoke:
        run["ablation"]["smoke_max_sources"] = 3
    groups = validation.groups_of(runs + smoke, "D2")
    assert len(groups) == 2
    assert validation.full_groups(groups)[48]["P1"]["ablation"]["smoke_max_sources"] is None


def test_validation_refuses_a_d1_reference_budget_run_in_its_own_root(validation_dirs):
    reference, results = validation_dirs
    source = reference / "PA-all-base-det-48"
    target = results / "PA-all-base-det-48" / "D1"
    target.mkdir(parents=True)
    for path in (source / "D1").iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="#128 reference"):
        validation.collect_runs(reference, results)


def test_validation_contrasts_reproduce_the_128_report_intervals(validation_dirs):
    runs = validation.collect_runs(*validation_dirs)
    group = validation.full_groups(validation.groups_of(runs, "D1"))[48]
    estimates = validation.contrast_estimates(group["P3"], group["P1"])
    rows = [line for line in report.controlled_contrast_section(group).splitlines() if line.startswith("| P3 − P1 |")]
    for line, stratum in zip(rows, validation.STRATA):
        assert validation._interval(estimates[stratum]["mrr"]) in line


def test_identical_selections_are_reported_as_not_identifiable(validation_dirs):
    runs = validation.collect_runs(*validation_dirs)
    group = validation.full_groups(validation.groups_of(runs, "D2"))[48]
    assert not validation.identifiable(group["P5"], group["P2"])
    table = validation.contrast_table({48: group}, (("P5", "P2", "vocabulary"),))
    assert validation._NOT_IDENTIFIABLE in table


# ---------------------------------------------------------------------------
# Predicate-role ablation (#131)
# ---------------------------------------------------------------------------

import scripts.generate_predicate_role_report as role_report  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import ROLE_MATRIX  # noqa: E402
from kgsemembed.verbalisation.predicate_roles import UnmappedPredicateError  # noqa: E402

_ROLE_IDS = {d: [c.condition_id for c in ABLATION_CONDITIONS if c.is_role_ablation and d in c.datasets]
             for d in ("D1", "D2")}


def _role_graph(prefix: str, n: int) -> Graph:
    """Classes with only roles_v1-mapped predicates; definitions on even entities only."""
    graph = Graph()
    for i in range(n):
        entity = URIRef(f"{EX}{prefix}{i}")
        graph.add((entity, RDF.type, OWL.Class))
        graph.add((entity, RDFS.label, Literal(f"organ {i}")))
        graph.add((entity, URIRef(f"{FMA}preferred_name"), Literal(f"organ {i}")))
        graph.add((entity, URIRef(f"{FMA}FMAID"), Literal(str(7000 + i))))
        for j in range(6):
            graph.add((entity, URIRef(f"{FMA}synonym"), Literal(f"organ {i} alias {j} term")))
        if i % 2 == 0:
            graph.add((entity, URIRef(f"{FMA}definition"), Literal(f"organ {i} is a structure of the body")))
    return graph


def _role_pair(dataset_id: str) -> AlignmentPair:
    return replace(_class_pair(), dataset_id=dataset_id, source_graph=_role_graph("s", 8),
                   target_graph=_role_graph("t", 8))


def _run_pair_into(monkeypatch, tmp_path, pair, condition_ids, results_dir):
    model = _FakeModel()
    info = {"model_key": "M2", "model_id": "fake/M2", "device": "mps", "hf_revision": "rev-1"}
    candidates = {f"{EX}s{i}": [f"{EX}t{j}" for j in range(8)] for i in range(8)}
    monkeypatch.setattr(ablation, "load_sentence_transformer", lambda _key: (model, info))
    monkeypatch.setattr(ablation, "load_dataset", lambda _d, _dir: [pair])
    monkeypatch.setattr(ablation, "load_candidates", lambda _d, _p, _dir: dict(candidates))
    kwargs = {"data_dir": tmp_path, "results_dir": results_dir, "force_recompute": False, "max_sources": None}
    return ablation.run_ablation([pair.dataset_id], condition_ids, kwargs)


@pytest.fixture
def role_dirs(monkeypatch, tmp_path):
    """Run every role condition on toy D1 and D2 pairs, plus a toy lexicographic P1 link at 48."""
    roles, links = tmp_path / "roles", tmp_path / "links"
    payloads = []
    for dataset_id in ("D1", "D2"):
        payloads += _run_pair_into(monkeypatch, tmp_path, _role_pair(dataset_id), _ROLE_IDS[dataset_id], roles)
    _run_pair_into(monkeypatch, tmp_path, _role_pair("D1"), ["PA-all-base-det-48"], links)
    monkeypatch.setattr(role_report, "_LINK_DIRS", {"D1": str(links), "D2": str(tmp_path / "absent")})
    return payloads, roles


def test_role_runs_refuse_unmapped_predicates_before_writing(monkeypatch, tmp_path):
    with pytest.raises(UnmappedPredicateError):
        _run_pair_into(monkeypatch, tmp_path, _class_pair(), ["PR-all-full"], tmp_path / "roles")
    assert not (tmp_path / "roles").exists()


@pytest.mark.parametrize("results_dir", [
    "data/results/", ablation.DEFAULT_RESULTS_DIR, ablation.CONTROLLED_RESULTS_DIR, ablation.VALIDATION_RESULTS_DIR,
])
def test_role_conditions_refuse_the_historical_directories(results_dir):
    with pytest.raises(ValueError):
        ablation._guard_results_dir(Path(results_dir), [get_ablation_condition("PR-no-syn-full")], ["D1"])


def test_the_role_directory_refuses_other_conditions():
    with pytest.raises(ValueError):
        ablation._guard_results_dir(Path(ablation.ROLE_RESULTS_DIR), [get_ablation_condition("PA-all-base-det-32")], ["D1"])
    ablation._guard_results_dir(Path(ablation.ROLE_RESULTS_DIR), [get_ablation_condition("PR-core-48")], ["D1"])


def test_role_results_record_arm_roles_removal_and_truncation(role_dirs):
    payloads, _ = role_dirs
    assert len(payloads) == 38
    by_id = {(p["dataset_id"], p["condition_id"]): p for p in payloads}
    block = by_id[("D1", "PR-no-syn-full")]["ablation"]
    assert block["role_arm"] == "R-syn" and block["excluded_roles"] == ["synonym"]
    assert block["role_map"]["name"] == "roles_v1" and len(block["role_map"]["config_hash"]) == 64
    assert block["candidate"]["role_removal"]["synonym"] == {"n_items": 48, "n_entities": 8, "item_tokens": 288}
    assert block["candidate"]["n_truncated_tokens"] == 0 and block["candidate"]["n_sampling_invoked"] == 0
    assert by_id[("D1", "PR-all-full")]["ablation"]["candidate"]["role_removal"] == {}


def test_budgeted_role_arms_never_exceed_the_budget(role_dirs):
    payloads, _ = role_dirs
    for payload in (p for p in payloads if p["token_budget"] is not None):
        for side in ("source", "candidate"):
            assert payload["ablation"][side]["tokens"]["max"] <= payload["token_budget"]
            assert payload["ablation"][side]["budget"]["n_over_budget"] == 0


def test_role_report_covers_the_matrix_and_every_section(role_dirs, tmp_path):
    _, roles = role_dirs
    text = role_report.build_report(roles, roles / "figures", tmp_path)
    assert "Expected full-population runs: 38; complete: 38." in text
    for heading in ("## D1", "## D2", "difference-in-differences", "Budget cost", "C+x − A-core",
                    "A0 core-first − P1 lexicographic"):
        assert heading in text
    assert (roles / "figures" / "d1_removal_effects.png").stat().st_size > 0


def test_absent_roles_are_reported_as_not_identifiable(role_dirs):
    _, roles = role_dirs
    groups = role_report.full_groups(role_report.load_role_runs(roles), "D1")
    assert not validation.identifiable(groups[None]["R-nonEn"], groups[None]["A0"])
    assert validation.identifiable(groups[None]["R-syn"], groups[None]["A0"])


def test_difference_in_differences_pairs_four_runs_per_source(role_dirs):
    _, roles = role_dirs
    groups = role_report.full_groups(role_report.load_role_runs(roles), "D1")
    runs = (groups[48]["R-syn"], groups[48]["A0"], groups[None]["R-syn"], groups[None]["A0"])
    per = [report._per_source(r) for r in runs]
    expected = np.mean([(per[0][s][0] - per[1][s][0]) - (per[2][s][0] - per[3][s][0]) for s in per[0]])
    n, mrr, _r1 = role_report.did_estimate(*runs)
    assert n == len(per[0]) and mrr[0] == pytest.approx(expected)


def test_role_groups_never_mix_budgets_or_smoke_runs(role_dirs):
    _, roles = role_dirs
    runs = role_report.load_role_runs(roles)
    smoke = json.loads(json.dumps([r for r in runs if r["dataset_id"] == "D2"]))
    for run in smoke:
        run["ablation"]["smoke_max_sources"] = 3
    groups = role_report.full_groups(runs + smoke, "D2")
    assert sorted(groups, key=lambda b: b or 0) == [None, 24, 48]
    assert all(r["ablation"]["smoke_max_sources"] is None for g in groups.values() for r in g.values())
    assert set(groups[24]) == set(ROLE_MATRIX["D2"][24])
