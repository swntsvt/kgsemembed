"""Tests for the controlled PPAS ablation registry and runner (#122)."""

import hashlib
import sys
import json
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

import kgsemembed.pipeline.ppas_ablation as ablation
from kgsemembed.datasets import AlignmentPair
from kgsemembed.embeddings.models import get_model_config
from kgsemembed.pipeline.ablation_conditions import (
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    PRIMARY_BUDGET,
    RANDOM_SEEDS,
    STRESS_BUDGETS,
    get_ablation_condition,
    stress_budget,
)
from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS
from kgsemembed.verbalisation.selection import DETERMINISTIC, FULL, PPAS, RANDOM

EX = "http://example.org/"
_DIM = 32
_N_ATTRIBUTES = 30
_CONDITIONS = ["PA-full", "PA-ppas-80", "PA-det-80", "PA-rand-80-s0", "PA-rand-80-s1"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_ids_are_unique_and_share_one_model():
    ids = [c.condition_id for c in ABLATION_CONDITIONS]
    # 22 historical + 12 controlled policy conditions (P3, P4, P5, P5-det at 3 budgets each)
    assert len(ids) == len(set(ids)) == 34
    assert {c.model_key for c in ABLATION_CONDITIONS} == {ABLATION_MODEL_KEY} == {"M2"}


def test_each_budget_has_ppas_deterministic_and_every_random_seed():
    # Only check historical conditions (filter_mode == "all" and predicate_config == "base")
    historical = [c for c in ABLATION_CONDITIONS if c.filter_mode == "all" and c.predicate_config == "base"]
    budgets = {c.token_budget for c in historical} - {None}
    assert budgets == {PRIMARY_BUDGET, *STRESS_BUDGETS.values()}
    for budget in budgets:
        family = [c for c in historical if c.token_budget == budget]
        assert sorted(c.policy for c in family if c.policy != RANDOM) == [DETERMINISTIC, PPAS]
        assert sorted(c.seed for c in family if c.policy == RANDOM) == list(RANDOM_SEEDS)
        assert len({c.datasets for c in family}) == 1


def test_random_seeds_are_at_least_five():
    assert len(RANDOM_SEEDS) >= 5


def test_primary_budget_runs_on_d5_only_and_d1_uses_its_stress_budget():
    primary = [c for c in ABLATION_CONDITIONS if c.token_budget == PRIMARY_BUDGET]
    assert {c.datasets for c in primary} == {("D5",)}
    d1_budgets = {c.token_budget for c in ABLATION_CONDITIONS if "D1" in c.datasets}
    assert d1_budgets == {None, STRESS_BUDGETS["D1"]}


def test_budgets_fit_the_model_input_limit():
    limit = get_model_config(ABLATION_MODEL_KEY).max_tokens
    assert all(c.token_budget <= limit for c in ABLATION_CONDITIONS if c.token_budget)


def test_only_full_is_uncapped_and_only_random_is_seeded():
    for condition in ABLATION_CONDITIONS:
        assert (condition.policy == FULL) == (condition.token_budget is None)
        assert (condition.policy == RANDOM) == (condition.seed is not None)


def test_historical_registry_is_unchanged():
    assert len(EXPERIMENT_CONDITIONS) == 20
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
    info = {"model_key": "M2", "model_id": "fake/M2", "device": "cpu", "hf_revision": "rev-1"}
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
