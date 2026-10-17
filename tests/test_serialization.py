"""Tests for the text-serialization-order experiment (#134)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import generate_serialization_report as report  # noqa: E402
import serialization_inventory as inventory  # noqa: E402

from kgsemembed.datasets import AlignmentPair  # noqa: E402
from kgsemembed.evaluation.order_selection import PRECEDENCE, choose_order, paired_interval  # noqa: E402
from kgsemembed.pipeline import ppas_ablation as base  # noqa: E402
from kgsemembed.pipeline import serialization_order as runner  # noqa: E402
from kgsemembed.pipeline.ablation_conditions import ABLATION_CONDITIONS, ROLE_MAP, AblationCondition, \
    _validate_registry  # noqa: E402
from kgsemembed.pipeline.serialization_conditions import (  # noqa: E402
    B_BUDGETS,
    EXECUTION_LABELS,
    FIXED_SET_GROUPS,
    LOGICAL_CONDITIONS,
    SERIALIZATION_CONDITIONS,
    executions,
    get_logical,
    reproduction_target,
)
from kgsemembed.verbalisation.predicate_config import expand_predicate  # noqa: E402
from kgsemembed.verbalisation.predicate_roles import RoleFilter, get_role_map  # noqa: E402
from kgsemembed.verbalisation.selection import (  # noqa: E402
    CORE_FIRST,
    FULL,
    LABEL_FIRST,
    ROLE_PRIORITY,
    EligibleItem,
    PolicySelector,
    join_items,
)
from kgsemembed.verbalisation.serialization import (  # noqa: E402
    SERIALIZATION_ORDERS,
    order_definitions_sha256,
    serialization_order,
)

ROLES = get_role_map(ROLE_MAP)
FMA = "http://purl.org/sig/ont/fma/"
ITEMS = sorted([
    EligibleItem(expand_predicate("fma:FMAID"), "FMAID: 12345"),
    EligibleItem(expand_predicate("fma:non-English_equivalent"), "non-English equivalent: Nomen"),
    EligibleItem(expand_predicate("fma:preferred_name"), "preferred name: Gold label"),
    EligibleItem(expand_predicate("fma:synonym"), "synonym: Gold synonym"),
    EligibleItem(expand_predicate("fma:definition"), "definition: Gold definition"),
    EligibleItem(expand_predicate("rdf:type"), "type: Class"),
    EligibleItem(expand_predicate("rdfs:label"), "label: Gold label"),
    EligibleItem(expand_predicate("rdfs:subClassOf"), "sub Class Of: Parent"),
    EligibleItem(expand_predicate("owl:disjointWith"), "disjoint With: Other"),
], key=lambda i: (i.predicate, i.text))


def _words(text: str) -> int:
    """Order-invariant token counter: words plus two special tokens."""
    return len(text.split()) + 2


def _keys(text: str) -> list:
    return [part.split(":")[0] for part in text.split(" | ")]


# ---------------------------------------------------------------------------
# Matrix accounting
# ---------------------------------------------------------------------------


def test_matrix_has_62_logical_conditions_and_57_executions():
    assert len(LOGICAL_CONDITIONS) == 62
    assert len(executions()) == 57
    assert len({c.name for c in LOGICAL_CONDITIONS}) == 62


def test_block_s_has_32_logical_conditions_and_30_executions():
    block = [c for c in LOGICAL_CONDITIONS if c.block == "S"]
    assert len(block) == 32
    assert len({(c.condition_id, c.dataset) for c in block}) == 30


def test_each_budget_configuration_has_10_logical_conditions_and_9_executions():
    for dataset, budget in B_BUDGETS:
        block = [c for c in LOGICAL_CONDITIONS if c.dataset == dataset and c.budget == budget]
        assert len(block) == 10
        assert len({c.condition_id for c in block}) == 9
    assert sum(c.block == "B" for c in LOGICAL_CONDITIONS) == 30


def test_exactly_five_logical_conditions_share_an_execution():
    shared = {c.name: c.condition_id for c in LOGICAL_CONDITIONS if c.shared_reason}
    assert shared == {
        "D1/S/A-core/core": "SO-core-uri-full", "D2/S/A-core/core": "SO-core-uri-full",
        "D1/B48/C-core": "SO-all-sel_core-ren_core-48", "D2/B24/C-core": "SO-all-sel_core-ren_core-24",
        "D2/B48/C-core": "SO-all-sel_core-ren_core-48",
    }


def test_budgeted_executions_are_b1_to_b9():
    for dataset, budget in B_BUDGETS:
        block = [c for c in LOGICAL_CONDITIONS if c.dataset == dataset and c.budget == budget]
        assert {EXECUTION_LABELS[(c.selection, c.render)] for c in block} == {f"B{i}" for i in range(1, 10)}


def test_registered_conditions_cover_every_execution_with_the_right_orders():
    by_id = {c.condition_id: c for c in SERIALIZATION_CONDITIONS}
    assert {(cid, d) for cid, d in executions()} == {(c.condition_id, d) for c in SERIALIZATION_CONDITIONS
                                                    for d in c.datasets}
    assert by_id["SO-all-sel_label-ren_uri-48"].order_mode == LABEL_FIRST
    assert by_id["SO-all-sel_role-ren_role-24"].order_mode == ROLE_PRIORITY
    assert by_id["SO-all-sel_uri-ren_uri-48"].order_mode == "deterministic"
    assert by_id["SO-all-sel_core-ren_label-48"].order_mode == CORE_FIRST
    assert by_id["SO-core-plus-syn-role-full"].order_mode == FULL
    assert by_id["SO-core-plus-id-uri-full"].datasets == ("D1",)
    assert by_id["SO-core-plus-def-uri-full"].datasets == ("D2",)
    assert by_id["SO-all-sel_core-ren_uri-24"].datasets == ("D2",)


def test_reproduction_targets_cover_every_canonical_execution_but_coupled_uri_at_24():
    targets = {c.name: reproduction_target(c) for c in LOGICAL_CONDITIONS if reproduction_target(c)}
    assert len(targets) == 13
    assert targets["D1/S/C+id/uri"] == ("data/results_predicate_roles", "PR-core-plus-id-full")
    assert targets["D2/B24/F-uri"] == ("data/results_predicate_roles", "PR-all-24")
    assert targets["D1/B48/C-uri"] == ("data/results_ppas_controlled", "PA-all-base-det-48")
    assert targets["D2/B48/C-uri"] == ("data/results_ppas_validation", "PA-all-base-det-48")
    assert reproduction_target(get_logical("D2/B24/C-uri")) is None


def test_fixed_set_groups_are_the_planned_ones():
    assert len(FIXED_SET_GROUPS) == 8 + 3 * 3
    assert ("D1/B48/C-label", "D1/B48/X-label") in FIXED_SET_GROUPS
    assert ("D2/B24/F-uri", "D2/B24/F-core", "D2/B24/F-label", "D2/B24/F-role", "D2/B24/C-core") in FIXED_SET_GROUPS


# ---------------------------------------------------------------------------
# Orders and the unchanged default
# ---------------------------------------------------------------------------


def test_orders_match_their_definitions():
    every = range(len(ITEMS))
    rendered = {o: _keys(" | ".join(ITEMS[i].text for i in serialization_order(ITEMS, every, o, ROLES)))
                for o in SERIALIZATION_ORDERS}
    assert rendered["uri"] == _keys(join_items([i.text for i in ITEMS], list(every)))
    assert rendered["uri"][:2] == ["FMAID", "definition"]
    assert rendered["core"][:3] == ["type", "label", "sub Class Of"]
    assert rendered["label"][0] == "label" and rendered["label"][1:] == [k for k in rendered["uri"] if k != "label"]
    assert rendered["role"] == ["label", "preferred name", "synonym", "definition", "sub Class Of", "type",
                                "disjoint With", "non-English equivalent", "FMAID"]


def test_every_roles_v1_predicate_has_a_place_in_every_order():
    for predicate in ROLES.roles:
        for order in SERIALIZATION_ORDERS:
            serialization_order([EligibleItem(predicate, "x: y")], [0], order, ROLES)


def test_order_definitions_hash_is_pinned():
    assert order_definitions_sha256() == "9429a87cfc824b14bc947d822ac2df93585550c8d5bf4564cbdc99d6c9a14769"


def test_default_render_is_canonical_for_every_existing_condition():
    assert all(c.render_order == "uri" for c in ABLATION_CONDITIONS)
    for condition in ABLATION_CONDITIONS:
        selector = base.condition_selector(condition, _words)
        selection = selector.select(ITEMS, "http://e", "class")
        kept = selector.kept_items(ITEMS)
        assert selection.text == join_items([i.text for i in kept], list(selection.selected))


def test_registry_refuses_a_non_canonical_render_order():
    rogue = AblationCondition("PA-rogue", FULL, None, None, ("D1",), render_order="label")
    with pytest.raises(ValueError, match="render canonically"):
        _validate_registry([*ABLATION_CONDITIONS, rogue])


def test_non_canonical_render_needs_a_role_map_and_a_known_order():
    with pytest.raises(ValueError):
        PolicySelector(order_mode=FULL, count_tokens=_words, render_order="label")
    with pytest.raises(ValueError):
        PolicySelector(order_mode=FULL, count_tokens=_words, role_filter=RoleFilter(ROLES), render_order="alpha")


def test_fixed_set_keeps_identical_items_across_render_orders():
    sets = set()
    for render in SERIALIZATION_ORDERS:
        selector = PolicySelector(order_mode=CORE_FIRST, count_tokens=_words, budget=14,
                                  role_filter=RoleFilter(ROLES), render_order=render)
        selection = selector.select(ITEMS, "http://e", "class")
        kept = selector.kept_items(ITEMS)
        sets.add(runner.item_set_fingerprint([kept[i] for i in selection.selected]))
        assert selection.sampling_invoked and selection.n_tokens <= 14
    assert len(sets) == 1


def test_item_set_fingerprint_ignores_order():
    assert runner.item_set_fingerprint(ITEMS) == runner.item_set_fingerprint(list(reversed(ITEMS)))
    assert runner.item_set_fingerprint(ITEMS) != runner.item_set_fingerprint(ITEMS[1:])


def test_rendered_text_is_asserted_within_budget():
    def order_sensitive(text: str) -> int:
        return _words(text) + (50 if text.startswith("label") else 0)

    selector = PolicySelector(order_mode=CORE_FIRST, count_tokens=order_sensitive, budget=14,
                              role_filter=RoleFilter(ROLES), render_order="label")
    with pytest.raises(AssertionError, match="budget"):
        selector.select(ITEMS, "http://e", "class")


def test_label_first_selection_keeps_the_label_before_anything_else():
    selector = PolicySelector(order_mode=LABEL_FIRST, count_tokens=_words, budget=6,
                              role_filter=RoleFilter(ROLES), render_order="label")
    assert selector.select(ITEMS, "http://e", "class").text == "label: Gold label"


@pytest.mark.slow
def test_token_counts_do_not_depend_on_item_order_under_the_m2_tokenizer():
    from predicate_role_diagnostics import token_counter
    count = token_counter()
    texts = [i.text for i in ITEMS] + ["synonym: Some (thing), other-thing", "label: Ünïcode name 3/4"]
    rng = np.random.default_rng(0)
    base_count = count(" | ".join(texts))
    for _ in range(50):
        assert count(" | ".join(rng.permutation(texts))) == base_count


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _rr(n: int, mean: float, seed: int) -> np.ndarray:
    return np.clip(np.random.default_rng(seed).normal(mean, 0.05, n), 0, 1)


def test_rule_chooses_a_clear_winner():
    base_rr = _rr(300, 0.5, 0)
    rr = {"uri": base_rr, "label": base_rr + 0.0, "core": base_rr + 0.2, "role": base_rr + 0.0}
    result = choose_order(rr)
    assert result["best"] == "core" and result["eligible"] == ["core"] and result["chosen"] == "core"


def test_indistinguishable_orders_go_to_the_precedence():
    rr = {"uri": _rr(300, 0.5, 1), "label": _rr(300, 0.5, 2), "core": _rr(300, 0.503, 3), "role": _rr(300, 0.4, 4)}
    result = choose_order(rr)
    assert result["best"] == "core" and "label" in result["eligible"] and result["chosen"] == "label"
    assert result["candidates"]["label"]["in_E"] and result["candidates"]["label"]["best_minus"]["low"] <= 0
    assert not result["candidates"]["role"]["in_E"]


def test_exact_ties_break_by_precedence_and_the_rule_is_deterministic():
    same = _rr(50, 0.5, 2)
    rr = {o: same for o in ("uri", *PRECEDENCE)}
    assert choose_order(rr)["best"] == PRECEDENCE[0]
    assert choose_order(rr) == choose_order(rr)


def test_rule_requires_every_order_aligned():
    with pytest.raises(ValueError):
        choose_order({"uri": np.zeros(3), "label": np.zeros(3), "core": np.zeros(3)})
    with pytest.raises(ValueError):
        choose_order({"uri": np.zeros(3), "label": np.zeros(3), "core": np.zeros(3), "role": np.zeros(4)})


def test_paired_interval_of_a_constant_difference_is_degenerate():
    out = paired_interval(np.full(20, 0.75), np.full(20, 0.5))
    assert out["estimate"] == pytest.approx(0.25) and out["low"] == pytest.approx(0.25) and out["n"] == 20


# ---------------------------------------------------------------------------
# Freeze guard
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    for key, value in (("user.name", "t"), ("user.email", "t@example.org"), ("commit.gpgsign", "false")):
        _git(repo, "config", key, value)
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / "code.txt").write_text("validation code\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "validation")
    validation_commit = _git(repo, "rev-parse", "HEAD")
    record = repo / "freeze.json"
    record.write_text(json.dumps({"order_definitions_sha256": order_definitions_sha256(), "chosen": "label",
                                  "validation_commit": validation_commit}))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "freeze")
    monkeypatch.chdir(repo)
    return repo, record


def test_freeze_guard_refuses_an_absent_record(tmp_path):
    with pytest.raises(runner.FreezeError, match="No freeze record"):
        runner.verify_freeze(tmp_path / "missing.json")


def test_freeze_guard_refuses_an_unpushed_freeze(frozen_repo):
    _repo, record = frozen_repo
    with pytest.raises(runner.FreezeError, match="remote"):
        runner.verify_freeze(Path(record.name))


def test_freeze_guard_accepts_a_pushed_clean_freeze(frozen_repo):
    repo, record = frozen_repo
    _git(repo, "push", "-q", "origin", "main")
    verified = runner.verify_freeze(Path(record.name))
    assert verified["chosen"] == "label" and len(verified["_sha256"]) == 64


def test_freeze_guard_refuses_a_dirty_tree(frozen_repo):
    repo, record = frozen_repo
    _git(repo, "push", "-q", "origin", "main")
    (repo / "code.txt").write_text("changed after the freeze\n")
    with pytest.raises(runner.FreezeError, match="not clean"):
        runner.verify_freeze(Path(record.name))


def test_freeze_guard_refuses_changed_order_definitions(frozen_repo):
    repo, record = frozen_repo
    data = json.loads(record.read_text())
    record.write_text(json.dumps({**data, "order_definitions_sha256": "0" * 64}))
    _git(repo, "commit", "-qam", "tamper")
    _git(repo, "push", "-q", "origin", "main")
    with pytest.raises(runner.FreezeError, match="order definitions"):
        runner.verify_freeze(Path(record.name))


def test_results_guard_refuses_historical_trees():
    for directory in ("data/results", "data/results_predicate_roles/", "data/results_ppas_controlled/"):
        with pytest.raises(ValueError):
            runner.guard_results_dir(Path(directory))
    runner.guard_results_dir(Path("data/results_serialization/"))


# ---------------------------------------------------------------------------
# Validation stage writes no test ranks
# ---------------------------------------------------------------------------

EX = "http://example.org/"


def _synthetic_pair() -> AlignmentPair:
    source, target = Graph(), Graph()
    for i in range(1, 9):
        s, t = URIRef(f"{EX}s{i}"), URIRef(f"{FMA}fma{i}")
        source.add((s, RDF.type, OWL.Class))
        source.add((s, RDFS.label, Literal(f"source thing {i}")))
        source.add((s, SKOS.altLabel, Literal(f"alt {i}")))
        target.add((t, RDF.type, OWL.Class))
        target.add((t, RDFS.label, Literal(f"target thing {i}")))
        target.add((t, URIRef(f"{FMA}synonym"), Literal(f"syn {i}")))
    return AlignmentPair(
        dataset_id="DX", pair_name="dx", source_graph=source, target_graph=target,
        source_entities=[f"{EX}s{i}" for i in range(1, 9)],
        val_refs=[(f"{EX}s1", f"{FMA}fma1"), (f"{EX}s2", f"{FMA}fma2")],
        test_refs=[(f"{EX}s3", f"{FMA}fma3"), (f"{EX}s4", f"{FMA}fma4")],
    )


class _StubEncoder:
    def encode_batch(self, texts, role, show_progress=False):
        vectors = [np.random.default_rng(sum(map(ord, t))).normal(size=8) for t in texts]
        return np.array([v / np.linalg.norm(v) for v in vectors])


def _stage(tmp_path: Path, population: str) -> runner.StageSettings:
    run = base.RunSettings(data_dir=tmp_path, results_dir=tmp_path / population, force_recompute=True,
                           model_info={"model_id": "stub", "model_key": "M2", "device": "cpu", "hf_revision": "r"},
                           tokenizer_id="stub", max_seq_length=512, encoder_settings={})
    return runner.StageSettings(run, population)


def _context(pair: AlignmentPair) -> base.PairContext:
    candidates = {s: [f"{FMA}fma{i}" for i in range(1, 9)] for s in pair.source_entities}
    builder = PolicySelector("full", _words)
    source = base.collect_side(pair.source_graph, pair.source_entities, "class", builder)
    candidate = base.collect_side(pair.target_graph, sorted({u for c in candidates.values() for u in c}), "class",
                                  builder)
    return base.PairContext(pair, candidates, source, candidate, base.EmbeddingCache(_StubEncoder()), _words)


def test_validation_stage_ranks_and_records_only_validation_sources(tmp_path):
    full = _synthetic_pair()
    scored = runner.scored_pair(full, runner.VALIDATION)
    assert set(scored.source_entities).isdisjoint({f"{EX}s3", f"{EX}s4"})
    condition = next(c for c in SERIALIZATION_CONDITIONS if c.condition_id == "SO-core-plus-syn-label-full")
    payload = runner.run_condition(condition, _context(scored), full, _stage(tmp_path, runner.VALIDATION))
    sidecar = json.loads((tmp_path / "validation" / condition.condition_id / "DX" / "dx_entities.json").read_text())
    assert set(sidecar["reference_sources"]) == {"validation"}
    assert [r["source"] for r in sidecar["reference_sources"]["validation"]] == [f"{EX}s1", f"{EX}s2"]
    assert payload["serialization"]["population"] == "validation"
    assert payload["population"]["population"] == "validation" and "n_test_refs" not in payload["population"]
    assert all(s not in sidecar["selection_fingerprints"]["source"] for s in (f"{EX}s3", f"{EX}s4"))


def test_test_stage_records_both_populations(tmp_path):
    full = _synthetic_pair()
    condition = next(c for c in SERIALIZATION_CONDITIONS if c.condition_id == "SO-core-plus-syn-uri-full")
    runner.run_condition(condition, _context(full), full, _stage(tmp_path, runner.TEST))
    sidecar = json.loads((tmp_path / "test" / condition.condition_id / "DX" / "dx_entities.json").read_text())
    assert set(sidecar["reference_sources"]) == {"validation", "test"}
    record = sidecar["reference_sources"]["test"][0]
    assert record["source_fp"][0] == sidecar["selection_fingerprints"]["source"][record["source"]]


# ---------------------------------------------------------------------------
# Reproduction gates
# ---------------------------------------------------------------------------


def test_fingerprint_comparison_counts_mismatched_and_missing_entities():
    stored = {"a": "1", "b": "2", "c": "3"}
    assert inventory.compare_fingerprints(stored, ["a", "b", "c"], ["1", "2", "3"], True) == (3, 0)
    assert inventory.compare_fingerprints(stored, ["a", "b", "c"], ["1", "x", "3"], True) == (3, 1)
    assert inventory.compare_fingerprints(stored, ["a", "b"], ["1", "2"], True) == (2, 1)


def test_rank_gate_flags_compensating_entity_differences():
    lmap = {"D2/S/A0/uri": {"condition_id": "SO-all-uri-full"}}
    cells = {("D2", "SO-all-uri-full"): {"s1": {"best_rank": "1"}, "s2": {"best_rank": "2"}}}
    same = report.rank_gate(cells, {("D2", "D2/S/A0/uri"): {"s1": 1, "s2": 2}}, lmap)
    swapped = report.rank_gate(cells, {("D2", "D2/S/A0/uri"): {"s1": 2, "s2": 1}}, lmap)
    assert same[0]["status"] == "pass"
    assert swapped[0]["mrr"] == swapped[0]["stored_mrr"] and swapped[0]["n_rank_mismatch"] == 2
    assert swapped[0]["status"] == "FAIL"


def test_label_offset_counts_tokens_before_the_label():
    assert inventory.label_offset("label: A | type: Class", ["label: A"], _words) == 0
    assert inventory.label_offset("type: Class | label: A", ["label: A"], _words) == 3
    assert inventory.label_offset("type: Class", ["label: A"], _words) is None


# ---------------------------------------------------------------------------
# Bundle: rebuild from artifacts alone, URI-only
# ---------------------------------------------------------------------------


def _synthetic_bundle(root: Path) -> Path:
    artifacts = root / "bundle"
    rng = np.random.default_rng(0)
    rows = []
    for condition_id, dataset in executions():
        for i in range(12):
            rank = int(rng.integers(1, 6))
            rows.append({"stage": "test", "dataset": dataset, "condition_id": condition_id, "population": "test",
                         "source": f"{EX}{dataset}s{i}", "ranked": 1, "n_gold": 1, "best_rank": rank,
                         "rr": f"{1 / rank:.10f}", "hit1": int(rank == 1), "hit5": 1, "hit10": 1,
                         "source_text_fp": condition_id[:8], "source_set_fp": "s", "gold_text_fps": "g",
                         "gold_set_fps": "g"})
    for dataset in ("D1", "D2"):
        report.write_gzip_csv(artifacts / "outcomes" / f"test_{dataset}.csv.gz", report.OUTCOME_COLUMNS,
                              [r for r in rows if r["dataset"] == dataset])
    report.write_csv(artifacts / "conditions.csv", report.CONDITION_COLUMNS,
                     [{k: getattr(c, k) if getattr(c, k) is not None else "" for k in report.CONDITION_COLUMNS}
                      for c in LOGICAL_CONDITIONS])
    gates = [{"gate": "fixed_set_identity", "subject": "x", "side": "source", "n_entities": 1, "n_mismatch": 0,
              "status": "pass", "detail": ""}]
    report.write_csv(artifacts / "diagnostics" / "identity_gates.csv", list(gates[0]), gates)
    labels = [{"logical": "D1/S/A0/uri", "side": "source", "n_with_label": 1, "label_retention": 1.0,
               "median_label_offset": 4.0, "share_label_first": 0.0}]
    report.write_csv(artifacts / "diagnostics" / "label_retention.csv", list(labels[0]), labels)
    changes = [{"logical": "D1/B48/C-uri", "side": "source", "n_set_differs": 1, "items_gained": "{}",
                "items_lost": "{}", "labels_lost": 0, "labels_gained": 0}]
    report.write_csv(artifacts / "diagnostics" / "selection_changes.csv", list(changes[0]), changes)
    return artifacts


def test_report_rebuilds_byte_identically_from_the_bundle_alone(tmp_path):
    artifacts = _synthetic_bundle(tmp_path)
    path = report.report("test", artifacts)
    assert "Block B" in path.read_text() and (artifacts / "figures" / "test_order_effects.png").exists()
    assert report.verify_rebuild("test", artifacts) == 0


def test_bundle_text_scan_finds_entity_text(tmp_path):
    artifacts = _synthetic_bundle(tmp_path)
    assert report.files_with_text(artifacts, {"Distinctive entity label"}) == []
    (artifacts / "leak.md").write_text("an example: Distinctive entity label\n")
    assert report.files_with_text(artifacts, {"Distinctive entity label"}) == [artifacts / "leak.md"]
