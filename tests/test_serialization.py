"""Tests for the text-serialization-order experiment (#134)."""

import csv
import hashlib
import io
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

import freeze_serialization_order as freeze  # noqa: E402
import generate_serialization_report as report  # noqa: E402
import serialization_inventory as inventory  # noqa: E402

from kgsemembed.datasets import AlignmentPair  # noqa: E402
from kgsemembed.evaluation.order_selection import (  # noqa: E402
    PRECEDENCE,
    SELECTION_REPLICATES,
    SELECTION_SEED,
    choose_order,
    paired_interval,
)
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
    (repo / "src" / "kgsemembed" / "pipeline").mkdir(parents=True)
    (repo / "src" / "kgsemembed" / "selection.py").write_text("frozen\n")
    (repo / "src" / "kgsemembed" / "pipeline" / "serialization_order.py").write_text("runner\n")
    (repo / ".gitignore").write_text("data/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "validation")
    validation_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "data").mkdir()
    (repo / "data" / "val.json").write_text("validation result\n")
    record = repo / "freeze.json"
    record.write_text(json.dumps(_freeze_record(validation_commit, repo / "data" / "val.json")))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "freeze")
    monkeypatch.chdir(repo)
    return repo, record


def _freeze_record(validation_commit: str, validation_file: Path) -> dict:
    return {"order_definitions_sha256": order_definitions_sha256(), "chosen": "label",
            "rule": {"replicates": SELECTION_REPLICATES, "seed": SELECTION_SEED, "precedence": list(PRECEDENCE),
                     "reference": "uri"},
            "validation_commit": validation_commit,
            "validation_files_sha256": {"data/val.json": hashlib.sha256(validation_file.read_bytes()).hexdigest()}}


def _commit_and_push(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", message)
    _git(repo, "push", "-q", "origin", "main")


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


def test_freeze_guard_refuses_changed_implementation(frozen_repo):
    repo, record = frozen_repo
    (repo / "src" / "kgsemembed" / "selection.py").write_text("changed after validation\n")
    _commit_and_push(repo, "change")
    with pytest.raises(runner.FreezeError, match="Implementation changed.*selection.py"):
        runner.verify_freeze(Path(record.name))


def test_freeze_guard_allows_changes_to_the_runner_only(frozen_repo):
    repo, record = frozen_repo
    (repo / "src" / "kgsemembed" / "pipeline" / "serialization_order.py").write_text("extra guard\n")
    _commit_and_push(repo, "guard")
    assert runner.verify_freeze(Path(record.name))["chosen"] == "label"


def test_freeze_guard_refuses_changed_validation_results(frozen_repo):
    repo, record = frozen_repo
    _git(repo, "push", "-q", "origin", "main")
    (repo / "data" / "val.json").write_text("rerun\n")
    with pytest.raises(runner.FreezeError, match="Validation file data/val.json"):
        runner.verify_freeze(Path(record.name))


@pytest.mark.parametrize("field, value", [("seed", 1), ("replicates", 10), ("precedence", ["core", "label", "role"])])
def test_freeze_guard_refuses_a_changed_rule(frozen_repo, field, value):
    repo, record = frozen_repo
    data = json.loads(record.read_text())
    record.write_text(json.dumps({**data, "rule": {**data["rule"], field: value}}))
    _commit_and_push(repo, "tamper")
    with pytest.raises(runner.FreezeError, match="selection rule"):
        runner.verify_freeze(Path(record.name))


def test_test_stage_refuses_a_smoke_subset(tmp_path):
    args = runner._build_arg_parser().parse_args(["--population", "test", "--max_sources", "10"])
    with pytest.raises(runner.FreezeError, match="max_sources"):
        runner.verify_test_stage(args, tmp_path)


def test_test_stage_refuses_results_from_another_freeze(tmp_path, monkeypatch):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    stage_dir = tmp_path / "test"
    path = stage_dir / "SO-all-uri-full" / "D2" / "d2_anatomy_results.json"
    path.parent.mkdir(parents=True)
    freeze_record = {"_sha256": "a" * 64}
    path.write_text(json.dumps({"git_commit": head, "git_dirty": False, "serialization": {"freeze": {"sha256": "a" * 64}}}))
    runner.guard_test_results(stage_dir, freeze_record)
    path.write_text(json.dumps({"git_commit": head, "git_dirty": False, "serialization": {"freeze": {"sha256": "b" * 64}}}))
    with pytest.raises(runner.FreezeError, match="another freeze"):
        runner.guard_test_results(stage_dir, freeze_record)


def test_freeze_verification_reports_differing_fields():
    committed = {"chosen": "label", "created": "t1", "rule": {"seed": 1}}
    assert freeze.differing_fields(committed, {**committed, "created": "t2"}) == []
    assert freeze.differing_fields(committed, {**committed, "chosen": "core", "extra": 1}) == ["chosen", "extra"]


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


GATE_LMAP = {"D2/S/A0/uri": {"condition_id": "SO-all-uri-full"}}
GATE_KEY = ("D2", "D2/S/A0/uri")


def _gate_cells(ranks: dict, gold_fp: str = "c") -> dict:
    rows = {s: {"best_rank": str(r) if r else "", "source_text_fp": f"t{s}", "gold_text_fps": f"g{s}={gold_fp}"}
            for s, r in ranks.items()}
    return {("D2", "SO-all-uri-full"): rows}


def _baseline(ranks: dict) -> dict:
    return {"test_sources": [{"source": s, "gold_targets": [f"g{s}"], "best_rank": r} for s, r in ranks.items()],
            "selection_fingerprints": {"source": {s: f"t{s}" for s in ranks}, "candidate": {f"g{s}": "c" for s in ranks}}}


def _gate(cells: dict, baseline) -> dict:
    return report.rank_gate(cells, {GATE_KEY: baseline}, GATE_LMAP)[0]


def test_rank_gate_flags_compensating_entity_differences():
    same = _gate(_gate_cells({"s1": 1, "s2": 2}), _baseline({"s1": 1, "s2": 2}))
    swapped = _gate(_gate_cells({"s1": 1, "s2": 2}), _baseline({"s1": 2, "s2": 1}))
    assert same["status"] == "pass"
    assert swapped["mrr"] == swapped["stored_mrr"] and swapped["n_rank_mismatch"] == 2
    assert swapped["status"] == "FAIL"


def test_rank_gate_matches_unranked_sources_as_rank_none():
    assert _gate(_gate_cells({"s1": None, "s2": 3}), _baseline({"s1": None, "s2": 3}))["status"] == "pass"


@pytest.mark.parametrize("cells, baseline, field", [
    (_gate_cells({"s1": 1}), _baseline({"s1": 1, "s2": 2}), "n_missing"),
    (_gate_cells({"s1": 1, "s2": 2, "s3": 1}), _baseline({"s1": 1, "s2": 2}), "n_extra"),
    (_gate_cells({"s1": 1}, gold_fp="other"), _baseline({"s1": 1}), "n_text_mismatch"),
    ({}, _baseline({"s1": 1}), "n_missing"),
])
def test_rank_gate_fails_on_missing_extra_or_changed_sources(cells, baseline, field):
    row = _gate(cells, baseline)
    assert row["status"] == "FAIL" and row[field] >= 1


def test_rank_gate_fails_on_a_changed_gold_set():
    baseline = _baseline({"s1": 1})
    baseline["test_sources"][0]["gold_targets"] = ["gs1", "gs9"]
    assert _gate(_gate_cells({"s1": 1}), baseline)["n_gold_mismatch"] == 1


def _exported_cells(source_fp, gold_fp) -> dict:
    record = {"source": "s1", "top": "g1", "gold_targets": ["g1"], "best_rank": 1,
              "source_fp": [source_fp, "set"], "gold_fp": {"g1": [gold_fp, "set"]}}
    rows = report.outcome_rows("test", "D2", "SO-all-uri-full", {"reference_sources": {"test": [record]}})
    parsed = csv.DictReader(io.StringIO(report.csv_bytes(report.OUTCOME_COLUMNS, rows).decode()))
    return {("D2", "SO-all-uri-full"): {r["source"]: r for r in parsed}}


def _stored(source_fp, gold_fp) -> dict:
    fingerprints = {"source": {"s1": source_fp}, "candidate": {"g1": gold_fp}}
    return {"test_sources": [{"source": "s1", "gold_targets": ["g1"], "best_rank": 1}],
            "selection_fingerprints": {side: {k: v for k, v in fps.items() if v is not None}
                                       for side, fps in fingerprints.items()}}


@pytest.mark.parametrize("exported, stored, status", [
    (("t", None), ("t", None), "pass"),
    ((None, None), (None, None), "pass"),
    (("t", None), ("t", "c"), "FAIL"),
    (("t", "c"), ("t", None), "FAIL"),
    ((None, "c"), ("t", "c"), "FAIL"),
    (("t", "c"), ("t", "d"), "FAIL"),
    (("t", "c"), ("t", "c"), "pass"),
])
def test_rank_gate_reads_unencoded_targets_in_the_export_format(exported, stored, status):
    row = _gate(_exported_cells(*exported), _stored(*stored))
    assert row["status"] == status
    assert row["n_text_mismatch"] == (status == "FAIL")


def test_exported_unencoded_gold_target_is_written_as_none():
    cells = _exported_cells("t", None)
    assert cells[("D2", "SO-all-uri-full")]["s1"]["gold_text_fps"] == "g1=None"
    assert report._fps("g1=None;g2=;g3=abc;g4=none") == {"g1": None, "g2": None, "g3": "abc", "g4": "none"}


def test_rank_gate_cannot_pass_without_baseline_records():
    assert _gate(_gate_cells({"s1": 1}), None)["status"] == "FAIL"
    assert _gate({}, _baseline({}))["status"] == "FAIL"


def test_rank_gate_expects_the_thirteen_registered_reproduction_targets():
    targets = report.expected_rank_targets()
    assert len(targets) == 13
    assert targets[("D1", "D1/B48/C-uri")] == ("data/results_ppas_controlled", "PA-all-base-det-48")
    assert ("D2", "D2/B24/C-uri") not in targets


def _result(commit="c", dirty=False, smoke=None, n_test_refs=2) -> dict:
    return {"git_commit": commit, "git_dirty": dirty, "ablation": {"smoke_max_sources": smoke},
            "population": {"n_test_refs": n_test_refs}}


def _sidecar(sources=("s1", "s2")) -> dict:
    return {"reference_sources": {"test": [{"source": s, "gold_targets": ["g"]} for s in sources]}}


@pytest.mark.parametrize("results, sidecars, message", [
    ([_result(), _result(commit="d")], [_sidecar(), _sidecar()], "one clean commit"),
    ([_result(), _result(dirty=True)], [_sidecar(), _sidecar()], "one clean commit"),
    ([_result(), _result(smoke=10)], [_sidecar(), _sidecar()], "smoke"),
    ([_result(), _result()], [_sidecar(), _sidecar(("s1",))], "every test reference"),
    ([_result(), _result()], [_sidecar(), _sidecar(("s1", "s3"))], "sources differ"),
])
def test_export_refuses_partial_or_mixed_runs(results, sidecars, message):
    keyed = {("D2", f"SO-{i}"): sc for i, sc in enumerate(sidecars)}
    report.check_complete("test", [_result(), _result()], {("D2", "SO-0"): _sidecar(), ("D2", "SO-1"): _sidecar()})
    with pytest.raises(ValueError, match=message):
        report.check_complete("test", results, keyed)


def test_primary_interval_widens_under_bonferroni_with_the_same_estimate():
    rng = np.random.default_rng(3)
    a, b = rng.random(200), rng.random(200)
    out = report.primary_interval(a, b)
    assert out["95"]["estimate"] == out["bonferroni"]["estimate"] == pytest.approx(float((a - b).mean()))
    assert out["bonferroni"]["low"] < out["95"]["low"] and out["bonferroni"]["high"] > out["95"]["high"]
    assert out["95"]["low"] == pytest.approx(paired_interval(a, b)["low"])


def _record(source: str, rank, gold: str = "g") -> dict:
    return {"source": source, "gold_targets": [gold], "best_rank": rank, "top": [[gold, 0.9]]}


def test_examples_are_the_largest_gains_and_losses_with_uri_ties():
    sidecars = {}
    for dataset in ("D1", "D2"):
        uri = [_record(f"s{i}", r) for i, r in enumerate([5, 1, 2, 3, 1, 4, 2])]
        frozen = [_record(f"s{i}", r) for i, r in enumerate([1, 1, 1, 1, 2, None, 1])]
        sidecars[(dataset, report.a0_condition(dataset, "uri"))] = {"reference_sources": {"test": uri}}
        sidecars[(dataset, report.a0_condition(dataset, "label"))] = {"reference_sources": {"test": frozen}}
    rows = [r for r in report.example_rows("test", "label", sidecars) if r["dataset"] == "D1"]
    assert [(r["direction"], r["source"]) for r in rows] == [
        ("gain", "s0"), ("gain", "s3"), ("gain", "s2"), ("gain", "s6"), ("loss", "s4"), ("loss", "s5")]
    assert rows[0]["top_uri"] == "g@0.9000" and rows[-1]["rank_frozen"] == ""


def test_selection_sentence_says_when_the_tie_break_chose_the_order():
    record = {"chosen": "label", "rule": {"precedence": ["label", "core", "role"]},
              "d2_validation": {"best": "core", "eligible": ["label", "core"],
                                "candidates": {"label": {"mrr": 0.82}, "core": {"mrr": 0.83}}}}
    assert "not** because it had the highest" in report._selection_sentence(record)
    record["d2_validation"]["best"] = "label"
    assert "because it had the highest" in report._selection_sentence(record)


def test_bundle_files_are_never_replaced_by_different_bytes(tmp_path):
    path = tmp_path / "outcomes" / "validation_D1.csv.gz"
    report.write_once(path, b"validation")
    report.write_once(path, b"validation")
    with pytest.raises(FileExistsError):
        report.write_once(path, b"test")


def test_export_refuses_results_of_another_stage(tmp_path):
    results = [{"serialization": {"population": "validation", "freeze": None}}]
    report.check_stage("validation", results, tmp_path)
    with pytest.raises(ValueError, match="not test-stage"):
        report.check_stage("test", results, tmp_path)


def test_export_refuses_test_results_from_another_freeze(tmp_path):
    (tmp_path / report.FREEZE_NAME).write_text("{}")
    results = [{"serialization": {"population": "test", "freeze": {"sha256": "0" * 64}}}]
    with pytest.raises(ValueError, match="committed freeze"):
        report.check_stage("test", results, tmp_path)


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
    interval = {"estimate": 0.01, "low": -0.01, "high": 0.03}
    record = {"chosen": "label", "rule": {"precedence": list(PRECEDENCE), "n_sources": 12},
              "d2_validation": {"best": "core", "eligible": list(PRECEDENCE), "uri_mrr": 0.7,
                                "candidates": {o: {"mrr": 0.75, "minus_uri": interval, "best_minus": interval,
                                                   "in_E": True} for o in PRECEDENCE}}}
    (artifacts / report.FREEZE_NAME).write_text(json.dumps(record))
    example = {"stage": "test", "dataset": "D1", "direction": "gain", "source": f"{EX}D1s0", "gold_targets": "g",
               "rank_uri": 2, "rank_frozen": 1, "top_uri": "g@0.9", "top_frozen": "g@0.9"}
    report.write_csv(artifacts / "examples" / "test_examples.csv", report.EXAMPLE_COLUMNS, [example])
    (artifacts / "PROVENANCE_test.json").write_text("{}\n")
    return artifacts


def _pass_rank_gate(artifacts: Path) -> None:
    gate = {"status": "pass", "inputs_sha256": report.report_input_digests(artifacts, "test")}
    (artifacts / "diagnostics" / report.GATE_NAME).write_text(json.dumps(gate))


def test_report_rebuilds_byte_identically_from_the_bundle_alone(tmp_path):
    artifacts = _synthetic_bundle(tmp_path)
    _pass_rank_gate(artifacts)
    path = report.report("test", artifacts)
    text = path.read_text()
    assert "Block B" in text and (artifacts / "figures" / "test_order_effects.png").exists()
    assert "Primary contrasts" in text and "Bonferroni 97.5%" in text and "tie-breaking rule" in text
    assert report.verify_rebuild("test", artifacts) == 0


def test_test_report_requires_a_passed_rank_gate_on_the_current_outcomes(tmp_path):
    artifacts = _synthetic_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="No rank gate"):
        report.report("test", artifacts)
    _pass_rank_gate(artifacts)
    gate_path = artifacts / "diagnostics" / report.GATE_NAME
    gate_path.write_text(json.dumps({**json.loads(gate_path.read_text()), "status": "FAIL"}))
    with pytest.raises(RuntimeError, match="failed"):
        report.report("test", artifacts)
    _pass_rank_gate(artifacts)
    outcomes = artifacts / "outcomes" / "test_D1.csv.gz"
    outcomes.write_bytes(outcomes.read_bytes() + b"\0")
    with pytest.raises(RuntimeError, match="different test outcomes"):
        report.report("test", artifacts)
    assert not (artifacts / "report_test.md").exists()


@pytest.mark.parametrize("name", ["conditions.csv", "serialization_freeze.json", "examples/test_examples.csv",
                                  "diagnostics/label_retention.csv", "PROVENANCE_test.json"])
def test_test_report_gate_covers_every_report_input(tmp_path, name):
    artifacts = _synthetic_bundle(tmp_path)
    _pass_rank_gate(artifacts)
    with (artifacts / name).open("a") as handle:
        handle.write(" ")
    with pytest.raises(RuntimeError, match="report inputs"):
        report.report("test", artifacts)


def test_report_refuses_an_incomplete_stage(tmp_path):
    artifacts = _synthetic_bundle(tmp_path)
    cells, lmap = report.load_outcomes(artifacts, "test", "test"), report.logical_map(artifacts)
    report.check_cells(cells, lmap)
    first = next(iter(lmap.values()))
    partial = {k: v for k, v in cells.items() if k != (first["dataset"], first["condition_id"])}
    with pytest.raises(ValueError, match="incomplete"):
        report.check_cells(partial, lmap)
    dropped = {k: dict(list(v.items())[1:]) if k == (first["dataset"], first["condition_id"]) else v
               for k, v in cells.items()}
    with pytest.raises(ValueError, match="other sources"):
        report.check_cells(dropped, lmap)


def test_bundle_text_scan_finds_entity_text(tmp_path):
    artifacts = _synthetic_bundle(tmp_path)
    assert report.files_with_text(artifacts, {"Distinctive entity label"}) == []
    (artifacts / "leak.md").write_text("an example: Distinctive entity label\n")
    assert report.files_with_text(artifacts, {"Distinctive entity label"}) == [artifacts / "leak.md"]
