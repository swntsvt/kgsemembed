"""Tests for the results inventory script (#138)."""

import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.results_inventory as inventory  # noqa: E402

_DIGEST = "d" * 64


def _payload(**overrides) -> dict:
    payload = {
        "condition_id": "C2", "dataset_id": "D2", "pair_name": "p", "strategy": "V2",
        "model_key": "M2", "hf_revision": "rev", "git_commit": "a" * 40, "git_dirty": False,
        "evaluation_protocol": inventory.EVALUATION_PROTOCOL_VERSION,
        "annotation_predicates_sha256": _DIGEST, "population": {"n_train_refs": 0},
        "candidates_sha256": "c", "versions": {"torch": "1", "numpy": "2"},
    }
    return {**payload, **overrides}


def _record(**overrides) -> inventory.ResultRecord:
    return inventory._record("results", _payload(**overrides))


def _write(root: Path, payload: dict) -> None:
    directory = root / payload["condition_id"] / payload["dataset_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{payload['pair_name']}_results.json").write_text(json.dumps(payload))


def test_load_records_reads_every_result_and_skips_sidecars(tmp_path: Path) -> None:
    root = tmp_path / "results_x"
    _write(root, _payload())
    _write(root, _payload(condition_id="C1", strategy="V1"))
    (root / "C1" / "D2" / "p_entities.json").write_text("{}")
    records = inventory.load_records(root)
    assert [r.condition_id for r in records] == ["C1", "C2"]
    assert records[0].tree == "results_x"
    assert records[0].versions == '{"numpy": "2", "torch": "1"}'


def test_reads_annotations_only_for_v1_v2_on_d1_d2() -> None:
    assert inventory.reads_annotations(_record(strategy="V2+V8"))
    assert inventory.reads_annotations(_record(strategy="V1", dataset_id="D1"))
    assert not inventory.reads_annotations(_record(strategy="V3"))
    assert not inventory.reads_annotations(_record(strategy="V2", dataset_id="D3"))


def test_digest_state() -> None:
    assert inventory.digest_state(_record(), _DIGEST) == "current"
    assert inventory.digest_state(_record(), "e" * 64) == "other"
    assert inventory.digest_state(_record(annotation_predicates_sha256=None), _DIGEST) == "absent"


def test_flags_report_every_reason() -> None:
    hashes = {("D2", "p"): {"c"}}
    assert inventory.flags(_record(), _DIGEST, hashes) == []
    assert inventory.flags(_record(git_dirty=None), _DIGEST, hashes) == ["dirty or unknown tree"]
    assert inventory.flags(_record(evaluation_protocol=1), _DIGEST, hashes) == ["protocol 1"]
    assert inventory.flags(_record(annotation_predicates_sha256=None), _DIGEST, hashes) == ["V1/V2 text before #136"]
    d1 = _record(dataset_id="D1", population={"n_train_refs": 0})
    assert inventory.flags(d1, _DIGEST, {}) == ["D1 before the known-pair fix (#137)"]
    assert inventory.flags(_record(), _DIGEST, {("D2", "p"): {"c", "x"}}) == ["candidate hash differs within the pair"]


def test_v8_text_before_145_is_flagged_only_where_it_changed() -> None:
    current = inventory.v8_exclusions_digest()
    assert inventory.flags(_record(strategy="V2+V8"), _DIGEST, {}) == ["V8 text before #145"]
    assert inventory.flags(_record(strategy="V2+V8", v8_exclusions_sha256=current), _DIGEST, {}) == []
    assert inventory.flags(_record(strategy="V8", dataset_id="D4_schema"), _DIGEST, {}) == []
    assert inventory.flags(_record(strategy="V8", dataset_id="D3"), _DIGEST, {}) == []
    assert inventory.reads_v8_exclusions(_record(strategy="V8", dataset_id="D5"))


def test_absent_digest_is_not_flagged_where_text_did_not_change() -> None:
    record = _record(strategy="V3", annotation_predicates_sha256=None)
    assert inventory.flags(record, _DIGEST, {}) == []


def test_candidate_hashes_group_by_pair() -> None:
    records = [_record(), _record(candidates_sha256="x"), _record(pair_name="q")]
    assert inventory.candidate_hashes(records) == {("D2", "p"): {"c", "x"}, ("D2", "q"): {"c"}}


def test_commit_info_for_missing_and_existing_commits() -> None:
    missing = inventory.commit_info("0" * 40, "develop", "main")
    assert (missing.exists, missing.release_tags, missing.on_develop, missing.on_main) == (False, (), False, False)
    head = inventory._git("rev-parse", "HEAD")
    info = inventory.commit_info(head, "HEAD", "HEAD")
    assert info.exists and info.subject and len(info.date) == 10
    assert info.on_develop and info.on_main


def test_release_tags_lists_only_v_tags_sorted(monkeypatch) -> None:
    calls = []

    def fake_git(*args):
        calls.append(args)
        return "v0.2.1-phase2-docs\nv0.2.0-phase2-results\n"

    monkeypatch.setattr(inventory, "_git", fake_git)
    assert inventory.release_tags("a" * 40) == ("v0.2.0-phase2-results", "v0.2.1-phase2-docs")
    assert calls == [("tag", "--contains", "a" * 40, "--list", "v*")]


def test_release_tags_ignore_new_branches_and_other_tags(tmp_path: Path, monkeypatch) -> None:
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "c"], check=True)
    subprocess.run([*git, "tag", "v1.0"], check=True)
    monkeypatch.setattr(inventory, "_REPO_ROOT", tmp_path)
    commit = inventory._git("rev-parse", "HEAD")
    before = inventory.commit_info(commit, "HEAD", "HEAD")
    subprocess.run([*git, "branch", "feature/x"], check=True)
    subprocess.run([*git, "tag", "results/issue-1"], check=True)
    assert inventory.commit_info(commit, "HEAD", "HEAD") == before
    assert before.release_tags == ("v1.0",)


def test_develop_ref_prefers_the_remote_tracking_branch(monkeypatch) -> None:
    monkeypatch.setattr(inventory, "_git", lambda *args: "abc")
    assert inventory.develop_ref() == "origin/develop"
    assert inventory.tracking_ref("main") == "origin/main"
    monkeypatch.setattr(inventory, "_git", lambda *args: None)
    assert inventory.develop_ref() == "develop"
    assert inventory.tracking_ref("main") == "main"


def test_group_section_lists_conditions_in_numeric_order() -> None:
    records = [_record(condition_id=c) for c in ("C10", "C2", "C1")]
    lines = inventory.group_section(records, _DIGEST)
    assert lines[4].endswith("| 3 | C1, C2, C10 |")


def test_commit_section_marks_untagged_and_missing_commits() -> None:
    infos = [
        inventory.CommitInfo("a" * 40, True, "2026-01-01", "x | y", True, False, ()),
        inventory.CommitInfo("b" * 40, False, "", "", False, False, ()),
        inventory.CommitInfo("c" * 40, True, "2026-01-02", "z", True, True, ("v1", "v2")),
    ]
    lines = inventory.commit_section(infos, Counter({"a" * 40: 2}))
    assert lines[4].endswith("| no | no | **missing** |")
    assert lines[5].endswith("| yes | no | none |") and "x \\| y" in lines[5]
    assert lines[6].endswith("| yes | yes | v1, v2 |")


def test_flag_section_states_when_nothing_is_flagged() -> None:
    assert inventory.flag_section([_record()], _DIGEST)[2].startswith("None:")
    dirty = replace(_record(), git_dirty=True)
    assert "dirty or unknown tree" in inventory.flag_section([dirty], _DIGEST)[4]
