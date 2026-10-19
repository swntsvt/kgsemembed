"""Tests for the per-pair reference counts and candidate ceilings (#139)."""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgsemembed.evaluation.reference_counts import (  # noqa: E402
    DEFAULT_COUNTS_PATH,
    load_reference_counts,
    source_hits_at_1,
)
from scripts.metric_ceilings import build_report, pair_counts  # noqa: E402
from scripts.reference_statistics import PairAudit, PopulationAudit  # noqa: E402


def _population(refs: int, ref_sources: int, single: int, listed: int, refs_listed: int) -> PopulationAudit:
    return PopulationAudit(
        refs=refs, ref_sources=ref_sources, gold_per_source={"1": single, "2": ref_sources - single},
        unmatched=0, unranked=0, gold_listed=listed, refs_listed=refs_listed,
    )


def _audit() -> PairAudit:
    return PairAudit(
        dataset_id="D1", pair_name="p", source_entities=10, target_pool=10, list_lengths=(20, 20),
        refs=(), kinds={}, outside_sources=0, outside_targets=0, known=None,
        populations={"validation": _population(3, 2, 1, 2, 3), "test": _population(5, 4, 3, 3, 3)},
    )


def test_pair_counts_ceilings_use_their_own_units() -> None:
    """Ranking ceiling per source, Recall@20 and F1 recall ceiling per reference."""
    counts = pair_counts(_audit())
    assert counts["n_test_refs"] == 5 and counts["n_test_reference_sources"] == 4
    assert counts["n_test_multi_target_sources"] == 1
    assert counts["ranking_ceiling"] == pytest.approx(3 / 4)
    assert counts["candidate_recall_at_20"] == pytest.approx(3 / 5)
    assert counts["f1_recall_ceiling"] == pytest.approx(3 / 5)
    assert counts["n_val_refs"] == 3


def test_report_lists_every_pair() -> None:
    lines = build_report([pair_counts(_audit())])
    assert any(line.startswith("| D1 | p | 3 | 5 | 4 | 1 | 0.7500 |") for line in lines)


def _payload(**metrics: float) -> dict:
    return {"dataset_id": "D1", "pair_name": "p", "metrics": metrics, "population": {"n_test_refs": 5}}


def test_source_hits_at_1_derives_from_recall_at_1() -> None:
    counts = {("D1", "p"): pair_counts(_audit())}
    assert source_hits_at_1(_payload(recall_at_1=2 / 5), counts) == pytest.approx(2 / 4)


def test_source_hits_at_1_prefers_the_recorded_value() -> None:
    assert source_hits_at_1(_payload(recall_at_1=0.2, hits_at_1=0.75), {}) == 0.75


def test_source_hits_at_1_refuses_a_different_reference_count() -> None:
    counts = {("D1", "p"): {**pair_counts(_audit()), "n_test_refs": 6}}
    with pytest.raises(ValueError, match="test references"):
        source_hits_at_1(_payload(recall_at_1=0.4), counts)


def test_load_reference_counts_keys_by_dataset_and_pair(tmp_path: Path) -> None:
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({"pairs": [pair_counts(_audit())]}))
    assert set(load_reference_counts(path)) == {("D1", "p")}


@pytest.mark.skipif(not (_REPO_ROOT / DEFAULT_COUNTS_PATH).exists(), reason="counts file not generated")
def test_tracked_counts_cover_every_pair_and_satisfy_the_unit_bounds() -> None:
    """26 pairs; per-source ceilings bound the per-reference ones from above."""
    counts = load_reference_counts(_REPO_ROOT / DEFAULT_COUNTS_PATH)
    assert len(counts) == 26
    for row in counts.values():
        assert row["n_test_reference_sources"] <= row["n_test_refs"]
        assert row["f1_recall_ceiling"] <= row["ranking_ceiling"]
        assert row["f1_recall_ceiling"] <= row["candidate_recall_at_20"]
