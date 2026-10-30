import sys
from pathlib import Path

import pytest
from rdflib import Graph

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.m4_length_check as check  # noqa: E402
from kgsemembed.datasets import AlignmentPair  # noqa: E402
from kgsemembed.embeddings.models import get_model_config  # noqa: E402
from kgsemembed.verbalisation.budget import budget_tokens  # noqa: E402


def _pair(val_refs, test_refs) -> AlignmentPair:
    return AlignmentPair(dataset_id="D2", pair_name="p", source_graph=Graph(), target_graph=Graph(),
                         source_entities=["s1", "s2", "s3"], val_refs=val_refs, test_refs=test_refs)


def test_validation_pair_keeps_only_validation_reference_sources() -> None:
    pair = check.validation_pair(_pair([("s1", "t1")], [("s2", "t2")]))
    assert pair.source_entities == ["s1"]
    assert pair.test_refs == [("s2", "t2")]


def test_validation_pair_refuses_a_source_in_both_splits() -> None:
    with pytest.raises(ValueError, match="overlap"):
        check.validation_pair(_pair([("s1", "t1")], [("s1", "t2")]))


def test_m4_limit_sets_the_budget_and_restores_the_registry() -> None:
    before = get_model_config("M4")
    with check.m4_limit(512):
        assert get_model_config("M4").max_tokens == 512
        assert budget_tokens("M4") == 420
    assert get_model_config("M4") is before
    assert budget_tokens("M4") == 128


def _row(high: float) -> dict:
    return {"ci_low": high - 0.02, "ci_high": high}


def test_rule_keeps_512_unless_c6_or_c16_is_worse() -> None:
    assert check.decide({"C6/D1": _row(0.01), "C16/D2": _row(0.003)})["limit"] == 512
    assert check.decide({"C11/D1": _row(-0.01), "C17/D2": _row(-0.01)})["limit"] == 512
    decision = check.decide({"C6/D1": _row(0.01), "C16/D2": _row(-0.001)})
    assert decision == {**decision, "limit": 128, "worse_at_512": ["C16/D2"]}
