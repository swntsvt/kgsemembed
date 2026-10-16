"""Tests for the pure helpers of scripts/predicate_role_diagnostics.py (#131)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.predicate_role_diagnostics as diagnostics  # noqa: E402
from kgsemembed.verbalisation.predicate_config import expand_predicate  # noqa: E402
from kgsemembed.verbalisation.selection import EligibleItem  # noqa: E402

ITEMS = [
    EligibleItem(expand_predicate("fma:FMAID"), "FMAID: 12345"),
    EligibleItem(expand_predicate("fma:synonym"), "synonym: Gold synonym"),
    EligibleItem(expand_predicate("rdf:type"), "type: Class"),
    EligibleItem(expand_predicate("rdfs:label"), "label: Gold label"),
]


def test_render_keeps_canonical_order_by_default():
    assert diagnostics.render(ITEMS, {"core", "synonym"}) == (
        "synonym: Gold synonym | type: Class | label: Gold label"
    )


def test_render_moves_last_roles_after_the_rest_without_changing_the_items():
    moved = diagnostics.render(ITEMS, {"core", "synonym"}, last={"synonym"})
    assert moved == "type: Class | label: Gold label | synonym: Gold synonym"
    assert sorted(moved.split(" | ")) == sorted(diagnostics.render(ITEMS, {"core", "synonym"}).split(" | "))


def test_render_masks_only_fmaid_values():
    masked = diagnostics.render(ITEMS, {"core", "identifier"}, mask_id=True)
    assert masked.startswith("FMAID: 0 | ") and "12345" not in masked and "Gold label" in masked


def test_gold_rank_is_the_best_gold_position_or_none():
    ranked = [("a", 0.9, ""), ("g2", 0.8, ""), ("g1", 0.7, "")]
    assert diagnostics.gold_rank(ranked, ["g1", "g2"]) == 2
    assert diagnostics.gold_rank(ranked, ["missing"]) is None


def test_bootstrap_of_a_constant_difference_is_degenerate():
    assert diagnostics.bootstrap(np.full(50, 0.25)) == pytest.approx((0.25, 0.25, 0.25))


def test_probe_contrasts_name_registered_variants():
    assert {name for pair in diagnostics.PROBE_CONTRASTS for name in pair} <= set(diagnostics.PROBES)
