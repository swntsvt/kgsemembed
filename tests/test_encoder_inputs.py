"""Tests for the encoder-input record and its guards (#183)."""

from types import SimpleNamespace

import pytest

from kgsemembed.pipeline.conditions import get_condition
from kgsemembed.pipeline.encoder_inputs import (
    EncoderInputError,
    condition_budget,
    encoder_input_record,
    length_summary,
)
from kgsemembed.verbalisation.base import VerbaliserBase


class _Encoder:
    def __init__(self, limit: int, lengths: dict) -> None:
        self.model = SimpleNamespace(max_seq_length=limit)
        self.lengths = lengths

    def input_lengths(self, texts, role):
        return [self.lengths.get(text, 2) for text in texts]


class _Plain(VerbaliserBase):
    def verbalise(self, graph, entity_uri, entity_type):
        return ""


def _roles(source_texts, candidate_texts):
    return {
        "source": (_Plain(), [f"s{i}" for i in range(len(source_texts))], source_texts),
        "candidate": (_Plain(), [f"t{i}" for i in range(len(candidate_texts))], candidate_texts),
    }


def test_length_summary_counts_truncation_and_budget() -> None:
    summary = length_summary([10, 420, 421, 600], 512, 420)
    assert summary == {"n_inputs": 4, "n_truncated": 1, "tokens_discarded": 88, "max_length": 600, "n_over_budget": 2}
    assert length_summary([600], 512, None)["n_over_budget"] == 0


def test_condition_budget_only_for_budgeted_strategies() -> None:
    assert condition_budget(get_condition("C4")) == 420
    assert condition_budget(get_condition("C6")) == 420
    assert condition_budget(get_condition("C10")) is None
    assert condition_budget(get_condition("C14")) is None


def test_record_and_sidecar_inputs_per_entity() -> None:
    encoder = _Encoder(512, {"long": 700, "ok": 30})
    record, inputs = encoder_input_record(get_condition("C10"), encoder, _roles(["long"], ["ok", "ok"]))
    assert record["max_seq_length"] == 512 and record["token_budget"] is None
    assert record["source"]["n_truncated"] == 1 and record["source"]["tokens_discarded"] == 188
    assert record["source"]["overhead_tokens"] == 2
    assert inputs["source"] == {"s0": [700, 188]}
    assert inputs["candidate"] == {"t0": [30, 0], "t1": [30, 0]}


def test_budgeted_condition_over_budget_is_refused() -> None:
    encoder = _Encoder(512, {"over": 421})
    with pytest.raises(EncoderInputError, match="C4"):
        encoder_input_record(get_condition("C4"), encoder, _roles(["over"], ["x"]))


def test_unbudgeted_condition_may_be_truncated() -> None:
    encoder = _Encoder(512, {"over": 900})
    record, _ = encoder_input_record(get_condition("C10"), encoder, _roles(["over"], ["x"]))
    assert record["source"]["n_truncated"] == 1


def test_model_at_another_limit_is_refused() -> None:
    with pytest.raises(EncoderInputError, match="registry says 512"):
        encoder_input_record(get_condition("C6"), _Encoder(128, {}), _roles(["x"], ["y"]))
