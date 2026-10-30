import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.encode_timing as timing  # noqa: E402
import scripts.encoded_input_size as input_size  # noqa: E402
import scripts.generate_cost_report as cost  # noqa: E402


def test_compared_conditions_leave_out_the_sanity_check():
    ids = [condition.condition_id for condition in input_size.compared_conditions("D1")]
    assert "C19" not in ids
    assert len(ids) == 15


def test_compared_conditions_of_d4_and_d5():
    assert len(input_size.compared_conditions("D4")) == 12
    assert len(input_size.compared_conditions("D5")) == 13


def test_length_summary_caps_encoded_tokens_at_the_limit():
    summary = input_size.length_summary([10, 20, 600], max_tokens=512)
    assert summary["n_texts"] == 3
    assert summary["tokens"] == 630
    assert summary["encoded_tokens"] == 542
    assert summary["n_over_limit"] == 1
    assert summary["median"] == 20
    assert summary["max"] == 600


def test_length_summary_of_no_texts():
    assert input_size.length_summary([], max_tokens=512)["n_texts"] == 0


def _stored(tmp_path: Path, **metrics: float) -> Path:
    path = tmp_path / "pair_results.json"
    path.write_text(json.dumps({"metrics": metrics}))
    return path


def test_check_reproduces_accepts_equal_metrics(tmp_path):
    values = dict(mrr=0.5, hits_at_1=0.4, f1=0.3, precision=0.2, recall=0.1, threshold=0.8)
    timing.check_reproduces(values, _stored(tmp_path, **values))


def test_check_reproduces_refuses_any_difference(tmp_path):
    values = dict(mrr=0.5, hits_at_1=0.4, f1=0.3, precision=0.2, recall=0.1, threshold=0.8)
    with pytest.raises(timing.TimingCheckError, match="mrr"):
        timing.check_reproduces({**values, "mrr": 0.5000001}, _stored(tmp_path, **values))


def _record() -> dict:
    return {
        "condition_id": "C1", "strategy": "V1", "model_key": "M1", "model": {"n_parameters": 22_713_216},
        "n_source_texts": 10, "n_candidate_texts": 30, "verbalise_seconds": 1.0,
        "source_encode_seconds": [2.0, 3.0, 2.5], "candidate_encode_seconds": [4.0, 4.0, 9.0],
        "rank_evaluate_seconds": 0.5, "metrics": {"mrr": 0.7},
    }


def test_stage2_row_takes_the_median_encoding_pass():
    row = cost.stage2_row(_record(), {"C1": {"total": {"n_texts": 40, "encoded_tokens": 1234}}})
    assert row["encode"] == 7.0
    assert row["encode_range"] == (6.0, 11.5)
    assert row["stage2"] == 1.0 + 7.0 + 0.5
    assert row["n_texts"] == 40
    assert row["encoded_tokens"] == 1234


def test_stage2_row_refuses_a_different_text_count():
    with pytest.raises(ValueError, match="40 texts"):
        cost.stage2_row(_record(), {"C1": {"total": {"n_texts": 41, "encoded_tokens": 1234}}})


def test_warmup_sample_spreads_over_the_texts():
    texts = [str(index) for index in range(1024)]
    sample = timing.warmup_sample(texts)
    assert len(sample) == timing.WARMUP_TEXTS
    assert sample[0] == "0" and sample[-1] == "1020"
    assert timing.warmup_sample(texts[:10]) == texts[:10]


def _rendered(column: str) -> tuple:
    row = cost.stage2_row(_record(), {"C1": {"total": {"n_texts": 40, "encoded_tokens": 1_234_567}}})
    b0 = {"stage1": 0.21, "stage1_range": (0.2, 0.3), "mrr": 0.75, "n_sources": 10}
    context = {"column": column, "note": "Apple M4 Pro, 48 GB, device mps", "repeats": 3}
    return cost.render_markdown([row], b0, context), cost.render_latex([row], b0, context)


def test_render_shows_rows_against_b0():
    markdown, latex = _rendered("D1")
    assert "| C1 | V1 | M1 | 23 | 40 | 1.23 | 1.0 | 7.0 (6.0–11.5) | 8.5 | 1.0× | 0.7000 | -0.0500 |" in markdown
    assert r"C1 & V1 & M1 & 23 & 1.23 & 0.1 & 1.0 & 0.700 & $-$0.050 \\" in latex
    assert "median of 3 passes" in latex and r"\label{tab:cost_d1}" in latex


def test_render_latex_escapes_the_column_in_the_caption():
    assert r"stage~2 on D4\_instance" in _rendered("D4_instance")[1]


def _provenance(**changes: object) -> dict:
    record = {"pair_name": "p", "git_commit": "abc", "machine": {"chip": "M4"}, "repeats": 3,
              "model": {"device": "mps"}}
    return {**record, **changes}


def test_check_uniform_accepts_agreeing_records():
    stage1 = {key: value for key, value in _provenance().items() if key != "model"}
    cost.check_uniform([_provenance(), _provenance(), stage1])


@pytest.mark.parametrize("change", [{"git_commit": "def"}, {"repeats": 2}, {"machine": {"chip": "M3"}},
                                    {"model": {"device": "cpu"}}])
def test_check_uniform_refuses_mixed_provenance(change):
    with pytest.raises(ValueError, match="differ"):
        cost.check_uniform([_provenance(), _provenance(**change)])


def test_timing_records_refuse_a_multi_pair_column(tmp_path):
    for pair in ("a", "b"):
        (tmp_path / f"C1_D3_{pair}.json").write_text(json.dumps({"pair_name": pair}))
    args = cost.argparse.Namespace(timing_dir=str(tmp_path), column="D3")
    with pytest.raises(ValueError, match="one pair"):
        cost.timing_records(args, "C")


def test_inventory_limit_is_the_encoders_registered_limit(monkeypatch):
    monkeypatch.setattr(input_size, "config_max_seq_length", lambda key: 128)
    pooled = {"source": [100, 200], "candidate": [50]}
    for condition in input_size.compared_conditions("D1"):
        record = input_size.condition_record(condition, pooled, 1)
        assert record["max_seq_length"] == input_size.get_model_config(condition.model_key).max_tokens


def test_inventory_counts_texts_over_the_budget(monkeypatch):
    monkeypatch.setattr(input_size, "config_max_seq_length", lambda key: 512)
    c4 = next(c for c in input_size.compared_conditions("D1") if c.condition_id == "C4")
    record = input_size.condition_record(c4, {"source": [400, 430], "candidate": [600]}, 1)
    assert record["token_budget"] == 420
    assert record["total"]["n_over_budget"] == 2
    assert record["total"]["n_over_limit"] == 1


@pytest.mark.slow
@pytest.mark.parametrize("model_key", ["M1", "M2", "M3", "M4", "M5"])
def test_inventory_and_encoder_agree_on_limit_and_lengths(model_key):
    from kgsemembed.embeddings import EmbeddingEncoder
    from kgsemembed.embeddings.models import load_sentence_transformer

    model, info = load_sentence_transformer(model_key)
    encoder = EmbeddingEncoder(model_key, model)
    texts = ["Heart", "type: Organ | label: Heart of the body " * 40, ""]
    counter = input_size.token_length_counter(model_key)
    for role in ("source", "candidate"):
        prefixed = [encoder._apply_prefix(text, role) for text in texts]
        assert encoder.input_lengths(texts, role) == counter(prefixed)
    assert model.max_seq_length == info["max_seq_length"] == input_size.get_model_config(model_key).max_tokens
