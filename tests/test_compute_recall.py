"""Tests for the stage-1 candidate recall script.

Candidate files are written to a temporary directory and read back through the
real ``load_candidates``; only ``load_dataset`` is substituted, so no
repository dataset or RDF parsing is involved.
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

import pytest
from rdflib import Graph

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.compute_recall_at_20 as script  # noqa: E402
from kgsemembed.candidates.ngram import dump_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair  # noqa: E402
from kgsemembed.utils.errors import DataError  # noqa: E402
from scripts.compute_recall_at_20 import (  # noqa: E402
    PairRecall,
    _as_ranked_lists,
    _group_by_dataset,
    _micro_average,
    _pair_recall,
)

_REFS = [("s1", "t1"), ("s2", "t2"), ("s3", "t3")]
_CANDIDATES = {"s1": ["t1"], "s2": ["tX", "t2"], "s3": ["tY", "tZ"]}


def _make_pair(dataset_id: str, pair_name: str, refs: List[tuple]) -> AlignmentPair:
    """Build a minimal alignment pair carrying only the references under test."""
    graph = Graph()
    return AlignmentPair(
        dataset_id=dataset_id,
        pair_name=pair_name,
        source_graph=graph,
        target_graph=graph,
        source_entities=[source for source, _ in refs],
        val_refs=[],
        test_refs=refs,
    )


def _write_candidates(data_dir: Path, pair: AlignmentPair, candidates: dict) -> None:
    """Persist candidates where ``load_candidates`` expects to find them."""
    path = data_dir / "candidates" / pair.dataset_id / f"{pair.pair_name}_candidates.json"
    dump_candidates(candidates, path)


@pytest.fixture
def one_pair(tmp_path: Path) -> AlignmentPair:
    """A single D1 pair whose candidates recover two of its three references."""
    pair = _make_pair("D1", "d1_pair", _REFS)
    _write_candidates(tmp_path, pair, _CANDIDATES)
    return pair


def _run_cli(argv: List[str]) -> None:
    original = sys.argv
    sys.argv = ["compute_recall_at_20.py", *argv]
    try:
        script.main()
    finally:
        sys.argv = original


@contextmanager
def _capture_warnings() -> Iterator[List[logging.LogRecord]]:
    """Collect warnings from the script logger, bypassing global log config.

    ``init_logging`` disables propagation on the ``kgsemembed`` logger, so
    pytest's ``caplog`` fixture cannot see these records once any earlier test
    has initialised logging.
    """
    records: List[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("kgsemembed.scripts.compute_recall_at_20")
    handler = _Collector(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_as_ranked_lists_preserves_candidate_order() -> None:
    """Rank order survives the conversion, with strictly descending scores."""
    ranked = _as_ranked_lists({"s1": ["tA", "tB", "tC"]})
    assert [target for _, target, _ in ranked[0]] == ["tA", "tB", "tC"]
    scores = [score for _, _, score in ranked[0]]
    assert scores == sorted(scores, reverse=True)


def test_as_ranked_lists_skips_nothing_for_empty_candidates() -> None:
    """A source with no candidates yields an empty ranked list, not a gap."""
    assert _as_ranked_lists({"s1": []}) == [[]]


def test_pair_recall_counts_hits_within_k(one_pair: AlignmentPair, tmp_path: Path) -> None:
    """Two of three gold targets sit inside the candidate lists."""
    record = _pair_recall(one_pair, str(tmp_path), 20)
    assert record == PairRecall("D1", "d1_pair", pytest.approx(2 / 3), 3)


def test_pair_recall_honours_the_cut_off(one_pair: AlignmentPair, tmp_path: Path) -> None:
    """At k=1 the gold target ranked second no longer counts."""
    record = _pair_recall(one_pair, str(tmp_path), 1)
    assert record.recall == pytest.approx(1 / 3)


def test_pair_recall_raises_for_missing_candidate_file(tmp_path: Path) -> None:
    """A pair with no candidate file surfaces a DataError."""
    pair = _make_pair("D1", "absent", _REFS)
    with pytest.raises(DataError):
        _pair_recall(pair, str(tmp_path), 20)


def test_micro_average_weights_by_reference_count() -> None:
    """Pooled recall follows reference counts, not the unweighted pair mean."""
    records = [PairRecall("D3", "a", 2 / 3, 3), PairRecall("D3", "b", 1.0, 1)]
    assert _micro_average(records) == pytest.approx(0.75)


def test_micro_average_without_references_is_zero() -> None:
    """A dataset whose pairs carry no references scores zero, not NaN."""
    assert _micro_average([PairRecall("D3", "a", 0.0, 0)]) == 0.0


def test_group_by_dataset_keeps_insertion_order() -> None:
    """Records group under their dataset id in the order they arrived."""
    records = [PairRecall("D4_schema", "a", 1.0, 1), PairRecall("D4_instance", "b", 1.0, 1)]
    assert list(_group_by_dataset(records)) == ["D4_schema", "D4_instance"]


def test_cli_reports_each_pair_and_a_dataset_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Both pairs of a dataset are listed, then pooled into one summary row."""
    pairs = [_make_pair("D3", "a", _REFS), _make_pair("D3", "b", [("s9", "t9")])]
    _write_candidates(tmp_path, pairs[0], _CANDIDATES)
    _write_candidates(tmp_path, pairs[1], {"s9": ["t9"]})
    monkeypatch.setattr(script, "load_dataset", lambda dataset_id, data_dir: pairs)

    _run_cli(["--datasets", "D3", "--data_dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "Recall@20" in out
    assert "0.6667" in out
    assert "D3                 2        4      0.7500" in out


def test_cli_k_flag_changes_the_reported_cut_off(
    one_pair: AlignmentPair, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """The header and the figures both follow --k."""
    monkeypatch.setattr(script, "load_dataset", lambda dataset_id, data_dir: [one_pair])

    _run_cli(["--datasets", "D1", "--data_dir", str(tmp_path), "--k", "1"])

    out = capsys.readouterr().out
    assert "Recall@1" in out
    assert "0.3333" in out


def test_cli_skips_a_pair_whose_candidates_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """One unusable pair is warned about; the remaining pair is still reported."""
    pairs = [_make_pair("D3", "present", _REFS), _make_pair("D3", "absent", _REFS)]
    _write_candidates(tmp_path, pairs[0], _CANDIDATES)
    monkeypatch.setattr(script, "load_dataset", lambda dataset_id, data_dir: pairs)

    with _capture_warnings() as records:
        _run_cli(["--datasets", "D3", "--data_dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "present" in out
    assert "absent" not in out
    assert any("absent" in record.getMessage() for record in records)


def test_cli_skips_a_dataset_that_cannot_be_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An unloadable dataset is warned about without aborting the others."""
    pair = _make_pair("D1", "d1_pair", _REFS)
    _write_candidates(tmp_path, pair, _CANDIDATES)

    def fake_load_dataset(dataset_id: str, data_dir: str) -> List[AlignmentPair]:
        if dataset_id == "D9":
            raise DataError("Unknown dataset identifier: 'D9'.")
        return [pair]

    monkeypatch.setattr(script, "load_dataset", fake_load_dataset)

    with _capture_warnings() as records:
        _run_cli(["--datasets", "D9", "D1", "--data_dir", str(tmp_path)])

    assert "d1_pair" in capsys.readouterr().out
    assert any("D9" in record.getMessage() for record in records)


def test_cli_raises_when_nothing_could_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that recovers no candidate file at all fails loudly."""
    pairs = [_make_pair("D3", "absent", _REFS)]
    monkeypatch.setattr(script, "load_dataset", lambda dataset_id, data_dir: pairs)

    with _capture_warnings():
        with pytest.raises(DataError, match="No candidate files"):
            _run_cli(["--datasets", "D3", "--data_dir", str(tmp_path)])
