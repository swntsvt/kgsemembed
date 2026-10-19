"""Tests for the reference statistics script (#138).

Reference files are small synthetic EDOAL and link files; pairs are built in
memory, so no repository dataset is read.
"""

import sys
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, Literal, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.reference_statistics as stats  # noqa: E402
from kgsemembed.datasets import AlignmentPair  # noqa: E402

_EDOAL = """<?xml version='1.0' encoding='utf-8'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
<Alignment>
{cells}
</Alignment>
</rdf:RDF>
"""
_CELL = """<map><Cell>
  <entity1 rdf:resource='{s}'/><entity2 rdf:resource='{t}'/>
  {measure}<relation>{relation}</relation>
</Cell></map>"""


def _cell(s: str, t: str, relation: str = "=", measure: str = "1.0") -> str:
    tag = f"<measure>{measure}</measure>" if measure else ""
    return _CELL.format(s=s, t=t, relation=relation, measure=tag)


def _write_edoal(path: Path, cells: list) -> Path:
    path.write_text(_EDOAL.format(cells="\n".join(cells)), encoding="utf-8")
    return path


def _graph(classes: list, untyped: tuple = ()) -> Graph:
    graph = Graph()
    for uri in classes:
        graph.add((URIRef(uri), RDF.type, OWL.Class))
        graph.add((URIRef(uri), RDFS.label, Literal(uri)))
    for uri in untyped:
        graph.add((URIRef(uri), RDFS.label, Literal(uri)))
    return graph


def _pair(val: list, test: list, train: tuple = (), sources: tuple = ("s1", "s2", "s3", "s4")) -> AlignmentPair:
    return AlignmentPair(
        dataset_id="DX", pair_name="p",
        source_graph=_graph(list(sources)), target_graph=_graph(["t1", "t2", "t3"]),
        source_entities=list(sources), val_refs=val, test_refs=test, train_refs=list(train),
    )


def test_audit_edoal_counts_every_cell_kind(tmp_path: Path) -> None:
    path = _write_edoal(tmp_path / "ref.rdf", [
        _cell("s1", "t1"), _cell("s1", "t1"), _cell("s2", "t2", relation="&lt;"),
        _cell("s3", "t3", measure="0.5"), _cell("s4", "", measure=""),
    ])
    audit = stats.audit_edoal(path, "ref.rdf")
    assert audit.cells == 5
    assert audit.relations == {"=": 4, "<": 1}
    assert audit.partial_measure == 1
    assert audit.malformed == 1
    assert audit.equivalences == (("s1", "t1"), ("s1", "t1"), ("s3", "t3"))


def test_audit_edoal_matches_loader_equivalences(tmp_path: Path) -> None:
    from kgsemembed.datasets.loader import _parse_alignment_refs

    path = _write_edoal(tmp_path / "ref.rdf", [
        _cell("s1", "t1"), _cell("s2", "t2", relation="&gt;"), _cell("s3", "t3", measure="0.2"),
    ])
    assert list(stats.audit_edoal(path, "ref.rdf").equivalences) == _parse_alignment_refs(path)


def test_audit_links_flags_short_lines(tmp_path: Path) -> None:
    path = tmp_path / "links"
    path.write_text("a\tb\nc\n\nd\te\n", encoding="utf-8")
    audit = stats.audit_links(path, "links")
    assert (audit.cells, audit.malformed) == (3, 1)
    assert audit.equivalences == (("a", "b"), ("d", "e"))


def test_file_overlaps_and_d5_union() -> None:
    def audit(name: str, pairs: tuple) -> stats.FileAudit:
        return stats.FileAudit(name, len(pairs), {"=": len(pairs)}, 0, 0, pairs)

    train, valid, test = audit("1/train_links", (("a", "x"),)), audit("1/valid_links", (("b", "y"),)), \
        audit("1/test_links", (("a", "z"),))
    ent = audit("ent_links", (("a", "x"), ("b", "y"), ("a", "z")))
    assert stats.file_overlaps([train, test]) == [("1/train_links", "1/test_links", 0, 1)]
    assert stats.d5_split_union([ent, train, valid, test]) == (3, 3, True)
    assert stats.d5_split_union([ent, train, valid]) is None


def test_cardinality_classifies_by_degree_of_both_ends() -> None:
    refs = [("a", "x"), ("a", "y"), ("b", "z"), ("c", "z"), ("d", "w"), ("d", "w")]
    assert stats.cardinality(refs) == {"1:1": 1, "1:N": 2, "N:1": 2, "N:M": 0}
    assert stats.cardinality([("a", "x"), ("a", "y"), ("b", "x")])["N:M"] == 1


def test_entity_kind_distinguishes_absent_and_untyped() -> None:
    graph = _graph(["c"], untyped=("u",))
    graph.add((URIRef("p"), RDF.type, OWL.ObjectProperty))
    graph.add((URIRef("i"), RDF.type, URIRef("http://example.org/Thing")))
    kinds = [stats.entity_kind(graph, uri) for uri in ("c", "p", "i", "u", "missing")]
    assert kinds == ["class", "predicate", "instance", "untyped", "absent"]


def test_audit_population_counts_gold_lists_and_unranked() -> None:
    refs = [("s1", "t1"), ("s1", "t2"), ("s2", "t3"), ("s3", "t1")]
    candidates = {"s1": ["t2", "t9"], "s2": ["t9"], "s3": [], "s4": ["t1"]}
    audit = stats.audit_population(refs, frozenset({"s1", "s2", "s3", "s4"}), candidates)
    assert (audit.refs, audit.ref_sources, audit.unmatched) == (4, 3, 1)
    assert audit.gold_per_source == {"2": 1, "1": 2}
    assert (audit.unranked, audit.gold_listed, audit.refs_listed) == (1, 1, 1)


def test_audit_known_splits_sources_and_counts_removed_candidates() -> None:
    pair = _pair(val=[("s1", "t1")], test=[("s2", "t2")], train=[("s1", "t3"), ("s2", "t3"), ("s4", "t1")])
    raw = {"s1": ["t3", "t1"], "s2": ["t2"], "s4": ["t1"]}
    known = stats.audit_known(pair, raw, frozenset({"s1", "s2"}))
    assert known == stats.KnownAudit(pairs=3, sources=3, excluded=1, kept_val=1, kept_test=1, removed=1)
    assert stats.audit_known(_pair(val=[], test=[("s1", "t1")]), raw, frozenset()) is None


def test_audit_pair_removes_known_pairs_before_counting_lists() -> None:
    pair = _pair(val=[("s1", "t1")], test=[("s2", "t2"), ("s5", "t9")], train=[("s2", "t1")])
    raw = {"s1": ["t1", "t2"], "s2": ["t1", "t3"], "s3": ["t2"], "s4": ["t3"]}
    audit = stats.audit_pair(pair, raw)
    assert (audit.source_entities, audit.target_pool, audit.list_lengths) == (4, 3, (1, 2))
    assert (audit.outside_sources, audit.outside_targets) == (1, 1)
    assert audit.kinds == {"absent -> absent": 1, "class -> class": 2}
    test = audit.populations["test"]
    assert (test.ref_sources, test.unranked, test.gold_listed, test.refs_listed) == (2, 1, 0, 0)
    assert audit.known.removed == 1


def test_report_adds_a_total_row_only_for_multi_pair_columns() -> None:
    raw = {"s1": ["t1"], "s2": ["t2"]}
    single = stats.audit_pair(_pair(val=[("s1", "t1")], test=[("s2", "t2")]), raw)
    lines = stats.structure_section([single])
    assert not any("**total**" in line for line in lines)
    lines = stats.structure_section([single, single])
    total = next(line for line in lines if "**total**" in line)
    assert total.startswith("| DX | **total** | 4 | 0 | 4 |")


def test_build_report_is_deterministic(tmp_path: Path) -> None:
    path = _write_edoal(tmp_path / "ref.rdf", [_cell("s1", "t1")])
    files = [("DX", stats.audit_edoal(path, "ref.rdf"))]
    audits = [stats.audit_pair(_pair(val=[("s1", "t1")], test=[("s2", "t2")]), {"s1": ["t1"]})]
    assert stats.build_report(files, audits) == stats.build_report(files, audits)
    assert "## 5. Known (training) pairs" not in stats.build_report(files, audits)
