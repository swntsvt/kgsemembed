"""Tests for the D3/D5 release comparison script (#138).

Archives are small synthetic zips written to a temporary directory.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.release_checks as checks  # noqa: E402

_ALIGNMENT = """<?xml version='1.0'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><Alignment>{cells}</Alignment></rdf:RDF>"""
_CELL = ("<map><Cell><entity1 rdf:resource='{s}'/><entity2 rdf:resource='{t}'/>"
         "<measure>1.0</measure><relation>=</relation></Cell></map>")


def _alignment(*pairs: tuple, indent: str = "") -> bytes:
    cells = (indent + "\n").join(_CELL.format(s=s, t=t) for s, t in pairs)
    return _ALIGNMENT.format(cells=cells).encode("utf-8")


def _zip(path: Path, members: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _d3_pair(data_dir: Path, name: str, data: bytes) -> None:
    directory = data_dir / "d3_conference" / name
    directory.mkdir(parents=True)
    (directory / "reference.rdf").write_bytes(data)


def test_compare_alignment_tells_formatting_from_content() -> None:
    base = _alignment(("a", "x"), ("b", "y"))
    assert checks.compare_alignment(base, base) == "identical"
    assert checks.compare_alignment(base, _alignment(("a", "x"), ("b", "y"), indent="  ")) == \
        "same cells, formatting differs"
    assert checks.compare_alignment(base, _alignment(("b", "y"), ("a", "x"))) == "same cells, order differs"
    assert checks.compare_alignment(base, _alignment(("a", "x"))) == "DIFFERENT cells (2 local, 1 release)"


def test_alignment_cells_reads_relation_and_measure() -> None:
    assert checks.alignment_cells(_alignment(("a", "x"))) == [("a", "x", "=", "1.0")]


def test_check_d3_maps_case_and_reports_unpaired_members(tmp_path: Path) -> None:
    data = _alignment(("a", "x"))
    _d3_pair(tmp_path, "cmt-confof", data)
    _d3_pair(tmp_path, "cmt-ekaw", data)
    ra1 = _zip(tmp_path / "ra1.zip", {"cmt-confOf.rdf": data, "edas-iasted.rdf": data})
    rows = checks.check_d3(tmp_path, ra1)
    assert [(r[0], r[2], r[3]) for r in rows] == [
        ("cmt-confof", "cmt-confOf.rdf", "identical"),
        ("cmt-ekaw", "", "not in ra1"),
        ("edas-iasted", "edas-iasted.rdf", "no local pair"),
    ]


def test_check_d5_compares_every_loader_file(tmp_path: Path) -> None:
    local = tmp_path / "d5_openea" / "D_W_15K_V2"
    members = {}
    for name in checks.D5_FILES:
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(name.encode())
        members[checks.OPENEA_PREFIX + name] = name.encode()
    members[checks.OPENEA_PREFIX + "ent_links"] = b"changed"
    del members[checks.OPENEA_PREFIX + "rel_triples_1"]
    rows = {r[0]: r for r in checks.check_d5(tmp_path, _zip(tmp_path / "openea.zip", members))}
    assert rows["rel_triples_1"][3] == "not in archive"
    assert rows["ent_links"][3] == "DIFFERENT"
    assert rows["attr_triples_1"][3] == "identical"
    assert rows["attr_triples_1"][2] == hashlib.sha256(b"attr_triples_1").hexdigest()


def test_d5_section_reports_pinned_digest_mismatch(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "OpenEA_dataset_v2.0.zip", {"x": b"x"})
    lines = checks.d5_section([["f", "a", "a", "identical"]], archive)
    assert "DOES NOT match" in lines[2]
    assert lines[4] == "Summary: identical: 1."
