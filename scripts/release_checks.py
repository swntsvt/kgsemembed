"""Compare D3 and D5 with their original published releases (#138).

``dataset_provenance.py`` compares every D1-D4 file with the OAEI MELT track
repository.  That establishes identity with MELT's ``conference-v1``, not
with the Conference track's own reference alignment, and MELT does not carry
D5.  This script closes both gaps from the original archives, read in place
(nothing is extracted or downloaded):

* **D3**: each local ``reference.rdf`` against the Conference track's ra1
  archive (``reference-alignment.zip``).  A file that differs in bytes is
  compared cell by cell (entities, relation, measure, in order), by its
  alignment header (every element outside the cells, with attributes and
  text), and by its element tree with depths, so a whitespace difference, a
  moved wrapper element and a different alignment are told apart.
* **D5**: every file the D5 loader reads against ``OpenEA_dataset_v2.0.zip``,
  whose SHA-512 is checked against the digest pykeen pins for that archive.

Usage
-----
curl -L -o reference-alignment.zip \\
    https://oaei.ontologymatching.org/2024/conference/data/reference-alignment.zip
curl -L -A Mozilla/5.0 -o OpenEA_dataset_v2.0.zip \\
    https://ndownloader.figshare.com/files/34234391
python scripts/release_checks.py --data_dir data/ \\
    --ra1_zip reference-alignment.zip --openea_zip OpenEA_dataset_v2.0.zip \\
    --output artifacts/issue-138/release_checks.md
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dataset_provenance import _table as _provenance_table  # noqa: E402
from scripts.dataset_provenance import file_digest, sha256  # noqa: E402
from scripts.reference_statistics import alignment_cells  # noqa: E402

RA1_URL = "https://oaei.ontologymatching.org/2024/conference/data/reference-alignment.zip"
OPENEA_URL = "https://figshare.com/ndownloader/files/34234391"
OPENEA_DIRECT_URL = "https://ndownloader.figshare.com/files/34234391"
OPENEA_SHA512 = (
    "c1589f185f86e05c497de147b4d6c243c66775cb4b50c6b41ecc71b36cfafb4c"
    "9f86fbee94e1e78a7ee056dd69df1ce3fc210ae07dc64955ad2bfda7450545ef"
)
OPENEA_PREFIX = "OpenEA_dataset_v2.0/D_W_15K_V2/"
D5_FILES = (
    "rel_triples_1", "rel_triples_2", "attr_triples_1", "attr_triples_2", "ent_links",
    "721_5fold/1/train_links", "721_5fold/1/valid_links", "721_5fold/1/test_links",
)
_ALIGNMENT_NS = "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}"
_MAP_TAG = f"{_ALIGNMENT_NS}map"
_CELL_TAG = f"{_ALIGNMENT_NS}Cell"

HeaderElement = Tuple[str, Tuple[Tuple[str, str], ...], str]
TreeElement = Tuple[int, str, Tuple[Tuple[str, str], ...], str]


def _element(e: ET.Element) -> HeaderElement:
    return e.tag, tuple(sorted(e.attrib.items())), (e.text or "").strip()


def alignment_header(data: bytes) -> List[HeaderElement]:
    """
    Read every element of an alignment file that is not part of a cell.

    Parameters
    ----------
    data : bytes
        Alignment file content.

    Returns
    -------
    List[HeaderElement]
        ``(tag, sorted attributes, stripped text)`` per element in document
        order, skipping each ``Cell`` with its contents and the ``map``
        wrappers themselves (wherever a ``map`` happens to enclose other
        elements, those are still read).
    """
    root = ET.fromstring(data)
    in_cells: Set[int] = {id(e) for cell in root.iter(_CELL_TAG) for e in cell.iter()}
    return [_element(e) for e in root.iter() if id(e) not in in_cells and e.tag != _MAP_TAG]


def element_tree(data: bytes) -> List[TreeElement]:
    """
    Flatten a whole XML document with each element's depth, ignoring whitespace.

    Parameters
    ----------
    data : bytes
        File content.

    Returns
    -------
    List[TreeElement]
        ``(depth, tag, sorted attributes, stripped text)`` in document order;
        two files with equal lists differ only in whitespace or quoting.
    """
    def walk(element: ET.Element, depth: int) -> List[TreeElement]:
        rows = [(depth, *_element(element))]
        for child in element:
            rows += walk(child, depth + 1)
        return rows

    return walk(ET.fromstring(data), 0)


def compare_alignment(local: bytes, release: bytes) -> str:
    """
    Describe how a local alignment relates to its released counterpart.

    Parameters
    ----------
    local, release : bytes
        File contents.

    Returns
    -------
    str
        ``"identical"``; with equal cells and header,
        ``"same cells and header, whitespace differs"`` or
        ``"same cells and header, element nesting differs"``;
        ``"same cells, header differs"``; ``"same cells, order differs"``;
        else the cell counts.
    """
    if local == release:
        return "identical"
    local_cells, release_cells = alignment_cells(local), alignment_cells(release)
    if local_cells != release_cells:
        if sorted(local_cells, key=str) == sorted(release_cells, key=str):
            return "same cells, order differs"
        return f"DIFFERENT cells ({len(local_cells)} local, {len(release_cells)} release)"
    if alignment_header(local) != alignment_header(release):
        return "same cells, header differs"
    if element_tree(local) == element_tree(release):
        return "same cells and header, whitespace differs"
    return "same cells and header, element nesting differs"


def ra1_members(archive: zipfile.ZipFile) -> Dict[str, str]:
    """
    Map each ra1 alignment to the local D3 pair directory name.

    Parameters
    ----------
    archive : zipfile.ZipFile
        The ra1 archive.

    Returns
    -------
    Dict[str, str]
        Local pair name (lower case, e.g. ``conference-confof``) to member name.
    """
    return {
        Path(name).stem.lower(): name
        for name in archive.namelist() if name.endswith(".rdf") and not name.endswith("/")
    }


def check_d3(data_dir: Path, ra1_zip: Path) -> List[List[str]]:
    """
    Compare every local D3 reference with the ra1 archive.

    Parameters
    ----------
    data_dir : Path
        Dataset root.
    ra1_zip : Path
        ``reference-alignment.zip``.

    Returns
    -------
    List[List[str]]
        ``[pair, local SHA-256, release member, status]`` per local pair, plus
        a row for every ra1 member without a local pair.
    """
    pairs = sorted(p.name for p in (data_dir / "d3_conference").iterdir() if p.is_dir())
    with zipfile.ZipFile(ra1_zip) as archive:
        members = ra1_members(archive)
        rows = [_d3_row(data_dir, archive, members, pair) for pair in pairs]
    return rows + [[name, "", members[name], "no local pair"] for name in sorted(set(members) - set(pairs))]


def _d3_row(data_dir: Path, archive: zipfile.ZipFile, members: Dict[str, str], pair: str) -> List[str]:
    path = data_dir / "d3_conference" / pair / "reference.rdf"
    member = members.get(pair)
    if member is None:
        return [pair, sha256(path), "", "not in ra1"]
    return [pair, sha256(path), member, compare_alignment(path.read_bytes(), archive.read(member))]


def d5_status(local_digest: str, release_digest: str) -> str:
    """
    Classify one D5 file from its local and released SHA-256.

    Parameters
    ----------
    local_digest, release_digest : str
        Hex digests, empty when the file is missing on that side.

    Returns
    -------
    str
        ``"missing locally"``, ``"not in archive"``, ``"identical"`` or
        ``"DIFFERENT"``.
    """
    if not local_digest:
        return "missing locally"
    if not release_digest:
        return "not in archive"
    return "identical" if local_digest == release_digest else "DIFFERENT"


def _d5_row(data_dir: Path, archive: zipfile.ZipFile, names: Set[str], name: str) -> List[str]:
    local = data_dir / "d5_openea" / "D_W_15K_V2" / name
    member = OPENEA_PREFIX + name
    local_digest = sha256(local) if local.is_file() else ""
    release_digest = hashlib.sha256(archive.read(member)).hexdigest() if member in names else ""
    return [name, local_digest, release_digest, d5_status(local_digest, release_digest)]


def check_d5(data_dir: Path, openea_zip: Path) -> List[List[str]]:
    """
    Compare every file the D5 loader reads with the OpenEA v2.0 archive.

    Parameters
    ----------
    data_dir : Path
        Dataset root.
    openea_zip : Path
        ``OpenEA_dataset_v2.0.zip``.

    Returns
    -------
    List[List[str]]
        ``[file, local SHA-256, release SHA-256, status]`` per file.
    """
    with zipfile.ZipFile(openea_zip) as archive:
        names = set(archive.namelist())
        return [_d5_row(data_dir, archive, names, name) for name in D5_FILES]


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return _provenance_table(header, rows) + [""]


def _status_summary(rows: List[List[str]]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row[-1]] = counts.get(row[-1], 0) + 1
    return "; ".join(f"{status}: {n}" for status, n in sorted(counts.items()))


def d3_section(rows: List[List[str]], ra1_zip: Path) -> List[str]:
    """
    Render the D3 comparison.

    Parameters
    ----------
    rows : List[List[str]]
        Output of :func:`check_d3`.
    ra1_zip : Path
        Archive compared against.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    return [
        "## D3: Conference ra1", "",
        f"Archive: `{ra1_zip.name}` from {RA1_URL}, SHA-256 `{sha256(ra1_zip)}`.", "",
        f"Summary: {_status_summary(rows)}.", "",
        *_table(["Pair", "Local SHA-256", "ra1 member", "Status"], rows),
    ]


def d5_section(rows: List[List[str]], openea_zip: Path) -> List[str]:
    """
    Render the D5 comparison and the archive's integrity check.

    Parameters
    ----------
    rows : List[List[str]]
        Output of :func:`check_d5`.
    openea_zip : Path
        Archive compared against.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    pinned = "matches" if file_digest(openea_zip, "sha512") == OPENEA_SHA512 else "DOES NOT match"
    return [
        "## D5: OpenEA v2.0", "",
        f"Archive: `{openea_zip.name}`, published at {OPENEA_URL} and fetched from {OPENEA_DIRECT_URL} "
        "(the published link answers a plain HTTP client with an empty 202 response). "
        f"Its SHA-512 {pinned} the digest pykeen pins for this archive (`{OPENEA_SHA512[:16]}…`).", "",
        f"Summary: {_status_summary(rows)}.", "",
        *_table(["File (under D_W_15K_V2/)", "Local SHA-256", "Release SHA-256", "Status"], rows),
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare D3 and D5 with their original releases.")
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--ra1_zip", required=True, help="Conference reference-alignment.zip (ra1).")
    parser.add_argument("--openea_zip", required=True, help="OpenEA_dataset_v2.0.zip.")
    parser.add_argument("--output", default="artifacts/issue-138/release_checks.md", help="Output path.")
    return parser


def main() -> None:
    """Write the release comparison report."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    data_dir, ra1_zip, openea_zip = Path(args.data_dir), Path(args.ra1_zip), Path(args.openea_zip)
    lines = [
        "# Release checks (D3, D5)", "",
        "Generated by `scripts/release_checks.py`; archives are read in place.", "",
        *d3_section(check_d3(data_dir, ra1_zip), ra1_zip),
        *d5_section(check_d5(data_dir, openea_zip), openea_zip),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
