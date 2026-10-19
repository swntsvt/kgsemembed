"""Compare D3 and D5 with their original published releases (#138).

``dataset_provenance.py`` compares every D1-D4 file with the OAEI MELT track
repository.  That establishes identity with MELT's ``conference-v1``, not
with the Conference track's own reference alignment, and MELT does not carry
D5.  This script closes both gaps from the original archives, read in place
(nothing is extracted or downloaded):

* **D3**: each local ``reference.rdf`` against the Conference track's ra1
  archive (``reference-alignment.zip``).  A file that differs in bytes is
  compared cell by cell (entities, relation, measure, in order), so a
  formatting difference is told apart from a different alignment.
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
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dataset_provenance import sha256  # noqa: E402

RA1_URL = "https://oaei.ontologymatching.org/2024/conference/data/reference-alignment.zip"
OPENEA_URL = "https://figshare.com/ndownloader/files/34234391"
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
_RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
_CHUNK = 1 << 20

Cell = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]


def file_digest(path: Path, algorithm: str) -> str:
    """Return the hex digest of a file under ``algorithm``, read in chunks."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _child(cell: ET.Element, tag: str) -> Optional[ET.Element]:
    return cell.find(f"{_ALIGNMENT_NS}{tag}")


def _cell(cell: ET.Element) -> Cell:
    entity1, entity2 = _child(cell, "entity1"), _child(cell, "entity2")
    relation, measure = _child(cell, "relation"), _child(cell, "measure")
    return (
        entity1.get(_RDF_RESOURCE) if entity1 is not None else None,
        entity2.get(_RDF_RESOURCE) if entity2 is not None else None,
        (relation.text or "").strip() if relation is not None else None,
        (measure.text or "").strip() if measure is not None else None,
    )


def alignment_cells(data: bytes) -> List[Cell]:
    """
    Read the cells of an EDOAL alignment in document order.

    Parameters
    ----------
    data : bytes
        Alignment file content.

    Returns
    -------
    List[Cell]
        ``(entity1, entity2, relation, measure)`` per cell.
    """
    return [_cell(cell) for cell in ET.fromstring(data).iter(f"{_ALIGNMENT_NS}Cell")]


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
        ``"identical"``; ``"same cells, formatting differs"`` when the bytes
        differ but every cell matches in order; else the cell counts.
    """
    if local == release:
        return "identical"
    local_cells, release_cells = alignment_cells(local), alignment_cells(release)
    if local_cells == release_cells:
        return "same cells, formatting differs"
    if sorted(local_cells, key=str) == sorted(release_cells, key=str):
        return "same cells, order differs"
    return f"DIFFERENT cells ({len(local_cells)} local, {len(release_cells)} release)"


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
    rows = []
    with zipfile.ZipFile(openea_zip) as archive:
        names = set(archive.namelist())
        for name in D5_FILES:
            local = data_dir / "d5_openea" / "D_W_15K_V2" / name
            member = OPENEA_PREFIX + name
            release = hashlib.sha256(archive.read(member)).hexdigest() if member in names else ""
            status = "identical" if release == sha256(local) else ("not in archive" if not release else "DIFFERENT")
            rows.append([name, sha256(local), release, status])
    return rows


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


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
        f"Archive: `{ra1_zip.name}` from {RA1_URL}, SHA-256 `{file_digest(ra1_zip, 'sha256')}`.", "",
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
    sha512 = file_digest(openea_zip, "sha512")
    pinned = "matches" if sha512 == OPENEA_SHA512 else "DOES NOT match"
    return [
        "## D5: OpenEA v2.0", "",
        f"Archive: `{openea_zip.name}` from {OPENEA_URL}. Its SHA-512 {pinned} the digest pykeen pins "
        f"for this archive (`{OPENEA_SHA512[:16]}…`).", "",
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
