"""Record dataset and reference-alignment provenance for the five datasets (#137).

For every file the loaders read, prints its SHA-256, size and, for references,
the number of correspondences (EDOAL ``<map>`` cells, or lines of an OpenEA
link file).  With ``--release_cache`` pointing at a local copy of the OAEI MELT
track repository (``oaei.webdatacommons.org/``), each local file is also
compared byte for byte with its published counterpart.  Nothing is downloaded.

Usage
-----
python scripts/dataset_provenance.py --data_dir data/ \\
    --release_cache ~/oaei_track_cache/oaei.webdatacommons.org \\
    --output artifacts/issue-137/dataset_provenance.md
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

_CHUNK = 1 << 20
_PAIR_FILES = ("source.rdf", "target.rdf", "reference.rdf")
_D5_FILES = (
    "rel_triples_1", "rel_triples_2", "attr_triples_1", "attr_triples_2", "ent_links",
    "721_5fold/1/train_links", "721_5fold/1/valid_links", "721_5fold/1/test_links",
)
_D4_FILES = (
    "ontologies/memoryalpha.rdf", "ontologies/stexpanded.rdf", "references/memoryalpha-stexpanded.rdf",
)


@dataclass(frozen=True)
class TrackedFile:
    """One file a loader reads, and where its published counterpart lives."""

    dataset_id: str
    local: str
    release: Optional[str]
    is_reference: bool


def _pair_files(dataset_id: str, local_dir: str, release_dir: str) -> List[TrackedFile]:
    return [
        TrackedFile(dataset_id, f"{local_dir}/{name}", f"{release_dir}/{name}", name == "reference.rdf")
        for name in _PAIR_FILES
    ]


def tracked_files(data_dir: Path) -> List[TrackedFile]:
    """
    List every file the loaders read, D1 to D5.

    Parameters
    ----------
    data_dir : Path
        Dataset root; D3 pair directories are discovered under it.

    Returns
    -------
    List[TrackedFile]
        Files with their release-relative paths (``None`` for D5, which the
        MELT repository does not carry).
    """
    d1_release = "bio-ml-equiv-unsupervised/2022/snomed.body-fma.body"
    files = _pair_files("D1", "d1_snomed_fma", d1_release)
    files.append(TrackedFile("D1", "d1_snomed_fma/input.rdf", f"{d1_release}/input.rdf", True))
    files += _pair_files("D2", "d2_anatomy", "anatomy_track/anatomy_track-default/mouse-human-suite")
    for pair in sorted(p.name for p in (data_dir / "d3_conference").iterdir() if p.is_dir()):
        files += _pair_files("D3", f"d3_conference/{pair}", f"conference/conference-v1/{pair}")
    files += [
        TrackedFile("D4", f"d4_kgtrack/{name}", f"knowledgegraph/v4/{name}", name.startswith("references/"))
        for name in _D4_FILES
    ]
    return files + [
        TrackedFile("D5", f"d5_openea/D_W_15K_V2/{name}", None, name.endswith("_links"))
        for name in _D5_FILES
    ]


def file_digest(path: Path, algorithm: str) -> str:
    """
    Return the hex digest of a file under a :mod:`hashlib` algorithm.

    Parameters
    ----------
    path : Path
        File to hash, read in chunks.
    algorithm : str
        :mod:`hashlib` algorithm name, e.g. ``"sha256"``.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    return file_digest(path, "sha256")


def correspondence_count(path: Path) -> int:
    """Count EDOAL ``<map>`` cells, or the lines of a tab-separated link file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rdf":
        return text.count("<map>")
    return sum(1 for line in text.splitlines() if line.strip())


def release_status(local: Path, release: Optional[Path]) -> str:
    """Describe how a local file relates to its published counterpart."""
    if release is None:
        return "not in the MELT repository"
    if not release.exists():
        return "release file not in cache"
    return "identical" if sha256(local) == sha256(release) else "DIFFERENT"


def _row(item: TrackedFile, data_dir: Path, cache: Optional[Path]) -> List[str]:
    path = data_dir / item.local
    release = cache / item.release if cache is not None and item.release else None
    status = release_status(path, release) if cache is not None else "not checked"
    count = str(correspondence_count(path)) if item.is_reference else ""
    return [item.dataset_id, item.local, sha256(path), str(path.stat().st_size), count, status]


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def build_report(data_dir: Path, cache: Optional[Path]) -> List[str]:
    """
    Build the provenance tables.

    Parameters
    ----------
    data_dir : Path
        Dataset root.
    cache : Optional[Path]
        Local MELT track repository root, or ``None`` to skip comparison.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    header = ["Dataset", "File (under data/)", "SHA-256", "Bytes", "Correspondences", "Release"]
    rows = [_row(item, data_dir, cache) for item in tracked_files(data_dir)]
    return ["# Dataset provenance", "", *_table(header, rows), ""]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record dataset checksums and release identity.")
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--release_cache", default=None, help="Local oaei.webdatacommons.org copy.")
    parser.add_argument("--output", default="artifacts/issue-137/dataset_provenance.md", help="Output path.")
    return parser


def main() -> None:
    """Write the provenance report."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    cache = Path(args.release_cache).expanduser() if args.release_cache else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(build_report(Path(args.data_dir), cache)), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
