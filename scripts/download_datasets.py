"""Check on-disk availability of the five Phase 2 datasets.

Every dataset used by :func:`kgsemembed.datasets.loader.load_dataset` requires
manual download steps, so this helper never fetches anything: it reports which
required files are present and prints the manual retrieval instructions.

``--check`` inspects file existence only; no dataset is parsed or loaded into
memory. Exit code is ``0`` when all five datasets are complete and ``1`` when
any dataset is partial or missing.

Usage
-----
python scripts/download_datasets.py --data_dir data/ --check
python scripts/download_datasets.py --data_dir data/ --instructions
python scripts/download_datasets.py --data_dir data/ --instructions D5
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

_STATUS_OK = "OK"
_STATUS_PARTIAL = "PARTIAL"
_STATUS_MISSING = "MISSING"

_ALL = "ALL"
_MAX_LISTED_PAIRS = 2
_WRAP_WIDTH = 78
_LABEL_INDENT = " " * 10

_PAIR_FILES = ("source.rdf", "target.rdf", "reference.rdf")

_D3_PAIRS = (
    "cmt-conference",
    "cmt-confof",
    "cmt-edas",
    "cmt-ekaw",
    "cmt-iasted",
    "cmt-sigkdd",
    "conference-confof",
    "conference-edas",
    "conference-ekaw",
    "conference-iasted",
    "conference-sigkdd",
    "confof-edas",
    "confof-ekaw",
    "confof-iasted",
    "confof-sigkdd",
    "edas-ekaw",
    "edas-iasted",
    "edas-sigkdd",
    "ekaw-iasted",
    "ekaw-sigkdd",
    "iasted-sigkdd",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Required layout and manual download instructions for one dataset."""

    dataset_id: str
    title: str
    root: str
    required: tuple[str, ...]
    source_url: str
    files: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DatasetStatus:
    """Availability of one dataset on disk."""

    dataset_id: str
    status: str
    missing: str


_D3_REQUIRED = tuple(f"{pair}/{name}" for pair in _D3_PAIRS for name in _PAIR_FILES)

_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="D1",
        title="SNOMED-FMA Body (Bio-ML, OAEI)",
        root="d1_snomed_fma",
        required=_PAIR_FILES,
        source_url="https://github.com/KRR-Oxford/DeepOnto",
        files="source.rdf, target.rdf, reference.rdf",
        notes=(
            "Use the Bio-ML equivalence dataset, unsupervised split.",
            "Copy source.rdf, target.rdf, reference.rdf from the "
            "snomed.body-fma.body/ subdirectory.",
        ),
    ),
    DatasetSpec(
        dataset_id="D2",
        title="Anatomy MA-NCI (OAEI Anatomy Track)",
        root="d2_anatomy",
        required=_PAIR_FILES,
        source_url="https://oaei.ontologymatching.org/2025/anatomy/",
        files="source.rdf (mouse anatomy), target.rdf (NCI human), reference.rdf",
        notes=(
            "Copy source.rdf, target.rdf, reference.rdf from the "
            "anatomy_track/anatomy_track-default/mouse-human-suite/ folder.",
        ),
    ),
    DatasetSpec(
        dataset_id="D3",
        title="Conference (OAEI Conference Track)",
        root="d3_conference",
        required=_D3_REQUIRED,
        source_url="https://oaei.ontologymatching.org/2025/conference/",
        files="21 pair subdirectories, each with source.rdf, target.rdf, reference.rdf",
        notes=(
            "Copy all pair subdirectories from conference/conference-v1/. "
            "Do not copy parameters.rdf — it is not used by the loader.",
            "Expected pairs (all 21 combinations of the 7 ontologies): "
            + ", ".join(_D3_PAIRS)
            + ".",
        ),
    ),
    DatasetSpec(
        dataset_id="D4",
        title="memoryalpha-stexpanded (OAEI KG Track)",
        root="d4_kgtrack",
        required=(
            "ontologies/memoryalpha.rdf",
            "ontologies/stexpanded.rdf",
            "references/memoryalpha-stexpanded.rdf",
        ),
        source_url="https://oaei.ontologymatching.org/2025/knowledgegraph/",
        files=(
            "ontologies/memoryalpha.rdf, ontologies/stexpanded.rdf, "
            "references/memoryalpha-stexpanded.rdf"
        ),
        notes=(
            "Copy ontologies/ and references/ subfolders from knowledgegraph/v4/. "
            'The ontology files are RDF/XML format (.rdf extension); parse with format="xml".',
            "Other .rdf files in those folders (starwars, marvel, etc.) belong to "
            "other KG track pairs and are not used.",
        ),
    ),
    DatasetSpec(
        dataset_id="D5",
        title="DBpedia-Wikidata 15K EN (OpenEA benchmark)",
        root="d5_openea/D_W_15K_V2",
        required=(
            "rel_triples_1",
            "rel_triples_2",
            "attr_triples_1",
            "attr_triples_2",
            "ent_links",
            "721_5fold/1/train_links",
            "721_5fold/1/valid_links",
            "721_5fold/1/test_links",
        ),
        source_url="https://github.com/nju-websoft/OpenEA",
        files=(
            "rel_triples_1, rel_triples_2, attr_triples_1, attr_triples_2, "
            "ent_links, 721_5fold/1/train_links, 721_5fold/1/valid_links, "
            "721_5fold/1/test_links"
        ),
        notes=(
            "Download OpenEA_dataset_v2.0.zip from the Figshare/Dropbox/Baidu links "
            "in the README. The target subdirectory inside the zip is D_W_15K_V2/ "
            "(DBpedia-Wikidata, 15K entities, version 2). Copy it to "
            "data/d5_openea/D_W_15K_V2/.",
            "WARNING: Do NOT use DBP2.0 (also from nju-websoft). DBP2.0 is "
            "multilingual (zh/ja/fr+en), has no attr_triples, and uses a splits/ "
            "folder structure — it is a different benchmark.",
        ),
    ),
)

_SPECS_BY_ID = {spec.dataset_id: spec for spec in _SPECS}


def _missing_paths(root: Path, required: tuple[str, ...]) -> list[str]:
    return [relative for relative in required if not (root / relative).exists()]


def _format_d3_missing(missing: list[str]) -> str:
    pairs = sorted({relative.split("/", 1)[0] for relative in missing})
    if len(pairs) > _MAX_LISTED_PAIRS:
        return f"{len(pairs)} pair directories missing"
    return ", ".join(f"{pair}/" for pair in pairs)


def _format_missing(dataset_id: str, missing: list[str]) -> str:
    if not missing:
        return "-"
    if dataset_id == "D3":
        return _format_d3_missing(missing)
    return ", ".join(missing)


def check_dataset(spec: DatasetSpec, data_dir: Path) -> DatasetStatus:
    """
    Report whether every required file of one dataset is present on disk.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset whose required layout is checked.
    data_dir : Path
        Dataset root directory containing the per-dataset subdirectories.

    Returns
    -------
    DatasetStatus
        Status ``OK``, ``PARTIAL``, or ``MISSING`` with a missing-file summary.
    """
    root = data_dir / spec.root
    missing = _missing_paths(root, spec.required)
    if not root.is_dir():
        status = _STATUS_MISSING
    else:
        status = _STATUS_PARTIAL if missing else _STATUS_OK
    return DatasetStatus(spec.dataset_id, status, _format_missing(spec.dataset_id, missing))


def check_all(data_dir: Path) -> list[DatasetStatus]:
    """
    Report the availability of all five Phase 2 datasets.

    Parameters
    ----------
    data_dir : Path
        Dataset root directory.

    Returns
    -------
    list[DatasetStatus]
        One status entry per dataset, ordered D1 to D5.
    """
    return [check_dataset(spec, data_dir) for spec in _SPECS]


def print_status_table(statuses: list[DatasetStatus]) -> None:
    """
    Print the ``--check`` status table to standard output.

    Parameters
    ----------
    statuses : list[DatasetStatus]
        Status entries to render, in display order.
    """
    print(f"{'Dataset':<9}{'Status':<10}Missing")
    print(f"{'-' * 7:<9}{'-' * 8:<10}{'-' * 50}")
    for entry in statuses:
        print(f"{entry.dataset_id:<9}{entry.status:<10}{entry.missing}")


def _wrap(text: str, label: str) -> str:
    return textwrap.fill(
        text,
        width=_WRAP_WIDTH,
        initial_indent=label,
        subsequent_indent=_LABEL_INDENT,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _note_lines(notes: tuple[str, ...]) -> list[str]:
    return [
        _wrap(note, "Notes:    " if index == 0 else _LABEL_INDENT)
        for index, note in enumerate(notes)
    ]


def format_instructions(spec: DatasetSpec, data_dir: Path) -> str:
    """
    Render the manual download instructions for one dataset.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset to describe.
    data_dir : Path
        Dataset root directory, used to render the target path.

    Returns
    -------
    str
        Multi-line instruction block without a trailing newline.
    """
    lines = [
        f"Dataset: {spec.dataset_id} — {spec.title}",
        f"Source:   {spec.source_url}",
        f"Target:   {(data_dir / spec.root).as_posix()}/",
        _wrap(spec.files, "Files:    "),
    ]
    lines.extend(_note_lines(spec.notes))
    return "\n".join(lines)


def print_instructions(specs: tuple[DatasetSpec, ...], data_dir: Path) -> None:
    """
    Print manual download instructions for each dataset in ``specs``.

    Parameters
    ----------
    specs : tuple[DatasetSpec, ...]
        Datasets to describe, in display order.
    data_dir : Path
        Dataset root directory, used to render target paths.
    """
    for spec in specs:
        print(format_instructions(spec, data_dir))
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 2 dataset availability and print manual download steps."
    )
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--check", action="store_true", help="Print the status table.")
    parser.add_argument(
        "--instructions",
        nargs="?",
        const=_ALL,
        choices=(_ALL, *_SPECS_BY_ID),
        help="Print manual download instructions, optionally for one dataset only.",
    )
    args = parser.parse_args()
    if not args.check and args.instructions is None:
        parser.error("one of --check or --instructions is required")
    return args


def _selected_specs(selection: str) -> tuple[DatasetSpec, ...]:
    if selection == _ALL:
        return _SPECS
    return (_SPECS_BY_ID[selection],)


def main() -> int:
    """
    Run the requested checks and instruction output.

    Returns
    -------
    int
        ``0`` when no check ran or every dataset is complete, ``1`` otherwise.
    """
    args = _parse_args()
    data_dir = Path(args.data_dir)
    if args.instructions is not None:
        print_instructions(_selected_specs(args.instructions), data_dir)
    if not args.check:
        return 0
    statuses = check_all(data_dir)
    print_status_table(statuses)
    return int(any(entry.status != _STATUS_OK for entry in statuses))


if __name__ == "__main__":
    sys.exit(main())
