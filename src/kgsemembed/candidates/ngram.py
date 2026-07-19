"""Loader for pre-generated character n-gram candidate lists.

Stage-1 lexical filtering is executed offline and its per-source candidate
lists are persisted as JSON under
``{data_dir}/candidates/{dataset_id}/{pair_name}_candidates.json``.  This module
reads those files back for the Phase 2 embedding pipeline; it performs no
candidate generation itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from kgsemembed.utils.errors import DataError

CandidateMap = Dict[str, List[str]]

_CANDIDATE_ROOT = "candidates"


def _candidates_path(dataset_id: str, pair_name: str, data_dir: str | Path) -> Path:
    return (
        Path(data_dir)
        / _CANDIDATE_ROOT
        / dataset_id
        / f"{pair_name}_candidates.json"
    )


def _parse_candidate_map(payload: dict) -> CandidateMap:
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise DataError("Candidate file must contain a 'candidates' object.")
    return {str(source): [str(target) for target in targets] for source, targets in candidates.items()}


def load_candidates(
    dataset_id: str, pair_name: str, data_dir: str | Path = "data/"
) -> CandidateMap:
    """
    Load pre-generated candidate target URIs for a single alignment pair.

    Parameters
    ----------
    dataset_id : str
        OAEI track identifier, e.g. ``"D1"``.
    pair_name : str
        Ontology pair name, e.g. ``"d1_snomed_fma"``.
    data_dir : str | Path
        Dataset root directory (default ``"data/"``).

    Returns
    -------
    CandidateMap
        Mapping from each source URI to its ordered candidate target URIs.

    Raises
    ------
    DataError
        If the candidate file is missing or is not valid candidate JSON.
    """
    path = _candidates_path(dataset_id, pair_name, data_dir)
    if not path.exists():
        raise DataError(f"Candidate file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Malformed candidate JSON at {path}: {exc}") from exc
    return _parse_candidate_map(payload)
