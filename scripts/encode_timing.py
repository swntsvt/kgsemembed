"""Wall-clock cost of every condition on one column, and of stage 1 (#181).

One model per process.  For every compared condition of ``--model`` registered
on ``--dataset`` (C19, the sanity check, left out), this re-runs the pipeline's
stage 2 exactly as ``run_experiment._process_pair`` does and times its parts:
verbalisation (once), source and candidate encoding (an untimed warm-up, then
``--repeats`` timed passes over every text), and ranking plus evaluation.  The
run must reproduce the stored result's metrics exactly, and every repeat must
return bitwise-identical embeddings, or nothing is written.  ``--stage1``
instead times the n-gram scoring over all targets (B0's whole cost and every
condition's shared first stage), checked against the stored candidate order.

Timings are descriptive and specific to this implementation and machine
(MPS, padding to a multiple of 64 tokens, batch reduction for long texts).
They are written to ``--output_dir`` only; nothing under ``data/`` changes.
The tree must be clean under ``src/`` and ``scripts/``.  Run it with nothing
else on the machine.

Usage
-----
python scripts/encode_timing.py --stage1 --dataset D1
for m in M1 M2 M3 M4 M5; do
    python scripts/encode_timing.py --model $m --dataset D1
done
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoded_input_size import compared_conditions  # noqa: E402
from kgsemembed.candidates import load_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair, load_dataset  # noqa: E402
from kgsemembed.embeddings import EmbeddingEncoder  # noqa: E402
from kgsemembed.pipeline.conditions import ExperimentCondition  # noqa: E402
from kgsemembed.pipeline.ngram_baseline import (  # noqa: E402
    count_order_mismatches,
    scored_candidates,
)
from kgsemembed.pipeline.run_experiment import (  # noqa: E402
    _build_ranked_lists,
    _evaluate_pair,
    _library_versions,
    _load_checked_model,
    _release_model,
    _seed_random_state,
    _unique_candidate_uris,
    _verbalise_uris,
)
from kgsemembed.verbalisation.registry import build_verbaliser  # noqa: E402

CHECKED_METRICS = ("mrr", "hits_at_1", "f1", "precision", "recall", "threshold")
WARMUP_TEXTS = 256
_CODE_PATHS = ("src", "scripts")


class TimingCheckError(RuntimeError):
    """Raised when a timed run does not reproduce the stored result."""


def timed(function: Callable[[], object]) -> Tuple[object, float]:
    """
    Call ``function`` and measure its wall-clock duration.

    Parameters
    ----------
    function : Callable[[], object]
        Zero-argument callable.

    Returns
    -------
    Tuple[object, float]
        The return value and the elapsed seconds.
    """
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def code_state() -> Dict[str, object]:
    """
    Return the commit and refuse uncommitted changes to the measured code.

    Returns
    -------
    Dict[str, object]
        ``git_commit``.

    Raises
    ------
    TimingCheckError
        If ``src/`` or ``scripts/`` has uncommitted or untracked changes.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *_CODE_PATHS], capture_output=True, text=True, check=True
    ).stdout.strip()
    if status:
        raise TimingCheckError(f"uncommitted changes under {_CODE_PATHS}:\n{status}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return {"git_commit": commit.stdout.strip()}


def machine() -> Dict[str, object]:
    """
    Describe the machine and library versions the timings come from.

    Returns
    -------
    Dict[str, object]
        Chip, memory, OS and library versions.
    """

    def sysctl(key: str) -> str:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(sysctl("hw.memsize") or 0),
        "platform": platform.platform(),
        "versions": _library_versions(),
    }


def verbalise_pair(
    pair: AlignmentPair, condition: ExperimentCondition, candidates: Dict[str, List[str]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Verbalise sources and unique candidates as ``_process_pair`` does.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair.
    condition : ExperimentCondition
        Condition supplying strategy and model key.
    candidates : Dict[str, List[str]]
        Stored stage-1 candidate lists.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Source texts, candidate URIs and candidate texts.
    """
    verbaliser = build_verbaliser(condition.strategy_name, condition.model_key)
    sources = _verbalise_uris(
        verbaliser, pair.source_graph, pair.source_entities, pair.entity_type, "source verbalisation"
    )
    uris = _unique_candidate_uris(pair, candidates)
    texts = _verbalise_uris(verbaliser, pair.target_graph, uris, pair.entity_type, "candidate verbalisation")
    return sources, uris, texts


def timed_encodes(
    encoder: EmbeddingEncoder, texts: List[str], role: str, repeats: int
) -> Tuple[np.ndarray, List[float]]:
    """
    Encode ``texts`` after a warm-up, ``repeats`` times, and time each pass.

    Parameters
    ----------
    encoder : EmbeddingEncoder
        Encoder of the loaded model.
    texts : List[str]
        Texts of one role.
    role : str
        ``"source"`` or ``"candidate"``.
    repeats : int
        Timed passes.

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        The embeddings of the first pass and the seconds of every pass.

    Raises
    ------
    TimingCheckError
        If a pass returns embeddings that differ from the first.
    """
    encoder.encode_batch(texts[:WARMUP_TEXTS], role=role, show_progress=False)
    first, seconds = None, []
    for _ in range(repeats):
        embeddings, elapsed = timed(lambda: encoder.encode_batch(texts, role=role, show_progress=True))
        if first is not None and not np.array_equal(first, embeddings):
            raise TimingCheckError(f"{role} embeddings differ between repeats")
        first = embeddings if first is None else first
        seconds.append(elapsed)
    return first, seconds


def check_reproduces(metrics: Dict[str, object], stored_path: Path) -> None:
    """
    Refuse a timed run whose metrics differ from the stored result.

    Parameters
    ----------
    metrics : Dict[str, object]
        Metrics of the timed run.
    stored_path : Path
        The stored ``*_results.json`` of the same condition and pair.

    Raises
    ------
    TimingCheckError
        If any checked metric differs.
    """
    stored = json.loads(stored_path.read_text())["metrics"]
    differing = {key: (metrics[key], stored[key]) for key in CHECKED_METRICS if metrics[key] != stored[key]}
    if differing:
        raise TimingCheckError(f"{stored_path}: metrics differ (run, stored): {differing}")


def time_condition(
    condition: ExperimentCondition, pair: AlignmentPair, encoder: EmbeddingEncoder, args: argparse.Namespace
) -> Dict[str, object]:
    """
    Time one condition on one pair and check that it reproduces its result.

    Parameters
    ----------
    condition : ExperimentCondition
        Condition to time.
    pair : AlignmentPair
        Alignment pair.
    encoder : EmbeddingEncoder
        Encoder of the condition's loaded model.
    args : argparse.Namespace
        ``data_dir``, ``results_dir`` and ``repeats``.

    Returns
    -------
    Dict[str, object]
        Text counts and seconds per stage, the stored metrics reproduced.
    """
    _seed_random_state()
    candidates = load_candidates(pair.dataset_id, pair.pair_name, args.data_dir)
    (sources, uris, texts), verbalise_seconds = timed(lambda: verbalise_pair(pair, condition, candidates))
    source_embeddings, source_seconds = timed_encodes(encoder, sources, "source", args.repeats)
    candidate_embeddings, candidate_seconds = timed_encodes(encoder, texts, "candidate", args.repeats)
    index = {uri: position for position, uri in enumerate(uris)}
    metrics, rank_seconds = timed(lambda: _evaluate_pair(pair, _build_ranked_lists(
        pair, source_embeddings, candidates, index, candidate_embeddings)))
    stored = Path(args.results_dir) / condition.condition_id / pair.dataset_id / f"{pair.pair_name}_results.json"
    check_reproduces(metrics, stored)
    return {
        "condition_id": condition.condition_id, "strategy": condition.strategy_name,
        "model_key": condition.model_key, "column": pair.dataset_id, "pair_name": pair.pair_name,
        "n_source_texts": len(sources), "n_candidate_texts": len(texts),
        "verbalise_seconds": verbalise_seconds, "source_encode_seconds": source_seconds,
        "candidate_encode_seconds": candidate_seconds, "rank_evaluate_seconds": rank_seconds,
        "metrics": {key: metrics[key] for key in CHECKED_METRICS}, "reproduces_stored": True,
    }


def model_record(model: object, model_info: Dict[str, object], load_seconds: float) -> Dict[str, object]:
    """
    Describe the loaded model: provenance, size and load time.

    Parameters
    ----------
    model : object
        Loaded ``SentenceTransformer``.
    model_info : Dict[str, object]
        Provenance from the loader, device check included.
    load_seconds : float
        Seconds the load took (from the local cache).

    Returns
    -------
    Dict[str, object]
        ``model_info`` plus parameter count, embedding dimension and load time.
    """
    return {
        **model_info,
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "embedding_dimension": int(model.get_sentence_embedding_dimension()),
        "load_seconds": load_seconds,
    }


def write_record(record: Dict[str, object], output_dir: Path, name: str) -> None:
    """
    Write one timing record as JSON.

    Parameters
    ----------
    record : Dict[str, object]
        The record.
    output_dir : Path
        Destination directory.
    name : str
        File stem.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")


def load_pairs(dataset_id: str, data_dir: str) -> Tuple[List[AlignmentPair], float]:
    """
    Load a dataset's pairs and time the load.

    Parameters
    ----------
    dataset_id : str
        Loader key.
    data_dir : str
        Dataset root directory.

    Returns
    -------
    Tuple[List[AlignmentPair], float]
        The pairs and the seconds the load took.
    """
    return timed(lambda: load_dataset(dataset_id, data_dir))


def run_model(args: argparse.Namespace, context: Dict[str, object]) -> None:
    """
    Time every compared condition of ``args.model`` on ``args.dataset``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    context : Dict[str, object]
        Code state and machine description, written into every record.
    """
    conditions = [c for c in compared_conditions(args.dataset) if c.model_key == args.model]
    pairs, load_seconds = load_pairs(args.dataset, args.data_dir)
    (model, model_info), model_seconds = timed(lambda: _load_checked_model(args.model, None))
    encoder = EmbeddingEncoder(args.model, model)
    model_entry = model_record(model, model_info, model_seconds)
    for condition in conditions:
        for pair in pairs:
            record = time_condition(condition, pair, encoder, args)
            record.update(context, model=model_entry, dataset_load_seconds=load_seconds, repeats=args.repeats)
            write_record(record, Path(args.output_dir), f"{condition.condition_id}_{pair.dataset_id}_{pair.pair_name}")
    _release_model(model)


def run_stage1(args: argparse.Namespace, context: Dict[str, object]) -> None:
    """
    Time the stage-1 n-gram scoring of every pair and check its order.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    context : Dict[str, object]
        Code state and machine description, written into every record.

    Raises
    ------
    TimingCheckError
        If the recomputed order differs from the stored candidates.
    """
    pairs, load_seconds = load_pairs(args.dataset, args.data_dir)
    for pair in pairs:
        runs = [timed(lambda: scored_candidates(pair)) for _ in range(args.repeats)]
        stored = load_candidates(pair.dataset_id, pair.pair_name, args.data_dir)
        if any(count_order_mismatches(scored, stored) for scored, _ in runs):
            raise TimingCheckError(f"{pair.pair_name}: stage-1 order differs from the stored candidates")
        record = {"column": pair.dataset_id, "pair_name": pair.pair_name, "n_sources": len(pair.source_entities),
                  "score_seconds": [seconds for _, seconds in runs], "dataset_load_seconds": load_seconds,
                  "repeats": args.repeats, **context}
        write_record(record, Path(args.output_dir), f"stage1_{pair.dataset_id}_{pair.pair_name}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        ``model`` or ``stage1``, ``dataset``, directories and ``repeats``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=["M1", "M2", "M3", "M4", "M5"])
    mode.add_argument("--stage1", action="store_true")
    parser.add_argument("--dataset", default="D1", choices=["D1", "D2", "D3", "D4", "D5"])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--results_dir", default="data/results/")
    parser.add_argument("--output_dir", default="artifacts/issue-181/timing/")
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Time stage 1 or one model's conditions and write a record per pair."""
    np.random.seed(42)
    random.seed(42)
    args = parse_args()
    context = {**code_state(), "machine": machine()}
    (run_stage1 if args.stage1 else run_model)(args, context)


if __name__ == "__main__":
    main()
