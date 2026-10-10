"""Pipeline entrypoint for Phase 2 experiments.

Two entry points coexist in this module:

* ``run_experiment`` — the Hydra-driven candidate-generation scaffold.
* ``run_condition`` / ``run_all_conditions`` — the Phase 2 experiment runner that
  orchestrates the end-to-end embedding pipeline for the registered
  experimental conditions and is exposed through the ``argparse`` command line.

The runner is a pure orchestration layer: it loads datasets, candidates,
verbalisers, and embedding models through existing components and delegates all
scoring and evaluation to them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig
from rdflib import Graph, URIRef
from tqdm import tqdm

from kgsemembed.candidates import build_candidate_table, generate_candidates, load_candidates
from kgsemembed.datasets import AlignmentPair, load_dataset, load_oaei_dataset
from kgsemembed.datasets.loader import _SPLIT_SEED, _resolve_entity_type
from kgsemembed.embeddings import EmbeddingEncoder, MODEL_REGISTRY, load_sentence_transformer
from kgsemembed.evaluation import (
    EntityPair,
    RankedList,
    EVALUATION_PROTOCOL_VERSION,
    compute_all_metrics,
    top_ranked_pairs,
    tune_threshold,
)
from kgsemembed.evaluation.population import (
    EvaluationPopulations,
    build_populations,
    judgeable_ranked_lists,
    restrict_ranked_lists,
)
from kgsemembed.pipeline.conditions import (
    EXPERIMENT_CONDITIONS,
    ExperimentCondition,
    get_condition,
)
from kgsemembed.utils.errors import DataError
from kgsemembed.utils.logging import RunContext, get_logger, init_logging
from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS, strategy_uses_ppas
from kgsemembed.verbalisation.registry import build_verbaliser

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

_LOGGER = logging.getLogger("kgsemembed.pipeline.runner")

_MIXED_ENTITY_TYPE = "mixed"
_ENTITY_TYPES = ("class", "predicate", "instance")
_DEFAULT_THRESHOLD = 0.5
_RANDOM_SEED = 42
_FAILURE_KEY = "n_failed_pairs"
_PER_ENTITY_TYPE_KEY = "per_entity_type"
_POPULATION_KEY = "population"
_COUNT_KEYS = ("tp", "fp", "fn")
_VAL = "val"
_TEST = "test"
_REF_COUNT_KEY = "n_refs"
_MIN_BUCKET_REFS = 3
_METRIC_KEYS = (
    "f1",
    "precision",
    "recall",
    "threshold",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
)


def run_experiment(cfg: DictConfig) -> int:
    logger, run_id = init_logging(level=str(cfg.experiment.log_level))
    context = RunContext(
        run_id=run_id,
        experiment_name=str(cfg.experiment.name),
        stage="ingest",
        dataset=str(cfg.dataset.name),
        model=str(cfg.model.name),
        device=str(cfg.model.device),
    )
    log = get_logger("kgsemembed.pipeline", context)
    logger.info("Pipeline bootstrap complete", extra={"run_id": run_id, "stage": "init"})
    log.info(
        "Resolved config: experiment=%s model=%s dataset=%s verbalisation=%s candidates=%s(n=%s,metric=%s,k=%s) device=%s ks=%s",
        cfg.experiment.name,
        cfg.model.name,
        cfg.dataset.name,
        cfg.verbalisation.strategy,
        cfg.candidates.method,
        cfg.candidates.n,
        cfg.candidates.metric,
        cfg.candidates.top_k,
        cfg.model.device,
        list(cfg.experiment.evaluation_ks),
    )

    try:
        bundle = load_oaei_dataset(
            source_path=cfg.dataset.source_rdf,
            target_path=cfg.dataset.target_rdf,
            alignment_path=cfg.dataset.alignment_rdf,
        )
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Dataset loading failed: {exc}") from exc

    log.info(
        "Loaded source graph: format=%s entities=%d triples=%d",
        bundle.source.format,
        len(bundle.source.entities),
        len(bundle.source.triples),
    )
    log.info(
        "Loaded target graph: format=%s entities=%d triples=%d",
        bundle.target.format,
        len(bundle.target.entities),
        len(bundle.target.triples),
    )
    log.info(
        "Loaded alignment graph: format=%s entities=%d triples=%d",
        bundle.alignment.format,
        len(bundle.alignment.entities),
        len(bundle.alignment.triples),
    )

    t0 = time.perf_counter()
    candidates = generate_candidates(
        source_labels=bundle.source.labels,
        target_labels=bundle.target.labels,
        n=int(cfg.candidates.n),
        metric=str(cfg.candidates.metric),
        top_k=int(cfg.candidates.top_k),
        strip_punctuation=bool(cfg.candidates.strip_punctuation),
    )
    elapsed = time.perf_counter() - t0
    table = build_candidate_table(candidates)

    log.info(
        "Generated candidates: source=%d target=%d pairs=%d metric=%s n=%d top_k=%d runtime_sec=%.4f",
        len(bundle.source.labels),
        len(bundle.target.labels),
        len(candidates),
        cfg.candidates.metric,
        int(cfg.candidates.n),
        int(cfg.candidates.top_k),
        elapsed,
    )

    if bool(cfg.candidates.persist):
        out_dir = Path(str(cfg.experiment.output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(cfg.candidates.output_file)
        table.to_csv(out_path, index=False)
        log.info("Persisted candidate pairs to %s", out_path)

    return 0


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    raise SystemExit(run_experiment(cfg))


def _result_path(
    results_dir: str | Path, condition_id: str, dataset_id: str, pair_name: str
) -> Path:
    return Path(results_dir) / condition_id / dataset_id / f"{pair_name}_results.json"


def _entity_type_for(pair_entity_type: str, graph: Graph, uri: str) -> str:
    if pair_entity_type == _MIXED_ENTITY_TYPE:
        return _resolve_entity_type(graph, uri)
    return pair_entity_type


def _verbalise_uris(
    verbaliser: VerbaliserBase,
    graph: Graph,
    uris: List[str],
    entity_type: str,
    desc: str,
) -> List[str]:
    return [
        verbaliser.verbalise(
            graph, URIRef(uri), _entity_type_for(entity_type, graph, uri)
        )
        for uri in tqdm(uris, desc=desc)
    ]


def _rank_candidates(
    source_uri: str,
    source_embedding: np.ndarray,
    candidate_uris: List[str],
    candidate_index: Dict[str, int],
    candidate_embeddings: np.ndarray,
) -> RankedList:
    scored = [
        (source_uri, uri, float(np.dot(source_embedding, candidate_embeddings[candidate_index[uri]])))
        for uri in candidate_uris
    ]
    scored.sort(key=lambda triple: (-triple[2], triple[1]))
    return scored


def _build_ranked_lists(
    pair: AlignmentPair,
    source_embeddings: np.ndarray,
    candidates: Dict[str, List[str]],
    candidate_index: Dict[str, int],
    candidate_embeddings: np.ndarray,
) -> List[RankedList]:
    ranked_lists: List[RankedList] = []
    for position, source_uri in enumerate(pair.source_entities):
        candidate_uris = candidates.get(source_uri, [])
        if not candidate_uris:
            continue
        ranked_lists.append(
            _rank_candidates(
                source_uri,
                source_embeddings[position],
                candidate_uris,
                candidate_index,
                candidate_embeddings,
            )
        )
    return ranked_lists


def _unique_candidate_uris(
    pair: AlignmentPair, candidates: Dict[str, List[str]]
) -> List[str]:
    return sorted(
        {uri for source in pair.source_entities for uri in candidates.get(source, [])}
    )


def _resolve_threshold(
    val_ranked_lists: List[RankedList], val_refs: List[EntityPair]
) -> float:
    """
    Tune the decision threshold on the validation population only.

    The caller restricts ``val_ranked_lists`` to the validation population, so
    no test source is observed.  Tuning scores the same top-1 predictions the
    test metrics are computed from.

    Parameters
    ----------
    val_ranked_lists : List[RankedList]
        Ranked lists of the validation population's sources.
    val_refs : List[EntityPair]
        Validation ``(source, target)`` reference pairs.

    Returns
    -------
    float
        Tuned threshold, or ``0.5`` when there are no validation references.
    """
    if not val_refs:
        _LOGGER.warning(
            "Empty validation references; using default threshold %.2f",
            _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    return tune_threshold(top_ranked_pairs(val_ranked_lists), val_refs)["best_threshold"]


def _split_refs_by_entity_type(
    refs: List[EntityPair], graph: Graph
) -> Dict[str, List[EntityPair]]:
    """
    Group reference pairs by the entity type of their source URI.

    Parameters
    ----------
    refs : List[EntityPair]
        Reference ``(source, target)`` pairs of a mixed-type alignment pair.
    graph : Graph
        Source graph supplying the ``rdf:type`` triples of each source URI.

    Returns
    -------
    Dict[str, List[EntityPair]]
        Mapping from ``"class"``, ``"predicate"``, or ``"instance"`` to the
        references of that type, in that fixed order.  Empty buckets are
        omitted, so a purely class-typed input yields a single key.
    """
    buckets: Dict[str, List[EntityPair]] = {}
    for source_uri, target_uri in refs:
        entity_type = _resolve_entity_type(graph, source_uri)
        buckets.setdefault(entity_type, []).append((source_uri, target_uri))
    return {key: buckets[key] for key in _ENTITY_TYPES if key in buckets}


def _split_ranked_lists_by_entity_type(
    ranked_lists: List[RankedList], graph: Graph
) -> Dict[str, List[RankedList]]:
    """
    Group ranked candidate lists by the entity type of their source URI.

    Every ranked source of the population is assigned to a bucket, not only the
    ones carrying a reference, so a bucket's precision is penalised by its own
    false positives exactly as the pair-level metrics are.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.
    graph : Graph
        Source graph supplying the ``rdf:type`` triples of each source URI.

    Returns
    -------
    Dict[str, List[RankedList]]
        Mapping from entity type to the ranked lists of its source entities.
    """
    buckets: Dict[str, List[RankedList]] = {}
    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        entity_type = _resolve_entity_type(graph, ranked_list[0][0])
        buckets.setdefault(entity_type, []).append(ranked_list)
    return buckets


def _has_enough_refs(entity_type: str, refs: List[EntityPair]) -> bool:
    """
    Report whether a bucket holds enough references to be evaluated.

    A bucket of one or two references cannot support threshold tuning, so it is
    dropped with a warning rather than reported as a near-random metric.

    Parameters
    ----------
    entity_type : str
        Entity type of the bucket, used only for the warning message.
    refs : List[EntityPair]
        Test references falling in the bucket.

    Returns
    -------
    bool
        ``True`` when the bucket carries at least three references.
    """
    if len(refs) >= _MIN_BUCKET_REFS:
        return True
    _LOGGER.warning(
        "Skipping %s entity-type metrics: %d reference pairs, minimum is %d",
        entity_type,
        len(refs),
        _MIN_BUCKET_REFS,
    )
    return False


def _bucket_threshold(
    ranked_lists: List[RankedList], val_refs: List[EntityPair], fallback: float
) -> float:
    """
    Tune a bucket's threshold on its own validation references.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        Ranked candidate lists of the bucket's validation-population sources.
    val_refs : List[EntityPair]
        Validation references falling in the bucket.
    fallback : float
        Threshold tuned for the alignment pair as a whole, used when the bucket
        holds no validation reference of its own.

    Returns
    -------
    float
        Threshold to evaluate the bucket with.
    """
    if not val_refs:
        return fallback
    return _resolve_threshold(ranked_lists, val_refs)


def _bucket_metrics(
    ranked_lists: List[RankedList], refs: List[EntityPair], threshold: float
) -> Dict[str, float]:
    """Evaluate one entity-type bucket and record how many references it held."""
    metrics = compute_all_metrics(ranked_lists, refs, threshold=threshold)
    metrics[_REF_COUNT_KEY] = len(refs)
    return metrics


def _per_entity_type_metrics(
    pair: AlignmentPair, split_lists: Dict[str, List[RankedList]], threshold: float
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a mixed-type alignment pair separately per source entity type.

    Only pairs declared ``"mixed"`` are split; every other pair already reports
    a single entity type at the top level and yields an empty mapping.  Each
    bucket is tuned on its own validation references so that a class threshold
    cannot be imposed on predicates, and falls back to the pair-level threshold
    when the bucket has no validation reference of its own.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair being evaluated.
    split_lists : Dict[str, List[RankedList]]
        Ranked lists of the ``"val"`` and ``"test"`` populations.
    threshold : float
        Threshold tuned for the alignment pair as a whole.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping from entity type to its metrics extended with ``"n_refs"``.
    """
    if pair.entity_type != _MIXED_ENTITY_TYPE:
        return {}
    graph = pair.source_graph
    val_refs = _split_refs_by_entity_type(pair.val_refs, graph)
    val_lists = _split_ranked_lists_by_entity_type(split_lists[_VAL], graph)
    test_lists = _split_ranked_lists_by_entity_type(split_lists[_TEST], graph)
    metrics: Dict[str, Dict[str, float]] = {}
    for entity_type, refs in _split_refs_by_entity_type(pair.test_refs, graph).items():
        if not _has_enough_refs(entity_type, refs):
            continue
        val_bucket = val_lists.get(entity_type, [])
        tuned = _bucket_threshold(val_bucket, val_refs.get(entity_type, []), threshold)
        metrics[entity_type] = _bucket_metrics(test_lists.get(entity_type, []), refs, tuned)
    return metrics


def _split_ranked_lists(
    ranked_lists: List[RankedList], populations: EvaluationPopulations
) -> Dict[str, List[RankedList]]:
    """
    Partition ranked lists into the validation and test populations.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        Ranked candidate lists of every source of the pair.
    populations : EvaluationPopulations
        Source assignment returned by :func:`build_populations`.

    Returns
    -------
    Dict[str, List[RankedList]]
        Mapping from ``"val"`` and ``"test"`` to that population's lists.
    """
    return {
        _VAL: restrict_ranked_lists(ranked_lists, populations.val_sources),
        _TEST: restrict_ranked_lists(ranked_lists, populations.test_sources),
    }


def _population_record(
    pair: AlignmentPair, split_lists: Dict[str, List[RankedList]],
    populations: EvaluationPopulations,
) -> Dict[str, int]:
    """
    Summarise the sizes of the populations a pair was tuned and tested on.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair being evaluated.
    split_lists : Dict[str, List[RankedList]]
        Ranked lists of the ``"val"`` and ``"test"`` populations.
    populations : EvaluationPopulations
        Source assignment returned by :func:`build_populations`.

    Returns
    -------
    Dict[str, int]
        Reference counts, ranked-source counts (after partial-reference
        filtering), unmatched-source counts, and ``reference_complete``.
    """
    return {
        "n_train_refs": len(pair.train_refs),
        "n_val_refs": len(pair.val_refs),
        "n_test_refs": len(pair.test_refs),
        "n_val_ranked_sources": len(split_lists[_VAL]),
        "n_test_ranked_sources": len(split_lists[_TEST]),
        "n_val_unmatched_sources": populations.n_unmatched_val,
        "n_test_unmatched_sources": populations.n_unmatched_test,
        "reference_complete": pair.reference_complete,
    }


def _process_pair(
    pair: AlignmentPair,
    condition: ExperimentCondition,
    encoder: EmbeddingEncoder,
    candidates: Dict[str, List[str]],
) -> Dict[str, object]:
    verbaliser = build_verbaliser(condition.strategy_name, condition.model_key)
    source_texts = _verbalise_uris(
        verbaliser,
        pair.source_graph,
        pair.source_entities,
        pair.entity_type,
        "source verbalisation",
    )
    candidate_uris = _unique_candidate_uris(pair, candidates)
    candidate_texts = _verbalise_uris(
        verbaliser,
        pair.target_graph,
        candidate_uris,
        pair.entity_type,
        "candidate verbalisation",
    )
    source_embeddings = encoder.encode_batch(source_texts, role="source", show_progress=True)
    candidate_embeddings = encoder.encode_batch(
        candidate_texts, role="candidate", show_progress=True
    )
    candidate_index = {uri: index for index, uri in enumerate(candidate_uris)}
    ranked_lists = _build_ranked_lists(
        pair, source_embeddings, candidates, candidate_index, candidate_embeddings
    )
    return _evaluate_pair(pair, ranked_lists)


def _evaluate_pair(
    pair: AlignmentPair, ranked_lists: List[RankedList]
) -> Dict[str, object]:
    """
    Tune on the validation population and score the test population.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair being evaluated.
    ranked_lists : List[RankedList]
        Ranked candidate lists of every source of the pair.

    Returns
    -------
    Dict[str, object]
        Test metrics with TP/FP/FN counts, the per-entity-type breakdown, and
        the population summary under ``"population"``.
    """
    populations = build_populations(pair)
    judgeable = judgeable_ranked_lists(pair, ranked_lists)
    split_lists = _split_ranked_lists(judgeable, populations)
    threshold = _resolve_threshold(split_lists[_VAL], pair.val_refs)
    metrics = compute_all_metrics(split_lists[_TEST], pair.test_refs, threshold=threshold)
    metrics[_PER_ENTITY_TYPE_KEY] = _per_entity_type_metrics(pair, split_lists, threshold)
    metrics[_POPULATION_KEY] = _population_record(pair, split_lists, populations)
    return metrics


def _candidate_count(candidates: Dict[str, List[str]]) -> int:
    """
    Return the largest candidate-list length across all source entities.

    Candidate lists are truncated per source entity, so the count of an
    arbitrary source under-reports the retrieval breadth of the pair.

    Parameters
    ----------
    candidates : Dict[str, List[str]]
        Mapping from source URI to its ranked candidate target URIs.

    Returns
    -------
    int
        Maximum candidate-list length, or ``0`` for an empty mapping.
    """
    return max(len(uris) for uris in candidates.values()) if candidates else 0


def _effective_ppas(condition: ExperimentCondition) -> bool:
    """
    Return whether the condition's verbalisation can apply PPAS sampling.

    PPAS runs only inside the V3, V4, and V6 verbalisers, and only when the
    model has a token budget, so V1, V2, and V8 strategies never sample under
    any model and a model without a budget (e.g. ``"M3"``) never samples under
    any strategy.  The value is recorded alongside the condition's declared
    ``apply_ppas`` flag, making any divergence between configured intent and
    executed behaviour visible in the results.

    Parameters
    ----------
    condition : ExperimentCondition
        Condition being run.

    Returns
    -------
    bool
        ``True`` when verbalisation can apply PPAS sampling.
    """
    return strategy_uses_ppas(condition.strategy_name, condition.model_key)


def _log_ppas_configuration(condition: ExperimentCondition) -> None:
    """Log the configured and effective PPAS setting for ``condition``."""
    _LOGGER.info(
        "Condition %s: model=%s apply_ppas=%s ppas_effective=%s",
        condition.condition_id,
        condition.model_key,
        condition.apply_ppas,
        _effective_ppas(condition),
    )


_PROVENANCE_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "numpy": "numpy",
    "scipy": "scipy",
    "rdflib": "rdflib",
}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_TIMEOUT_SECONDS = 10

_PACKAGE_DISTRIBUTION = "kgsemembed"
_UNKNOWN_VERSION = "unknown"


def _library_versions() -> Dict[str, str]:
    """
    Report versions of the interpreter and libraries that determine embeddings.

    Encoder output depends on the installed embedding stack, so the versions are
    recorded alongside every result to make a run reproducible after upgrades.

    Returns
    -------
    Dict[str, str]
        Mapping from ``"python"`` and each library name to its version, with
        ``"unknown"`` for a library that is not installed.
    """
    versions = {"python": platform.python_version()}
    for name, distribution in _PROVENANCE_DISTRIBUTIONS.items():
        try:
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = _UNKNOWN_VERSION
    return versions


def _package_version() -> str:
    """
    Report the installed ``kgsemembed`` version.

    Returns
    -------
    str
        The distribution version, or ``"unknown"`` when the package is not
        installed as a distribution, e.g. when run straight from a source tree.
    """
    try:
        return metadata.version(_PACKAGE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return _UNKNOWN_VERSION


def _model_revision(model_info: Optional[Dict[str, str]]) -> str:
    """
    Report the Hugging Face revision of the model that produced a result.

    Parameters
    ----------
    model_info : Optional[Dict[str, str]]
        Provenance metadata returned by the model-loading layer.

    Returns
    -------
    str
        The loaded revision, or ``"unknown"`` when the loader supplied none.
    """
    if not model_info:
        return _UNKNOWN_VERSION
    return model_info.get("hf_revision", _UNKNOWN_VERSION)


def _git_output(*args: str) -> Optional[str]:
    """Return the stripped stdout of a git command in the repository, or ``None``."""
    try:
        completed = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True,
            timeout=_GIT_TIMEOUT_SECONDS, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _git_state() -> Dict[str, object]:
    """
    Report the source revision a result was produced from.

    Returns
    -------
    Dict[str, object]
        ``git_commit`` (``"unknown"`` outside a work tree) and ``git_dirty``,
        ``True`` when tracked or untracked changes were present, ``None`` when
        unknown.
    """
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    return {
        "git_commit": commit or _UNKNOWN_VERSION,
        "git_dirty": None if status is None else bool(status),
    }


def _candidate_digest(candidates: Dict[str, List[str]]) -> str:
    """
    Fingerprint the candidate lists a result was computed from.

    Parameters
    ----------
    candidates : Dict[str, List[str]]
        Mapping from source URI to its ranked candidate target URIs.

    Returns
    -------
    str
        SHA-256 hex digest of the mapping serialised with sorted keys.
    """
    encoded = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def _provenance(model_info: Optional[Dict[str, str]]) -> Dict[str, object]:
    """
    Assemble the model, library, runtime, and execution provenance of a result.

    ``python_version`` intentionally duplicates ``versions["python"]``: the
    top-level field carries the full interpreter string for auditing a single
    result file, while the nested value stays available for aggregation.

    Parameters
    ----------
    model_info : Optional[Dict[str, str]]
        Provenance metadata returned by the model-loading layer.

    Returns
    -------
    Dict[str, object]
        Provenance fields to merge into a result payload.
    """
    return {
        "hf_revision": _model_revision(model_info),
        "kgsemembed_version": _package_version(),
        "python_version": sys.version,
        "run_timestamp": _utc_timestamp(),
        "versions": _library_versions(),
        "random_seed": _RANDOM_SEED,
        "split_seed": _SPLIT_SEED,
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        **_git_state(),
    }


def _write_result(
    results_dir: str | Path,
    condition: ExperimentCondition,
    pair: AlignmentPair,
    metrics: Dict[str, object],
    candidates: Dict[str, List[str]],
    model_info: Optional[Dict[str, str]],
) -> None:
    path = _result_path(results_dir, condition.condition_id, pair.dataset_id, pair.pair_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition_id": condition.condition_id,
        "dataset_id": pair.dataset_id,
        "pair_name": pair.pair_name,
        "strategy": condition.strategy_name,
        "model_key": condition.model_key,
        "model_id": MODEL_REGISTRY[condition.model_key].model_id,
        "apply_ppas": condition.apply_ppas,
        "ppas_effective": _effective_ppas(condition),
        "token_budget": PPAS_BUDGETS.get(condition.model_key),
        "metrics": {key: metrics[key] for key in _METRIC_KEYS},
        "counts": {key: metrics[key] for key in _COUNT_KEYS},
        "per_entity_type": metrics.get(_PER_ENTITY_TYPE_KEY, {}),
        "population": metrics[_POPULATION_KEY],
        "n_source_entities": len(pair.source_entities),
        "n_candidates_per_entity": _candidate_count(candidates),
        "candidates_sha256": _candidate_digest(candidates),
        **_provenance(model_info),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_metrics(path: Path) -> Dict[str, float]:
    """
    Read the headline metrics of an existing result file.

    A file written under an older evaluation protocol is still read, since
    existing results are never overwritten without ``--force_recompute``, but
    a warning names it so stale metrics cannot pass unnoticed.

    Parameters
    ----------
    path : Path
        Location of the result JSON file.

    Returns
    -------
    Dict[str, float]
        The file's ``metrics`` block.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("evaluation_protocol")
    if protocol != EVALUATION_PROTOCOL_VERSION:
        _LOGGER.warning(
            "Existing result %s uses evaluation protocol %s, current is %s; "
            "rerun with --force_recompute",
            path, protocol, EVALUATION_PROTOCOL_VERSION,
        )
    return payload["metrics"]


def _summary_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    """
    Keep only the scalar headline metrics of a pair's computed metrics.

    The breakdown, counts, and population summary are written to the result
    JSON only; the runner's in-memory summary stays a flat mapping of scalar
    metrics, matching what a result file read back from disk supplies.

    Parameters
    ----------
    metrics : Dict[str, object]
        Metrics as returned by :func:`_process_pair`.

    Returns
    -------
    Dict[str, float]
        The scalar metrics of the pair.
    """
    return {key: metrics[key] for key in _METRIC_KEYS}


def _run_pair(
    condition: ExperimentCondition,
    pair: AlignmentPair,
    encoder: EmbeddingEncoder,
    data_dir: str | Path,
    results_dir: str | Path,
    force_recompute: bool,
    model_info: Optional[Dict[str, str]],
) -> Optional[Dict[str, float]]:
    path = _result_path(results_dir, condition.condition_id, pair.dataset_id, pair.pair_name)
    try:
        if path.exists() and not force_recompute:
            _LOGGER.info("Skipping existing result: %s", path)
            return _read_metrics(path)
        candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
        metrics = _process_pair(pair, condition, encoder, candidates)
        _write_result(results_dir, condition, pair, metrics, candidates, model_info)
        return _summary_metrics(metrics)
    except Exception as exc:
        _LOGGER.error(
            "Failed alignment pair %s/%s/%s: %s",
            condition.condition_id,
            pair.dataset_id,
            pair.pair_name,
            exc,
        )
        return None


def _report_failures(dataset_id: str, n_pairs: int, n_successes: int) -> int:
    """
    Log how many alignment pairs of a dataset failed and return that count.

    The printed summary is only reached by ``run_all_conditions``, so the count
    is also logged to keep failures visible to a single-condition run.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier, e.g. ``"D3"``.
    n_pairs : int
        Number of alignment pairs the dataset holds.
    n_successes : int
        Number of pairs that produced metrics.

    Returns
    -------
    int
        Number of pairs that failed.
    """
    failures = n_pairs - n_successes
    if failures:
        _LOGGER.warning(
            "Dataset %s: %d of %d alignment pairs failed", dataset_id, failures, n_pairs
        )
    return failures


def _run_dataset(
    condition: ExperimentCondition,
    dataset_id: str,
    encoder: EmbeddingEncoder,
    data_dir: str | Path,
    results_dir: str | Path,
    force_recompute: bool,
    model_info: Optional[Dict[str, str]],
) -> Optional[Dict[str, float]]:
    """
    Run every alignment pair of a dataset and retain the failed outcomes.

    Metrics of the last successful pair are returned, extended with the number
    of pairs that failed, so a partially failed dataset is distinguishable from
    a clean one.  A dataset whose pairs all failed reports the failure count
    without metrics.  A pair read back from an existing result file counts as a
    success, matching the fact that it was not executed and did not fail.

    Parameters
    ----------
    condition : ExperimentCondition
        Condition being executed.
    dataset_id : str
        Dataset identifier, e.g. ``"D3"``.
    encoder : EmbeddingEncoder
        Encoder shared across every pair of the dataset.
    data_dir : str | Path
        Dataset and candidate root directory.
    results_dir : str | Path
        Output directory for result JSON files.
    force_recompute : bool
        Recompute and overwrite existing result files when ``True``.
    model_info : Optional[Dict[str, str]]
        Provenance metadata of the loaded model, recorded in each result.

    Returns
    -------
    Optional[Dict[str, float]]
        Metrics carrying a ``"n_failed_pairs"`` entry, or ``None`` when the
        dataset holds no alignment pairs.
    """
    outcomes = [
        _run_pair(
            condition, pair, encoder, data_dir, results_dir, force_recompute, model_info
        )
        for pair in load_dataset(dataset_id, data_dir)
    ]
    if not outcomes:
        return None
    successes = [metrics for metrics in outcomes if metrics is not None]
    dataset_metrics = dict(successes[-1]) if successes else {}
    dataset_metrics[_FAILURE_KEY] = _report_failures(
        dataset_id, len(outcomes), len(successes)
    )
    return dataset_metrics


def _select_datasets(
    condition: ExperimentCondition, dataset_ids: Optional[List[str]]
) -> List[str]:
    if dataset_ids is None:
        return list(condition.datasets)
    requested = set(dataset_ids)
    return [dataset_id for dataset_id in condition.datasets if dataset_id in requested]


def _seed_random_state() -> None:
    """
    Reset the Python and NumPy global generators to the fixed experiment seed.

    Called once per condition rather than once per process so that a condition
    produces the same results whether it is run alone or after other conditions
    in the same interpreter.
    """
    random.seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)


def _run_condition_with_encoder(
    condition: ExperimentCondition,
    encoder: EmbeddingEncoder,
    dataset_ids: Optional[List[str]],
    data_dir: str | Path,
    results_dir: str | Path,
    force_recompute: bool,
    model_info: Optional[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """
    Run the selected datasets of a condition against an already-loaded encoder.

    ``model_info`` is the provenance metadata of the model backing ``encoder``
    and is recorded in every result written by this call.  It is required at
    every level of the call chain so that provenance cannot be dropped by
    omission, which is how the loaded revision went unrecorded before.
    """
    results: Dict[str, Dict[str, float]] = {}
    _seed_random_state()
    _log_ppas_configuration(condition)
    for dataset_id in _select_datasets(condition, dataset_ids):
        try:
            metrics = _run_dataset(
                condition,
                dataset_id,
                encoder,
                data_dir,
                results_dir,
                force_recompute,
                model_info,
            )
        except Exception as exc:
            _LOGGER.error(
                "Failed dataset %s for condition %s: %s",
                dataset_id,
                condition.condition_id,
                exc,
            )
            continue
        if metrics is not None:
            results[dataset_id] = metrics
    return results


def _release_model(model: object) -> None:
    del model
    gc.collect()


def run_condition(
    condition_id: str,
    dataset_ids: Optional[List[str]] = None,
    data_dir: str | Path = "data/",
    results_dir: str | Path = "data/results/",
    force_recompute: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Execute the full Phase 2 pipeline for a single experimental condition.

    The condition's embedding model is loaded once, reused for every alignment
    pair, and released on completion.  Results are written per pair and skipped
    when a result file already exists unless ``force_recompute`` is set.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    dataset_ids : Optional[List[str]]
        Datasets to run; ``None`` runs every dataset configured for the
        condition.
    data_dir : str | Path
        Dataset and candidate root directory.
    results_dir : str | Path
        Output directory for result JSON files.
    force_recompute : bool
        Recompute and overwrite existing result files when ``True``.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping from dataset identifier to its computed metrics dictionary.
        When a dataset holds several alignment pairs, the value is the metrics
        of the last successfully processed pair; every pair is always written to
        its own result JSON regardless.  Each entry additionally carries
        ``"n_failed_pairs"``, the number of alignment pairs that failed.

    Raises
    ------
    KeyError
        If ``condition_id`` is not registered.
    """
    _seed_random_state()
    condition = get_condition(condition_id)
    model, model_info = load_sentence_transformer(condition.model_key)
    try:
        encoder = EmbeddingEncoder(condition.model_key, model)
        return _run_condition_with_encoder(
            condition,
            encoder,
            dataset_ids,
            data_dir,
            results_dir,
            force_recompute,
            model_info,
        )
    finally:
        _release_model(model)


def _select_conditions(condition_ids: Optional[List[str]]) -> List[ExperimentCondition]:
    if condition_ids is None:
        return list(EXPERIMENT_CONDITIONS)
    selected: List[ExperimentCondition] = []
    for condition_id in condition_ids:
        try:
            selected.append(get_condition(condition_id))
        except KeyError:
            _LOGGER.error("Unknown condition identifier: %s", condition_id)
    return selected


def _group_by_model(
    conditions: List[ExperimentCondition],
) -> Dict[str, List[ExperimentCondition]]:
    grouped: Dict[str, List[ExperimentCondition]] = {}
    for condition in conditions:
        grouped.setdefault(condition.model_key, []).append(condition)
    return grouped


def _summary_metric(metrics: Dict[str, float], key: str) -> str:
    """Format a metric for the summary, reading ``'n/a'`` when no pair succeeded."""
    value = metrics.get(key)
    return "n/a" if value is None else f"{value:.4f}"


def _failure_cell(metrics: Dict[str, float]) -> str:
    """Render the failure count so a failed dataset reads as ``'7 failed'``."""
    failures = int(metrics.get(_FAILURE_KEY, 0))
    return f"{failures} failed" if failures else "0"


def _print_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\nExperiment summary")
    print(
        f"{'condition':<12}{'dataset':<10}{'f1':>8}{'precision':>11}"
        f"{'recall':>9}{'failures':>12}"
    )
    for condition_id, dataset_metrics in summary.items():
        for dataset_id, metrics in dataset_metrics.items():
            print(
                f"{condition_id:<12}{dataset_id:<10}"
                f"{_summary_metric(metrics, 'f1'):>8}"
                f"{_summary_metric(metrics, 'precision'):>11}"
                f"{_summary_metric(metrics, 'recall'):>9}"
                f"{_failure_cell(metrics):>12}"
            )


def run_all_conditions(
    condition_ids: Optional[List[str]] = None,
    dataset_ids: Optional[List[str]] = None,
    data_dir: str | Path = "data/",
    results_dir: str | Path = "data/results/",
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Execute multiple conditions, grouping by embedding model to load each once.

    Conditions are grouped by ``model_key`` so that a shared
    ``SentenceTransformer`` is loaded a single time per group, reused across
    every condition in the group, and released before the next group begins.

    Parameters
    ----------
    condition_ids : Optional[List[str]]
        Conditions to run; ``None`` runs every registered condition.
    dataset_ids : Optional[List[str]]
        Datasets to run; ``None`` runs every dataset configured per condition.
    data_dir : str | Path
        Dataset and candidate root directory.
    results_dir : str | Path
        Output directory for result JSON files.
    force_recompute : bool
        Recompute and overwrite existing result files when ``True``.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Mapping from condition identifier to its per-dataset metrics.
    """
    conditions = _select_conditions(condition_ids)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_key, group in _group_by_model(conditions).items():
        model, model_info = load_sentence_transformer(model_key)
        try:
            encoder = EmbeddingEncoder(model_key, model)
            for condition in group:
                try:
                    summary[condition.condition_id] = _run_condition_with_encoder(
                        condition,
                        encoder,
                        dataset_ids,
                        data_dir,
                        results_dir,
                        force_recompute,
                        model_info,
                    )
                except Exception as exc:
                    _LOGGER.error("Failed condition %s: %s", condition.condition_id, exc)
        finally:
            _release_model(model)
    _print_summary(summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 embedding experiment pipeline."
    )
    parser.add_argument("--conditions", nargs="*", default=None, help="Condition IDs to run.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset IDs to run.")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    parser.add_argument("--results_dir", default="data/results/", help="Result output directory.")
    parser.add_argument(
        "--force_recompute", action="store_true", help="Overwrite existing result files."
    )
    return parser


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Parse command-line arguments and run the requested conditions."""
    _seed_random_state()
    args = _build_arg_parser().parse_args(argv)
    run_all_conditions(
        condition_ids=args.conditions,
        dataset_ids=args.datasets,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    cli_main()
