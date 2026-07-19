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
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig
from rdflib import Graph, URIRef
from tqdm import tqdm

from kgsemembed.candidates import build_candidate_table, generate_candidates, load_candidates
from kgsemembed.datasets import AlignmentPair, load_dataset, load_oaei_dataset
from kgsemembed.embeddings import EmbeddingEncoder, MODEL_REGISTRY, load_sentence_transformer
from kgsemembed.evaluation import RankedList, compute_all_metrics, tune_threshold
from kgsemembed.pipeline.conditions import (
    EXPERIMENT_CONDITIONS,
    ExperimentCondition,
    get_condition,
)
from kgsemembed.utils.errors import DataError
from kgsemembed.utils.logging import RunContext, get_logger, init_logging
from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.registry import build_verbaliser

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

_LOGGER = logging.getLogger("kgsemembed.pipeline.runner")

_DEFAULT_ENTITY_TYPE = "class"
_N_CANDIDATES_PER_ENTITY = 20
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


def _verbalise_uris(
    verbaliser: VerbaliserBase, graph: Graph, uris: List[str], desc: str
) -> List[str]:
    return [
        verbaliser.verbalise(graph, URIRef(uri), _DEFAULT_ENTITY_TYPE)
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


def _process_pair(
    pair: AlignmentPair,
    condition: ExperimentCondition,
    encoder: EmbeddingEncoder,
    candidates: Dict[str, List[str]],
) -> Dict[str, float]:
    verbaliser = build_verbaliser(condition.strategy_name, condition.model_key)
    source_texts = _verbalise_uris(
        verbaliser, pair.source_graph, pair.source_entities, "source verbalisation"
    )
    candidate_uris = _unique_candidate_uris(pair, candidates)
    candidate_texts = _verbalise_uris(
        verbaliser, pair.target_graph, candidate_uris, "candidate verbalisation"
    )
    source_embeddings = encoder.encode_batch(source_texts, role="source", show_progress=True)
    candidate_embeddings = encoder.encode_batch(
        candidate_texts, role="candidate", show_progress=True
    )
    candidate_index = {uri: index for index, uri in enumerate(candidate_uris)}
    ranked_lists = _build_ranked_lists(
        pair, source_embeddings, candidates, candidate_index, candidate_embeddings
    )
    scored_pairs = [pair_triple for ranked in ranked_lists for pair_triple in ranked]
    threshold = tune_threshold(scored_pairs, pair.val_refs)["best_threshold"]
    return compute_all_metrics(ranked_lists, pair.test_refs, threshold=threshold)


def _write_result(
    results_dir: str | Path,
    condition: ExperimentCondition,
    pair: AlignmentPair,
    metrics: Dict[str, float],
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
        "metrics": {key: metrics[key] for key in _METRIC_KEYS},
        "n_source_entities": len(pair.source_entities),
        "n_candidates_per_entity": _N_CANDIDATES_PER_ENTITY,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metrics"]


def _run_pair(
    condition: ExperimentCondition,
    pair: AlignmentPair,
    encoder: EmbeddingEncoder,
    data_dir: str | Path,
    results_dir: str | Path,
    force_recompute: bool,
) -> Optional[Dict[str, float]]:
    path = _result_path(results_dir, condition.condition_id, pair.dataset_id, pair.pair_name)
    try:
        if path.exists() and not force_recompute:
            _LOGGER.info("Skipping existing result: %s", path)
            return _read_metrics(path)
        candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
        metrics = _process_pair(pair, condition, encoder, candidates)
        _write_result(results_dir, condition, pair, metrics)
        return metrics
    except Exception as exc:
        _LOGGER.error(
            "Failed alignment pair %s/%s/%s: %s",
            condition.condition_id,
            pair.dataset_id,
            pair.pair_name,
            exc,
        )
        return None


def _run_dataset(
    condition: ExperimentCondition,
    dataset_id: str,
    encoder: EmbeddingEncoder,
    data_dir: str | Path,
    results_dir: str | Path,
    force_recompute: bool,
) -> Optional[Dict[str, float]]:
    pairs = load_dataset(dataset_id, data_dir)
    dataset_metrics: Optional[Dict[str, float]] = None
    for pair in pairs:
        metrics = _run_pair(condition, pair, encoder, data_dir, results_dir, force_recompute)
        if metrics is not None:
            dataset_metrics = metrics
    return dataset_metrics


def _select_datasets(
    condition: ExperimentCondition, dataset_ids: Optional[List[str]]
) -> List[str]:
    if dataset_ids is None:
        return list(condition.datasets)
    requested = set(dataset_ids)
    return [dataset_id for dataset_id in condition.datasets if dataset_id in requested]


def _run_condition_with_encoder(
    condition: ExperimentCondition,
    encoder: EmbeddingEncoder,
    dataset_ids: Optional[List[str]],
    data_dir: str | Path,
    results_dir: str | Path,
    force_recompute: bool,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for dataset_id in _select_datasets(condition, dataset_ids):
        try:
            metrics = _run_dataset(
                condition, dataset_id, encoder, data_dir, results_dir, force_recompute
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
        its own result JSON regardless.

    Raises
    ------
    KeyError
        If ``condition_id`` is not registered.
    """
    condition = get_condition(condition_id)
    model = load_sentence_transformer(condition.model_key)
    try:
        encoder = EmbeddingEncoder(condition.model_key, model)
        return _run_condition_with_encoder(
            condition, encoder, dataset_ids, data_dir, results_dir, force_recompute
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


def _print_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\nExperiment summary")
    print(f"{'condition':<12}{'dataset':<10}{'f1':>8}{'precision':>11}{'recall':>9}")
    for condition_id, dataset_metrics in summary.items():
        for dataset_id, metrics in dataset_metrics.items():
            print(
                f"{condition_id:<12}{dataset_id:<10}"
                f"{metrics['f1']:>8.4f}{metrics['precision']:>11.4f}{metrics['recall']:>9.4f}"
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
        model = load_sentence_transformer(model_key)
        try:
            encoder = EmbeddingEncoder(model_key, model)
            for condition in group:
                try:
                    summary[condition.condition_id] = _run_condition_with_encoder(
                        condition, encoder, dataset_ids, data_dir, results_dir, force_recompute
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
    random.seed(42)
    np.random.seed(42)
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
