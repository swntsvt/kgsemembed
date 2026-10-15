"""Runner for the controlled PPAS ablation (#122).

Loads the shared embedding model once, then for each alignment pair builds
every entity's eligible items once and runs each ablation condition over them:
select under the condition's policy and budget, encode, rank the candidate
files, tune on validation, and score on test exactly as the main pipeline does
(:func:`kgsemembed.pipeline.run_experiment._evaluate_pair`).

Within a pair, embeddings are cached by ``(role, text)``.  An entity that needs
no sampling has the same text under every policy, so it receives a bit-identical
embedding in every condition, and differences between conditions arise only
from entities whose selection differs.

Results go to ``data/results_ppas_ablation/`` by default, never to the
historical ``data/results/`` tree.  Each pair writes a result JSON in the main
schema, extended with an ``"ablation"`` block, and an ``_entities.json`` sidecar
holding per-test-source ranks and sampling flags for stratified analysis.

Usage
-----
python -m kgsemembed.pipeline.ppas_ablation --datasets D5 \\
    --data_dir data/ --results_dir data/results_ppas_ablation/
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdflib import URIRef
from tqdm import tqdm

from kgsemembed.candidates import load_candidates
from kgsemembed.candidates.ngram import _candidates_path
from kgsemembed.datasets import AlignmentPair, load_dataset
from kgsemembed.embeddings import EmbeddingEncoder, MODEL_REGISTRY, load_sentence_transformer
from kgsemembed.embeddings.models import PAD_TO_MULTIPLE_OF
from kgsemembed.evaluation import RankedList
from kgsemembed.pipeline.ablation_conditions import (
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    AblationCondition,
    get_ablation_condition,
)
from kgsemembed.pipeline.run_experiment import (
    _COUNT_KEYS,
    _METRIC_KEYS,
    _POPULATION_KEY,
    _build_ranked_lists,
    _candidate_count,
    _candidate_digest,
    _entity_type_for,
    _evaluate_pair,
    _provenance,
    _seed_random_state,
    _unique_candidate_uris,
)
from kgsemembed.verbalisation.predicate_config import get_predicate_config
from kgsemembed.verbalisation.selection import (
    DETERMINISTIC_ORDER,
    ITEM_SEPARATOR,
    PPAS,
    EligibleItem,
    PolicySelector,
    Selection,
    TokenCounter,
    make_token_counter,
    tier_config_digest,
)

_LOGGER = logging.getLogger("kgsemembed.pipeline.ppas_ablation")

DEFAULT_RESULTS_DIR = "data/results_ppas_ablation/"
_HISTORICAL_RESULTS_DIR = Path("data/results")
_SOURCE = "source"
_CANDIDATE = "candidate"
_REPRESENTATION = (
    "Structured key-value items (V6 rendering) from _collect_triples; literal "
    "datatype suffix stripped; empty and duplicate items dropped; selected "
    "items joined in canonical order"
)
_SIMILARITY = "cosine (dot product of unit-normalised embeddings)"
_THRESHOLD_PROCEDURE = (
    "tuned on the validation population only, grid 0.10-0.99 step 0.01, "
    "ties to the lower median; one top-ranked prediction per source"
)
_TOKEN_COUNTING = (
    "model tokenizer on the final joined text, special tokens included, "
    "no truncation"
)


@dataclass(frozen=True)
class SideItems:
    """
    Eligible items of every entity on one side of a pair.

    Attributes
    ----------
    uris : List[str]
        Entity URIs, in encoding order.
    entity_types : List[str]
        Resolved entity type of each URI.
    items : List[List[EligibleItem]]
        Eligible items of each URI, in canonical order.
    """

    uris: List[str]
    entity_types: List[str]
    items: List[List[EligibleItem]]


class EmbeddingCache:
    """
    Encode texts once per ``(role, text)`` and reuse them across conditions.

    Parameters
    ----------
    encoder : EmbeddingEncoder
        Encoder of the shared ablation model.
    """

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        self.encoder = encoder
        self._store: Dict[Tuple[str, str], np.ndarray] = {}

    def encode(self, texts: List[str], role: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Return embeddings of ``texts``, encoding only those not yet cached.

        Parameters
        ----------
        texts : List[str]
            Texts to embed, one per entity.
        role : str
            ``"source"`` or ``"candidate"``.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            Embeddings in ``texts`` order, and the new-text count and seconds.
        """
        missing = sorted({text for text in texts if (role, text) not in self._store})
        start = time.perf_counter()
        if missing:
            vectors = self.encoder.encode_batch(missing, role=role, show_progress=True)
            self._store.update({(role, t): v for t, v in zip(missing, vectors)})
        seconds = time.perf_counter() - start
        stacked = np.stack([self._store[(role, text)] for text in texts])
        return stacked, {"n_texts_encoded": len(missing), "encode_seconds": seconds}


@dataclass
class PairContext:
    """
    Everything shared by the conditions run on one alignment pair.

    Attributes
    ----------
    pair : AlignmentPair
        Alignment pair under evaluation.
    candidates : Dict[str, List[str]]
        Candidate lists of the pair, identical for every condition.
    source : SideItems
        Eligible items of the pair's source entities.
    candidate : SideItems
        Eligible items of every unique candidate entity.
    cache : EmbeddingCache
        Embedding cache shared by the pair's conditions.
    count_tokens : TokenCounter
        Encoded-length counter of the shared model's tokenizer.
    """

    pair: AlignmentPair
    candidates: Dict[str, List[str]]
    source: SideItems
    candidate: SideItems
    cache: EmbeddingCache
    count_tokens: TokenCounter


def collect_side(
    graph: object, uris: List[str], pair_entity_type: str, selector: PolicySelector
) -> SideItems:
    """
    Build the eligible items of every entity on one side of a pair.

    Parameters
    ----------
    graph : rdflib.Graph
        Graph holding the side's triples.
    uris : List[str]
        Entities to build items for.
    pair_entity_type : str
        The pair's entity type, ``"mixed"`` resolving per URI.
    selector : PolicySelector
        Any selector; item building does not depend on the policy.

    Returns
    -------
    SideItems
        Items and entity types of every URI, in ``uris`` order.
    """
    types = [_entity_type_for(pair_entity_type, graph, uri) for uri in uris]
    items = [selector.eligible_items(graph, URIRef(uri)) for uri in tqdm(uris, desc="eligible items")]
    return SideItems(list(uris), types, items)


def select_side(selector: PolicySelector, side: SideItems) -> Tuple[List[Selection], float]:
    """
    Select every entity of a side under one policy and time it.

    Parameters
    ----------
    selector : PolicySelector
        Selector configured with the condition's policy, budget and seed.
    side : SideItems
        Eligible items of the side.

    Returns
    -------
    Tuple[List[Selection], float]
        One selection per entity, and the wall-clock seconds spent selecting.
    """
    start = time.perf_counter()
    selections = [
        selector.select(items, uri, entity_type)
        for uri, entity_type, items in zip(side.uris, side.entity_types, side.items)
    ]
    return selections, time.perf_counter() - start


def _distribution(values: np.ndarray) -> Dict[str, float]:
    """Summarise integer counts by mean, median, p90, max and total."""
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0, "total": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
        "total": int(values.sum()),
    }


def side_summary(selections: List[Selection], max_seq_length: int) -> Dict[str, object]:
    """
    Account for the sampling, token usage and truncation of one side.

    Parameters
    ----------
    selections : List[Selection]
        Selections of every entity on the side.
    max_seq_length : int
        Length beyond which the model truncates its input.

    Returns
    -------
    Dict[str, object]
        Counts of sampled, empty, item-less and model-truncated entities, the
        oversized-item count, and token and eligible-item distributions.
        For Issue #126 controlled policies, also includes filtering metrics.
    """
    tokens = np.array([s.n_tokens for s in selections], dtype=np.int64)
    eligible = np.array([s.n_eligible for s in selections], dtype=np.int64)
    summary = {
        "n_entities": len(selections),
        "n_sampling_invoked": int(sum(s.sampling_invoked for s in selections)),
        "n_empty_text": int(sum(not s.text for s in selections)),
        "n_no_eligible_items": int((eligible == 0).sum()),
        "n_oversized_items": int(sum(s.n_oversized for s in selections)),
        "n_model_truncated": int((tokens > max_seq_length).sum()),
        "n_selected_items": int(sum(len(s.selected) for s in selections)),
        "tokens": _distribution(tokens),
        "model_input_tokens_total": int(np.minimum(tokens, max_seq_length).sum()),
        "eligible_items": _distribution(eligible),
    }

    # Add filtering metrics if present (Issue #126)
    if selections and selections[0].filter_mode != "all":
        n_filtered = np.array([s.n_filtered_items for s in selections], dtype=np.int64)
        coverage = np.array([s.predicate_coverage for s in selections], dtype=np.float64)
        summary["n_filtered_items"] = int(n_filtered.sum())
        summary["predicate_coverage"] = {
            "mean": float(coverage.mean()),
            "min": float(coverage.min()),
            "max": float(coverage.max()),
        }

    return summary


def selection_digest(uris: Dict[str, List[str]], selections: Dict[str, List[Selection]]) -> str:
    """
    Fingerprint the exact texts a condition encoded on both sides.

    Parameters
    ----------
    uris : Dict[str, List[str]]
        Entity URIs per side.
    selections : Dict[str, List[Selection]]
        Selections per side, aligned with ``uris``.

    Returns
    -------
    str
        SHA-256 hex digest of ``[side, uri, text]`` rows.
    """
    rows = [
        [side, uri, selection.text]
        for side in sorted(uris)
        for uri, selection in zip(uris[side], selections[side])
    ]
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_fingerprint(text: str) -> str:
    """Return a short stable fingerprint of a selected text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def sampled_fingerprints(side: SideItems, selections: List[Selection]) -> Dict[str, str]:
    """
    Fingerprint the selections of the side's sampled entities.

    Unsampled entities carry their full text under every policy, so only
    sampled ones can differ between conditions.

    Parameters
    ----------
    side : SideItems
        Entities of the side.
    selections : List[Selection]
        Their selections under one condition.

    Returns
    -------
    Dict[str, str]
        URI to text fingerprint, for entities whose sampling was invoked.
    """
    return {
        uri: _text_fingerprint(selection.text)
        for uri, selection in zip(side.uris, selections)
        if selection.sampling_invoked
    }


def _best_ranks(ranked_lists: List[RankedList], refs: List[Tuple[str, str]]) -> Dict[str, Optional[int]]:
    """Return each reference source's best 1-based gold rank, ``None`` if unranked."""
    positions = {
        ranked[0][0]: {target: rank for rank, (_s, target, _score) in enumerate(ranked, start=1)}
        for ranked in ranked_lists if ranked
    }
    best: Dict[str, Optional[int]] = {}
    for source, target in refs:
        rank = positions.get(source, {}).get(target)
        current = best.get(source)
        best[source] = rank if current is None else (current if rank is None else min(current, rank))
    return best


def reference_source_records(
    context: PairContext, selections: Dict[str, List[Selection]], ranked_lists: List[RankedList]
) -> List[Dict[str, object]]:
    """
    Describe every test reference source for stratified analysis.

    Parameters
    ----------
    context : PairContext
        Pair under evaluation.
    selections : Dict[str, List[Selection]]
        Selections per side under the condition.
    ranked_lists : List[RankedList]
        The condition's ranked lists.

    Returns
    -------
    List[Dict[str, object]]
        Per source: best gold rank (``None`` when unranked, scored as
        reciprocal rank 0), its eligible-item count, and whether sampling was
        invoked for it or for any of its gold targets.
    """
    refs = context.pair.test_refs
    best = _best_ranks(ranked_lists, refs)
    source_sel = dict(zip(context.source.uris, selections[_SOURCE]))
    target_sel = dict(zip(context.candidate.uris, selections[_CANDIDATE]))
    targets: Dict[str, List[str]] = {}
    for source, target in refs:
        targets.setdefault(source, []).append(target)
    return [
        {
            "source": source,
            "best_rank": best[source],
            "source_degree": source_sel[source].n_eligible if source in source_sel else 0,
            "source_sampled": bool(source in source_sel and source_sel[source].sampling_invoked),
            "target_sampled": any(t in target_sel and target_sel[t].sampling_invoked for t in gold),
        }
        for source, gold in sorted(targets.items())
    ]


def _encoder_settings(encoder: EmbeddingEncoder, max_seq_length: int) -> Dict[str, object]:
    """Record the encoder configuration that shapes every embedding."""
    config = encoder.config
    return {
        "batch_size": config.batch_size,
        "normalize_embeddings": True,
        "pad_to_multiple_of": PAD_TO_MULTIPLE_OF,
        "max_seq_length": max_seq_length,
        "query_prefix": config.query_prefix,
        "doc_prefix": config.doc_prefix,
        "similarity": _SIMILARITY,
        "embedding_cache": "per pair, keyed by (role, text)",
    }


@dataclass(frozen=True)
class RunSettings:
    """
    Settings shared by every pair of one ablation run.

    Attributes
    ----------
    data_dir : Path
        Dataset and candidate root.
    results_dir : Path
        Output root for ablation results.
    force_recompute : bool
        Overwrite existing results when ``True``.
    model_info : Dict[str, str]
        Provenance of the loaded model.
    tokenizer_id : str
        ``name_or_path`` of the tokenizer used for budgets.
    max_seq_length : int
        Model input length limit.
    encoder_settings : Dict[str, object]
        Output of :func:`_encoder_settings`.
    max_sources : Optional[int]
        Smoke-run source cap, ``None`` for a full run.
    """

    data_dir: Path
    results_dir: Path
    force_recompute: bool
    model_info: Dict[str, str]
    tokenizer_id: str
    max_seq_length: int
    encoder_settings: Dict[str, object]
    max_sources: Optional[int] = None


def ablation_block(
    condition: AblationCondition, settings: RunSettings, summaries: Dict[str, object]
) -> Dict[str, object]:
    """
    Assemble the reproducibility record of one condition on one pair.

    Parameters
    ----------
    condition : AblationCondition
        Condition that was run.
    settings : RunSettings
        Run-wide settings.
    summaries : Dict[str, object]
        Per-side summaries, ``selection_sha256`` and runtime measurements.

    Returns
    -------
    Dict[str, object]
        The ``"ablation"`` block of the result JSON.
    """
    block = {
        "policy": condition.policy,
        "seed": condition.seed,
        "nominal_token_budget": condition.token_budget,
        "budget_includes_special_tokens": True,
        "token_counting": _TOKEN_COUNTING,
        "tokenizer": settings.tokenizer_id,
        "tokenizer_revision": settings.model_info.get("hf_revision", "unknown"),
        "representation": _REPRESENTATION,
        "item_separator": ITEM_SEPARATOR,
        "deterministic_order": DETERMINISTIC_ORDER,
        "tier_config_sha256": tier_config_digest(),
        "encoder": settings.encoder_settings,
        "threshold_procedure": _THRESHOLD_PROCEDURE,
        "smoke_max_sources": settings.max_sources,
        "status": "completed",
        **summaries,
    }

    # Add Issue #126 controlled policy fields if present
    # Conditions with non-default filter_mode or predicate_config are controlled policies
    if condition.filter_mode != "all" or condition.predicate_config != "base":
        block["filter_mode"] = condition.filter_mode
        block["order_mode"] = condition.order_mode
        block["predicate_config"] = {
            "name": condition.predicate_config,
            "config_hash": get_predicate_config(condition.predicate_config).config_hash,
            "n_covered_predicates": len(
                get_predicate_config(condition.predicate_config).covered_predicates
            ),
            "fma_special_predicates": list(
                get_predicate_config(condition.predicate_config).fma_special_predicates.keys()
            ),
        }

    return block


def _result_paths(results_dir: Path, condition_id: str, pair: AlignmentPair) -> Tuple[Path, Path]:
    """Return the result and sidecar paths of a condition on a pair."""
    directory = results_dir / condition_id / pair.dataset_id
    return (
        directory / f"{pair.pair_name}_results.json",
        directory / f"{pair.pair_name}_entities.json",
    )


def _result_payload(
    condition: AblationCondition, context: PairContext, metrics: Dict[str, object],
    ablation: Dict[str, object], settings: RunSettings,
) -> Dict[str, object]:
    """Lay out a result in the main pipeline's schema plus the ablation block."""
    sampled = ablation[_SOURCE]["n_sampling_invoked"] + ablation[_CANDIDATE]["n_sampling_invoked"]
    pair = context.pair
    return {
        "condition_id": condition.condition_id,
        "dataset_id": pair.dataset_id,
        "pair_name": pair.pair_name,
        "strategy": "V6-KV-select",
        "model_key": condition.model_key,
        "model_id": MODEL_REGISTRY[condition.model_key].model_id,
        "apply_ppas": condition.policy == PPAS,
        "ppas_effective": condition.policy == PPAS and sampled > 0,
        "token_budget": condition.token_budget,
        "metrics": {key: metrics[key] for key in _METRIC_KEYS},
        "counts": {key: metrics[key] for key in _COUNT_KEYS},
        "per_entity_type": metrics.get("per_entity_type", {}),
        "population": metrics[_POPULATION_KEY],
        "n_source_entities": len(pair.source_entities),
        "n_candidates_per_entity": _candidate_count(context.candidates),
        "candidates_sha256": _candidate_digest(context.candidates),
        "candidates_file": str(_candidates_path(pair.dataset_id, pair.pair_name, settings.data_dir)),
        "ablation": ablation,
        **_provenance(settings.model_info),
    }


def _select_and_encode(
    condition: AblationCondition, context: PairContext
) -> Tuple[Dict[str, List[Selection]], Dict[str, np.ndarray], Dict[str, object]]:
    """Select both sides under the condition, then embed them through the cache."""
    # Support both legacy (policy only) and new (filter_mode + order_mode) interfaces
    # Conditions with non-default filter_mode or predicate_config are controlled policies
    if condition.filter_mode != "all" or condition.predicate_config != "base":
        # New interface: use explicit filter_mode, order_mode, predicate_config
        predicate_config = get_predicate_config(condition.predicate_config)
        selector = PolicySelector(
            order_mode=condition.order_mode,
            count_tokens=context.count_tokens,
            budget=condition.token_budget,
            seed=condition.seed,
            filter_mode=condition.filter_mode,
            predicate_config=predicate_config,
        )
    else:
        # Legacy interface: use policy parameter
        selector = PolicySelector(
            policy=condition.policy,
            count_tokens=context.count_tokens,
            budget=condition.token_budget,
            seed=condition.seed,
        )
    selections, embeddings, runtime = {}, {}, {}
    for side_name, side in ((_SOURCE, context.source), (_CANDIDATE, context.candidate)):
        selections[side_name], seconds = select_side(selector, side)
        texts = [selection.text for selection in selections[side_name]]
        embeddings[side_name], encoded = context.cache.encode(texts, side_name)
        runtime[side_name] = {"selection_seconds": seconds, **encoded}
    return selections, embeddings, runtime


def run_condition_on_pair(
    condition: AblationCondition, context: PairContext, settings: RunSettings
) -> Dict[str, object]:
    """
    Run one ablation condition on one pair and write its result files.

    Parameters
    ----------
    condition : AblationCondition
        Condition to run.
    context : PairContext
        Shared items, candidates and cache of the pair.
    settings : RunSettings
        Run-wide settings.

    Returns
    -------
    Dict[str, object]
        The written result payload.
    """
    selections, embeddings, runtime = _select_and_encode(condition, context)
    candidate_index = {uri: i for i, uri in enumerate(context.candidate.uris)}
    ranked = _build_ranked_lists(
        context.pair, embeddings[_SOURCE], context.candidates, candidate_index, embeddings[_CANDIDATE]
    )
    metrics = _evaluate_pair(context.pair, ranked)
    summaries = {
        side: side_summary(selections[side], settings.max_seq_length) for side in selections
    }
    uris = {_SOURCE: context.source.uris, _CANDIDATE: context.candidate.uris}
    summaries["selection_sha256"] = selection_digest(uris, selections)
    summaries["runtime"] = runtime
    payload = _result_payload(condition, context, metrics, ablation_block(condition, settings, summaries), settings)
    _write_outputs(condition, context, payload, selections, ranked, settings)
    return payload


def _write_outputs(
    condition: AblationCondition, context: PairContext, payload: Dict[str, object],
    selections: Dict[str, List[Selection]], ranked: List[RankedList], settings: RunSettings,
) -> None:
    """Write the result JSON and the per-entity sidecar of a condition."""
    result_path, sidecar_path = _result_paths(settings.results_dir, condition.condition_id, context.pair)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "condition_id": condition.condition_id,
        "test_sources": reference_source_records(context, selections, ranked),
        "sampled_fingerprints": {
            _SOURCE: sampled_fingerprints(context.source, selections[_SOURCE]),
            _CANDIDATE: sampled_fingerprints(context.candidate, selections[_CANDIDATE]),
        },
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def restrict_sources(pair: AlignmentPair, max_sources: Optional[int]) -> AlignmentPair:
    """
    Keep a stable subset of a pair's sources for a smoke run.

    Sources are ordered by ``zlib.crc32`` of their URI and the first
    ``max_sources`` kept, with references restricted to them.

    Parameters
    ----------
    pair : AlignmentPair
        Full alignment pair.
    max_sources : Optional[int]
        Number of sources to keep; ``None`` returns ``pair`` unchanged.

    Returns
    -------
    AlignmentPair
        The pair, or a copy restricted to the kept sources.
    """
    if max_sources is None:
        return pair
    kept = sorted(pair.source_entities, key=lambda u: zlib.crc32(u.encode("utf-8")))[:max_sources]
    keep = set(kept)
    return dataclasses.replace(
        pair, source_entities=[u for u in pair.source_entities if u in keep],
        val_refs=_refs_from(pair.val_refs, keep), test_refs=_refs_from(pair.test_refs, keep),
        train_refs=_refs_from(pair.train_refs, keep),
    )


def _refs_from(refs: List[Tuple[str, str]], sources: set) -> List[Tuple[str, str]]:
    """Return the references whose source is in ``sources``."""
    return [ref for ref in refs if ref[0] in sources]


def build_pair_context(
    pair: AlignmentPair, encoder: EmbeddingEncoder, count_tokens: TokenCounter, data_dir: Path
) -> PairContext:
    """
    Load a pair's candidates and build the items every condition shares.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair, possibly restricted for a smoke run.
    encoder : EmbeddingEncoder
        Encoder of the shared model.
    count_tokens : TokenCounter
        Encoded-length counter of the model's tokenizer.
    data_dir : Path
        Dataset and candidate root.

    Returns
    -------
    PairContext
        Candidates, eligible items of both sides, and a fresh embedding cache.
    """
    candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    builder = PolicySelector("full", count_tokens)
    source = collect_side(pair.source_graph, pair.source_entities, pair.entity_type, builder)
    candidate_uris = _unique_candidate_uris(pair, candidates)
    candidate = collect_side(pair.target_graph, candidate_uris, pair.entity_type, builder)
    return PairContext(pair, candidates, source, candidate, EmbeddingCache(encoder), count_tokens)


def _guard_results_dir(results_dir: Path) -> None:
    """Refuse to write into the historical results tree."""
    if results_dir.resolve() == _HISTORICAL_RESULTS_DIR.resolve():
        raise ValueError(
            f"Ablation results must not be written to {_HISTORICAL_RESULTS_DIR}; "
            f"use {DEFAULT_RESULTS_DIR}."
        )


def _pending(conditions: List[AblationCondition], pair: AlignmentPair, settings: RunSettings) -> List[AblationCondition]:
    """Return the conditions whose result for ``pair`` must be (re)computed."""
    pending = []
    for condition in conditions:
        result_path, _ = _result_paths(settings.results_dir, condition.condition_id, pair)
        if result_path.exists() and not settings.force_recompute:
            _LOGGER.info("Skipping existing result: %s", result_path)
            continue
        pending.append(condition)
    return pending


def run_dataset(
    dataset_id: str, conditions: List[AblationCondition], encoder: EmbeddingEncoder,
    count_tokens: TokenCounter, settings: RunSettings,
) -> List[Dict[str, object]]:
    """
    Run every applicable condition, in registry order, on a dataset's pairs.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier, e.g. ``"D5"``.
    conditions : List[AblationCondition]
        Conditions selected for the run.
    encoder : EmbeddingEncoder
        Encoder of the shared model.
    count_tokens : TokenCounter
        Encoded-length counter of the model's tokenizer.
    settings : RunSettings
        Run-wide settings.

    Returns
    -------
    List[Dict[str, object]]
        Result payloads written by this call.
    """
    applicable = [c for c in conditions if dataset_id in c.datasets]
    payloads = []
    for pair in load_dataset(dataset_id, settings.data_dir):
        pair = restrict_sources(pair, settings.max_sources)
        pending = _pending(applicable, pair, settings)
        if not pending:
            continue
        context = build_pair_context(pair, encoder, count_tokens, settings.data_dir)
        for condition in pending:
            _seed_random_state()
            _LOGGER.info("Running %s on %s/%s", condition.condition_id, pair.dataset_id, pair.pair_name)
            payloads.append(run_condition_on_pair(condition, context, settings))
    return payloads


def _select_conditions(condition_ids: Optional[List[str]]) -> List[AblationCondition]:
    """Return the requested conditions in registry order; all when ``None``."""
    if condition_ids is None:
        return list(ABLATION_CONDITIONS)
    requested = [get_ablation_condition(cid) for cid in condition_ids]
    return [c for c in ABLATION_CONDITIONS if c in requested]


def run_ablation(
    dataset_ids: List[str], condition_ids: Optional[List[str]], settings_kwargs: Dict[str, object]
) -> List[Dict[str, object]]:
    """
    Load the shared model once and run the ablation on the requested datasets.

    Parameters
    ----------
    dataset_ids : List[str]
        Datasets to run, e.g. ``["D5"]``.
    condition_ids : Optional[List[str]]
        Conditions to run; ``None`` runs every registered one.
    settings_kwargs : Dict[str, object]
        ``data_dir``, ``results_dir``, ``force_recompute`` and ``max_sources``.

    Returns
    -------
    List[Dict[str, object]]
        Result payloads written by the run.
    """
    conditions = _select_conditions(condition_ids)
    _guard_results_dir(Path(settings_kwargs["results_dir"]))
    model, model_info = load_sentence_transformer(ABLATION_MODEL_KEY)
    encoder = EmbeddingEncoder(ABLATION_MODEL_KEY, model)
    settings = RunSettings(
        model_info=model_info, tokenizer_id=model.tokenizer.name_or_path,
        max_seq_length=model.max_seq_length,
        encoder_settings=_encoder_settings(encoder, model.max_seq_length), **settings_kwargs,
    )
    count_tokens = make_token_counter(model.tokenizer)
    payloads = []
    for dataset_id in dataset_ids:
        payloads += run_dataset(dataset_id, conditions, encoder, count_tokens, settings)
    return payloads


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controlled PPAS ablation (#122).")
    parser.add_argument("--datasets", nargs="+", default=["D5", "D1"], help="Dataset IDs.")
    parser.add_argument("--conditions", nargs="*", default=None, help="Ablation condition IDs.")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Ablation result root.")
    parser.add_argument("--force_recompute", action="store_true", help="Overwrite existing results.")
    parser.add_argument(
        "--max_sources", type=int, default=None,
        help="Smoke run: keep this many sources (stable crc32 subset).",
    )
    return parser


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the ablation."""
    _seed_random_state()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _build_arg_parser().parse_args(argv)
    settings_kwargs = {
        "data_dir": Path(args.data_dir), "results_dir": Path(args.results_dir),
        "force_recompute": args.force_recompute, "max_sources": args.max_sources,
    }
    run_ablation(args.datasets, args.conditions, settings_kwargs)


if __name__ == "__main__":
    cli_main()
