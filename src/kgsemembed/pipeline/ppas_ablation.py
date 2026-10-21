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

The #128 controlled policies (P1-P7) must go to a separate directory,
``data/results_ppas_controlled/`` by convention; the runner refuses to write
them beside the #122 results.  The validation runs that extend them (P1-P4 on
D1 at 32 and 64, P1-P7 on D2 at 48) go to ``data/results_ppas_validation/``,
which in turn refuses D1 at 48, so the #128 reference stays the only copy of
those results.  The #131 predicate-role conditions (``PR-*``) go only to
``data/results_predicate_roles/``, which accepts nothing else.  Every condition, legacy or controlled, is
selected through one path: the condition's filter mode, order mode and
predicate vocabulary.

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
from kgsemembed.embeddings.models import PAD_TO_MULTIPLE_OF, require_device
from kgsemembed.evaluation import RankedList
from kgsemembed.evaluation.population import without_known_pairs
from kgsemembed.pipeline.ablation_conditions import (
    ABLATION_CONDITIONS,
    ABLATION_MODEL_KEY,
    CONTROLLED_BUDGET,
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
from kgsemembed.utils.errors import DataError
from kgsemembed.verbalisation.predicate_config import get_predicate_config
from kgsemembed.verbalisation.predicate_roles import RoleFilter, UnmappedPredicateError, get_role_map
from kgsemembed.verbalisation.selection import (
    DETERMINISTIC_ORDER,
    ITEM_SEPARATOR,
    PPAS,
    EligibleItem,
    PolicySelector,
    Selection,
    TokenCounter,
    make_token_counter,
)

_LOGGER = logging.getLogger("kgsemembed.pipeline.ppas_ablation")

DEFAULT_RESULTS_DIR = "data/results_ppas_ablation/"
CONTROLLED_RESULTS_DIR = "data/results_ppas_controlled/"
VALIDATION_RESULTS_DIR = "data/results_ppas_validation/"
ROLE_RESULTS_DIR = "data/results_predicate_roles/"
_HISTORICAL_RESULTS_DIR = Path("data/results")
_ISSUE_122_RESULTS_DIR = Path(DEFAULT_RESULTS_DIR)
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
        oversized-item count, token and eligible-item distributions, and the
        eligibility accounting of :func:`eligibility_summary`.
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
        "n_truncated_tokens": int(np.maximum(tokens - max_seq_length, 0).sum()),
        "n_selected_items": int(sum(len(s.selected) for s in selections)),
        "tokens": _distribution(tokens),
        "model_input_tokens_total": int(np.minimum(tokens, max_seq_length).sum()),
        "eligible_items": _distribution(eligible),
    }

    summary.update(eligibility_summary(selections))
    return summary


def eligibility_summary(selections: List[Selection]) -> Dict[str, object]:
    """
    Account for filtering, vocabulary coverage and skipped items on one side.

    Parameters
    ----------
    selections : List[Selection]
        Selections of every entity on the side.

    Returns
    -------
    Dict[str, object]
        Item totals before filtering, removed by the filter, covered by the
        vocabulary, tiered, and eligible but skipped by the budget, plus the
        covered share of all items and the mean per-entity predicate coverage.
    """
    before = sum(s.n_eligible + s.n_filtered_items for s in selections)
    covered = sum(s.n_covered_items for s in selections)
    coverage = np.array([s.predicate_coverage for s in selections], dtype=np.float64)
    return {
        "n_items_before_filter": int(before),
        "n_filtered_items": int(sum(s.n_filtered_items for s in selections)),
        "n_covered_items": int(covered),
        "n_tiered_items": int(sum(s.n_tiered_items for s in selections)),
        "n_skipped_items": int(sum(s.n_eligible - len(s.selected) for s in selections)),
        "vocabulary_item_coverage": covered / before if before else 0.0,
        "predicate_coverage_mean": float(coverage.mean()) if coverage.size else 1.0,
    }


def budget_summary(selections: List[Selection], budget: Optional[int]) -> Dict[str, object]:
    """
    Measure budget utilisation and violations of one side.

    Parameters
    ----------
    selections : List[Selection]
        Selections of every entity on the side.
    budget : Optional[int]
        The condition's token budget; ``None`` for the uncapped reference.

    Returns
    -------
    Dict[str, object]
        Entities over budget, and the token-to-budget ratio over all entities
        and over sampled ones; empty when there is no budget.
    """
    if budget is None:
        return {}
    tokens = np.array([s.n_tokens for s in selections], dtype=np.float64)
    sampled = np.array([s.sampling_invoked for s in selections], dtype=bool)
    return {
        "n_over_budget": int((tokens > budget).sum()),
        "utilisation_mean": float(tokens.mean() / budget) if tokens.size else 0.0,
        "utilisation_sampled_mean": float(tokens[sampled].mean() / budget) if sampled.any() else 0.0,
    }


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


def selection_fingerprints(side: SideItems, selections: List[Selection]) -> Dict[str, str]:
    """
    Fingerprint the selected text of every entity on a side.

    Items have distinct texts and are joined in canonical order, so equal
    fingerprints mean equal selected item sets.  Under different filters even
    unsampled entities can differ, hence every entity is recorded.

    Parameters
    ----------
    side : SideItems
        Entities of the side.
    selections : List[Selection]
        Their selections under one condition.

    Returns
    -------
    Dict[str, str]
        URI to text fingerprint.
    """
    return {uri: _text_fingerprint(s.text) for uri, s in zip(side.uris, selections)}


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
        Per source: best gold rank, counted without known pairs as the MRR
        is (``None`` when unranked, scored as reciprocal rank 0), its
        eligible-item count, and whether sampling was
        invoked for it or for any of its gold targets.
    """
    refs = context.pair.test_refs
    best = _best_ranks(without_known_pairs(context.pair, ranked_lists), refs)
    source_sel = dict(zip(context.source.uris, selections[_SOURCE]))
    target_sel = dict(zip(context.candidate.uris, selections[_CANDIDATE]))
    targets: Dict[str, List[str]] = {}
    for source, target in refs:
        targets.setdefault(source, []).append(target)
    return [
        {
            "source": source,
            "gold_targets": sorted(gold),
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
        "policy_label": condition.policy_label,
        "role_arm": condition.role_arm,
        "excluded_roles": list(condition.excluded_roles),
        "role_map": role_map_record(condition.role_map) if condition.role_map else None,
        "filter_mode": condition.filter_mode,
        "order_mode": condition.order_mode,
        "predicate_config": predicate_config_record(condition.predicate_config),
        "tier_config_sha256": get_predicate_config(condition.predicate_config).tiers_hash(),
        "encoder": settings.encoder_settings,
        "threshold_procedure": _THRESHOLD_PROCEDURE,
        "smoke_max_sources": settings.max_sources,
        "status": "completed",
        **summaries,
    }

    return block


def role_map_record(name: str) -> Dict[str, object]:
    """
    Describe the role map a predicate-role condition used.

    Parameters
    ----------
    name : str
        Role-map name.

    Returns
    -------
    Dict[str, object]
        Name, configuration hash and the role of every mapped predicate.
    """
    role_map = get_role_map(name)
    return {"name": role_map.name, "config_hash": role_map.config_hash, "roles": dict(sorted(role_map.roles.items()))}


def predicate_config_record(name: str) -> Dict[str, object]:
    """
    Describe the predicate vocabulary a condition's selector actually used.

    Parameters
    ----------
    name : str
        Vocabulary name.

    Returns
    -------
    Dict[str, object]
        Name, configuration and tier hashes, covered predicates and roles.
    """
    vocabulary = get_predicate_config(name)
    return {
        "name": vocabulary.name,
        "config_hash": vocabulary.config_hash,
        "tiers_sha256": vocabulary.tiers_hash(),
        "covered_predicates": sorted(vocabulary.covered_predicates),
        "predicate_roles": dict(vocabulary.predicate_roles),
    }


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
        "dataset_files_sha256": dataset_file_digests(pair.dataset_id, settings.data_dir),
        "ablation": ablation,
        **_provenance(settings.model_info),
    }


_DATASET_FILES = {
    "D1": ("d1_snomed_fma/source.rdf", "d1_snomed_fma/target.rdf", "d1_snomed_fma/reference.rdf"),
    "D2": ("d2_anatomy/source.rdf", "d2_anatomy/target.rdf", "d2_anatomy/reference.rdf"),
}


def dataset_file_digests(dataset_id: str, data_dir: Path) -> Dict[str, Optional[str]]:
    """
    Fingerprint the input files of a dataset, as its version record.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.
    data_dir : Path
        Dataset root.

    Returns
    -------
    Dict[str, Optional[str]]
        Relative path to SHA-256, ``None`` for a missing file; empty for
        datasets without a registered file list.
    """
    paths = {rel: Path(data_dir) / rel for rel in _DATASET_FILES.get(dataset_id, ())}
    return {rel: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            for rel, path in paths.items()}


def role_filter_for(condition: AblationCondition) -> Optional[RoleFilter]:
    """
    Return the role filter of a predicate-role condition, ``None`` otherwise.

    Parameters
    ----------
    condition : AblationCondition
        A registered condition.

    Returns
    -------
    Optional[RoleFilter]
        The arm's role map and excluded roles.
    """
    if not condition.is_role_ablation:
        return None
    return RoleFilter(get_role_map(condition.role_map), frozenset(condition.excluded_roles))


def condition_selector(condition: AblationCondition, count_tokens: TokenCounter) -> PolicySelector:
    """
    Build the selector a condition prescribes: order, budget, seed, vocabulary and roles.

    Parameters
    ----------
    condition : AblationCondition
        A registered condition.
    count_tokens : TokenCounter
        Encoded-length counter of the model's tokenizer.

    Returns
    -------
    PolicySelector
        Selector used by the runner and the inventories alike.
    """
    return PolicySelector(
        order_mode=condition.order_mode, count_tokens=count_tokens, budget=condition.token_budget,
        seed=condition.seed, filter_mode=condition.filter_mode,
        predicate_config=get_predicate_config(condition.predicate_config),
        role_filter=role_filter_for(condition), render_order=condition.render_order,
    )


def role_removal_summary(side: SideItems, role_filter: RoleFilter, count_tokens: TokenCounter) -> Dict[str, object]:
    """
    Count, per role, the items and item tokens a role filter removes from a side.

    Item tokens are the item's own encoded length without special tokens, so
    they approximate, not equal, the change in joined-text length.

    Parameters
    ----------
    side : SideItems
        Eligible items of the side.
    role_filter : RoleFilter
        The arm's role filter.
    count_tokens : TokenCounter
        Encoded-length counter, special tokens included.

    Returns
    -------
    Dict[str, object]
        ``{role: {"n_items", "n_entities", "item_tokens"}}`` for removed roles.
    """
    removed: Dict[str, Dict[str, int]] = {}
    for items in side.items:
        touched = set()
        for item in items:
            role = role_filter.role_map.role_of(item.predicate)
            if role in role_filter.excluded_roles:
                entry = removed.setdefault(role, {"n_items": 0, "n_entities": 0, "item_tokens": 0})
                entry["n_items"] += 1
                entry["item_tokens"] += count_tokens(item.text) - 2
                touched.add(role)
        for role in touched:
            removed[role]["n_entities"] += 1
    return removed


def check_role_coverage(context: PairContext, conditions: List[AblationCondition]) -> None:
    """
    Raise before any encoding if a role condition meets an unmapped predicate.

    Parameters
    ----------
    context : PairContext
        Items of both sides of the pair.
    conditions : List[AblationCondition]
        Conditions about to run on the pair.

    Raises
    ------
    UnmappedPredicateError
        If any item predicate has no role in a condition's role map.
    """
    predicates = {i.predicate for side in (context.source, context.candidate) for items in side.items for i in items}
    for name in sorted({c.role_map for c in conditions if c.is_role_ablation}):
        missing = get_role_map(name).unmapped(predicates)
        if missing:
            raise UnmappedPredicateError(f"{name} has no role for {missing}.")


def _select_and_encode(
    condition: AblationCondition, context: PairContext
) -> Tuple[Dict[str, List[Selection]], Dict[str, np.ndarray], Dict[str, object]]:
    """Select both sides under the condition, then embed them through the cache."""
    selector = condition_selector(condition, context.count_tokens)
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
        side: {
            **side_summary(selections[side], settings.max_seq_length),
            "budget": budget_summary(selections[side], condition.token_budget),
        }
        for side in selections
    }
    role_filter = role_filter_for(condition)
    if role_filter is not None:
        for side_name, side in ((_SOURCE, context.source), (_CANDIDATE, context.candidate)):
            summaries[side_name]["role_removal"] = role_removal_summary(side, role_filter, context.count_tokens)
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
        "selection_fingerprints": {
            _SOURCE: selection_fingerprints(context.source, selections[_SOURCE]),
            _CANDIDATE: selection_fingerprints(context.candidate, selections[_CANDIDATE]),
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


def is_reference_run(condition: AblationCondition, dataset_id: str) -> bool:
    """
    Tell whether a controlled condition on a dataset belongs to the #128 reference.

    Parameters
    ----------
    condition : AblationCondition
        A registered condition.
    dataset_id : str
        Dataset it would run on.

    Returns
    -------
    bool
        True for a controlled condition on D1 at the #128 budget.
    """
    return condition.is_controlled and dataset_id == "D1" and condition.token_budget == CONTROLLED_BUDGET


def _guard_role_results_dir(resolved: Path, conditions: List[AblationCondition]) -> None:
    """Keep #131 role conditions out of every historical tree, and other conditions out of theirs."""
    historical = {Path(d).resolve() for d in (_HISTORICAL_RESULTS_DIR, DEFAULT_RESULTS_DIR,
                                              CONTROLLED_RESULTS_DIR, VALIDATION_RESULTS_DIR)}
    if resolved in historical and any(c.is_role_ablation for c in conditions):
        raise ValueError(f"Predicate-role conditions must not be written to {resolved}; use {ROLE_RESULTS_DIR}.")
    if resolved == Path(ROLE_RESULTS_DIR).resolve() and not all(c.is_role_ablation for c in conditions):
        raise ValueError(f"{ROLE_RESULTS_DIR} holds only predicate-role (PR-*) conditions.")


def _guard_results_dir(
    results_dir: Path, conditions: List[AblationCondition], dataset_ids: List[str]
) -> None:
    """Refuse the historical tree, and controlled runs outside their own tree."""
    resolved = results_dir.resolve()
    _guard_role_results_dir(resolved, [c for c in conditions if any(d in c.datasets for d in dataset_ids)])
    runs = [(c, d) for c in conditions for d in dataset_ids if d in c.datasets and c.is_controlled]
    if resolved == Path(CONTROLLED_RESULTS_DIR).resolve() and not all(is_reference_run(c, d) for c, d in runs):
        raise ValueError(
            f"{CONTROLLED_RESULTS_DIR} holds only the #128 D1 runs at {CONTROLLED_BUDGET} tokens; "
            f"write validation runs to {VALIDATION_RESULTS_DIR}."
        )
    if resolved == Path(VALIDATION_RESULTS_DIR).resolve() and any(is_reference_run(c, d) for c, d in runs):
        raise ValueError(
            f"D1 controlled runs at {CONTROLLED_BUDGET} tokens are the #128 reference in "
            f"{CONTROLLED_RESULTS_DIR}; they must not be rerun into {VALIDATION_RESULTS_DIR}."
        )
    if resolved == _HISTORICAL_RESULTS_DIR.resolve():
        raise ValueError(
            f"Ablation results must not be written to {_HISTORICAL_RESULTS_DIR}; "
            f"use {DEFAULT_RESULTS_DIR}."
        )
    if resolved == _ISSUE_122_RESULTS_DIR.resolve() and any(c.is_controlled for c in conditions):
        raise ValueError(
            f"Controlled conditions must not be written beside the #122 results in "
            f"{_ISSUE_122_RESULTS_DIR}; use {CONTROLLED_RESULTS_DIR}."
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
    if pending:
        check_known_pairs(pair, settings, pending)
    return pending


def check_known_pairs(pair: AlignmentPair, settings: RunSettings, pending: List[AblationCondition]) -> None:
    """
    Refuse to add results beside stored ones scored with other known pairs.

    A stored result of the same pair and smoke restriction that this run
    keeps must record as many known (training) pairs as ``pair`` has, so
    results from before and after a change to them (D1's Bio-ML validation
    mappings, #137) are never compared within one directory.

    Parameters
    ----------
    pair : AlignmentPair
        Pair about to be run.
    settings : RunSettings
        Run settings supplying the results directory and smoke restriction.
    pending : List[AblationCondition]
        Conditions this run (re)writes; their stored results are replaced.

    Raises
    ------
    DataError
        If a kept result records a different number of known pairs.
    """
    rewritten = {condition.condition_id for condition in pending}
    for path in sorted(settings.results_dir.glob(f"*/{pair.dataset_id}/{pair.pair_name}_results.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        kept = stored["condition_id"] not in rewritten
        same_run = stored.get("ablation", {}).get("smoke_max_sources") == settings.max_sources
        if kept and same_run and stored["population"].get("n_train_refs", 0) != len(pair.train_refs):
            raise DataError(
                f"{path} was scored with {stored['population'].get('n_train_refs', 0)} known pairs, "
                f"this run with {len(pair.train_refs)}; re-run every condition with "
                "--force_recompute or use a new results directory."
            )


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
        check_role_coverage(context, pending)
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
        ``data_dir``, ``results_dir``, ``force_recompute`` and ``max_sources``,
        plus an optional ``allowed_device`` (a non-MPS device accepted
        explicitly; MPS is required otherwise).

    Returns
    -------
    List[Dict[str, object]]
        Result payloads written by the run.
    """
    settings_kwargs = dict(settings_kwargs)
    allowed_device = settings_kwargs.pop("allowed_device", None)
    conditions = _select_conditions(condition_ids)
    _guard_results_dir(Path(settings_kwargs["results_dir"]), conditions, dataset_ids)
    model, model_info = load_sentence_transformer(ABLATION_MODEL_KEY)
    model_info = {**model_info, **require_device(model_info, allowed_device)}
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
    parser.add_argument(
        "--allow_device", default=None, choices=("cpu", "cuda"),
        help="Accept this non-MPS device explicitly (recorded as device_override).",
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
        "allowed_device": args.allow_device,
    }
    run_ablation(args.datasets, args.conditions, settings_kwargs)


if __name__ == "__main__":
    cli_main()
