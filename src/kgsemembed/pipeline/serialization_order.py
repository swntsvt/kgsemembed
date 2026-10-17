"""Runner for the text-serialization-order experiment (#134).

Reuses the #122/#131 ablation machinery (eligible items, selector, embedding
cache, ranking and evaluation) for the ``SO-*`` conditions of
:mod:`kgsemembed.pipeline.serialization_conditions`, with two additions:

* ``--population validation`` ranks only the validation population's sources
  (their candidates are the only ones encoded) and computes no test rank: the
  threshold and every metric come from validation sources.
* ``--population test`` runs the full pair exactly as #131 did, and refuses to
  start unless the freeze record is committed, pushed, on a clean tree,
  matches the current order definitions, rule and validation files, and no
  frozen implementation file changed since the validation run
  (:func:`verify_freeze`).  It refuses smoke subsets and existing test
  results written under another freeze or commit.

Results go to ``<results_dir>/<population>/<condition>/<dataset>/``; the
default root ``data/results_serialization/`` holds nothing else, and no
historical results tree accepts ``SO-*`` conditions.  Each run writes a result
JSON (main schema plus ``ablation`` and ``serialization`` blocks) and a sidecar
with per-reference-source ranks and the text and item-set fingerprints of every
entity.

Usage
-----
python -m kgsemembed.pipeline.serialization_order --population validation \\
    --datasets D1 D2 --data_dir data/ --results_dir data/results_serialization/
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kgsemembed.datasets import AlignmentPair, load_dataset
from kgsemembed.evaluation import order_selection
from kgsemembed.embeddings import EmbeddingEncoder, load_sentence_transformer
from kgsemembed.evaluation import RankedList
from kgsemembed.evaluation.metrics import compute_all_metrics
from kgsemembed.evaluation.population import build_populations, judgeable_ranked_lists, restrict_ranked_lists
from kgsemembed.pipeline import ppas_ablation as base
from kgsemembed.pipeline.ablation_conditions import ABLATION_MODEL_KEY, AblationCondition
from kgsemembed.pipeline.run_experiment import (
    _POPULATION_KEY,
    _build_ranked_lists,
    _evaluate_pair,
    _resolve_threshold,
    _seed_random_state,
)
from kgsemembed.pipeline.serialization_conditions import (
    LOGICAL_CONDITIONS,
    SERIALIZATION_CONDITIONS,
    get_serialization_condition,
)
from kgsemembed.verbalisation.selection import EligibleItem, PolicySelector, Selection, make_token_counter
from kgsemembed.verbalisation.serialization import SERIALIZATION_ORDERS, order_definitions_sha256

_LOGGER = logging.getLogger("kgsemembed.pipeline.serialization_order")

SERIALIZATION_RESULTS_DIR = "data/results_serialization/"
VALIDATION, TEST = "validation", "test"
POPULATIONS = (VALIDATION, TEST)
FREEZE_PATH = Path("artifacts/issue-134/serialization_freeze.json")
_TOP_K = 4
FROZEN_IMPLEMENTATION = ("src/kgsemembed", ":(exclude)src/kgsemembed/pipeline/serialization_order.py")
"""Pathspec that must be identical between the validation commit and the test
run: every module that loads data, builds items, selects, renders, encodes,
ranks or scores.  Only this runner is excluded, so its guards can be extended
after the freeze; its diff since the validation commit is reviewed instead."""
_FORBIDDEN_DIRS = (base._HISTORICAL_RESULTS_DIR, base.DEFAULT_RESULTS_DIR, base.CONTROLLED_RESULTS_DIR,
                   base.VALIDATION_RESULTS_DIR, base.ROLE_RESULTS_DIR)


class FreezeError(RuntimeError):
    """Raised when test evaluation is attempted without a verified freeze record."""


def item_set_fingerprint(items: List[EligibleItem]) -> str:
    """
    Fingerprint a selected item set independently of its render order.

    Item texts are unique per entity (duplicates are dropped when items are
    built), so ``(predicate URI, item text)`` identifies an item.

    Parameters
    ----------
    items : List[EligibleItem]
        Selected items, in any order.

    Returns
    -------
    str
        First 16 hex digits of the SHA-1 of the sorted keys as JSON.
    """
    keys = sorted([item.predicate, item.text] for item in items)
    return hashlib.sha1(json.dumps(keys, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def side_fingerprints(
    selector: PolicySelector, side: base.SideItems, selections: List[Selection]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return the text and item-set fingerprints of every entity on a side.

    Parameters
    ----------
    selector : PolicySelector
        The condition's selector, whose filters ``Selection.selected`` indexes.
    side : SideItems
        Entities of the side.
    selections : List[Selection]
        Their selections under the condition.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        URI to text fingerprint, and URI to item-set fingerprint.
    """
    texts, sets = {}, {}
    for uri, items, selection in zip(side.uris, side.items, selections):
        kept = selector.kept_items(items)
        texts[uri] = base._text_fingerprint(selection.text)
        sets[uri] = item_set_fingerprint([kept[i] for i in selection.selected])
    return texts, sets


def scored_pair(pair: AlignmentPair, population: str) -> AlignmentPair:
    """
    Restrict a pair to the sources a population stage ranks.

    Parameters
    ----------
    pair : AlignmentPair
        Full pair; its populations are always built from the full pair.
    population : str
        ``"validation"`` keeps the validation population's sources only;
        ``"test"`` keeps every source, as #131 did.

    Returns
    -------
    AlignmentPair
        The pair to encode and rank.
    """
    if population == TEST:
        return pair
    keep = build_populations(pair).val_sources
    return dataclasses.replace(pair, source_entities=[u for u in pair.source_entities if u in keep])


def validation_metrics(pair: AlignmentPair, ranked: List[RankedList]) -> Dict[str, object]:
    """
    Tune and score on the validation population only.

    Parameters
    ----------
    pair : AlignmentPair
        Full pair (references and populations).
    ranked : List[RankedList]
        Ranked lists of the validation population's sources.

    Returns
    -------
    Dict[str, object]
        Validation metrics with counts and a population record; no test field.
    """
    populations = build_populations(pair)
    lists = restrict_ranked_lists(judgeable_ranked_lists(pair, ranked), populations.val_sources)
    threshold = _resolve_threshold(lists, pair.val_refs)
    refs = [ref for ref in pair.val_refs if ref[0] in populations.val_sources]
    metrics = compute_all_metrics(lists, refs, threshold=threshold)
    metrics[_POPULATION_KEY] = {
        "population": VALIDATION, "n_val_refs": len(pair.val_refs), "n_val_population_refs": len(refs),
        "n_val_reference_sources": len({s for s, _ in refs}), "n_val_ranked_sources": len(lists),
        "n_val_unmatched_sources": populations.n_unmatched_val, "reference_complete": pair.reference_complete,
    }
    return metrics


def reference_records(
    refs: List[Tuple[str, str]], ranked: List[RankedList], fingerprints: Dict[str, Dict[str, Dict[str, str]]]
) -> List[Dict[str, object]]:
    """
    Describe every reference source of one population.

    Parameters
    ----------
    refs : List[Tuple[str, str]]
        The population's ``(source, target)`` references.
    ranked : List[RankedList]
        The condition's ranked lists.
    fingerprints : Dict[str, Dict[str, Dict[str, str]]]
        ``{"text"|"set": {side: {uri: fingerprint}}}``.

    Returns
    -------
    List[Dict[str, object]]
        Per source: gold targets, best gold rank (``None`` = reciprocal rank 0),
        the top candidates with scores, and the fingerprints of the source and
        of each gold target.
    """
    best = base._best_ranks(ranked, refs)
    tops = {r[0][0]: [[t, round(float(score), 6)] for _s, t, score in r[:_TOP_K]] for r in ranked if r}
    gold: Dict[str, List[str]] = {}
    for source, target in refs:
        gold.setdefault(source, []).append(target)
    return [
        {"source": source, "gold_targets": sorted(targets), "best_rank": best[source], "top": tops.get(source, []),
         "source_fp": [fingerprints[k][base._SOURCE].get(source) for k in ("text", "set")],
         "gold_fp": {t: [fingerprints[k][base._CANDIDATE].get(t) for k in ("text", "set")] for t in sorted(targets)}}
        for source, targets in sorted(gold.items())
    ]


def population_refs(pair: AlignmentPair, population: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Return the references whose sources each stage records.

    Parameters
    ----------
    pair : AlignmentPair
        Full pair.
    population : str
        Stage being run.

    Returns
    -------
    Dict[str, List[Tuple[str, str]]]
        ``validation`` always (restricted to the validation population), and
        ``test`` only in the test stage.
    """
    val_sources = build_populations(pair).val_sources
    refs = {VALIDATION: [ref for ref in pair.val_refs if ref[0] in val_sources]}
    if population == TEST:
        refs[TEST] = list(pair.test_refs)
    return refs


@dataclasses.dataclass(frozen=True)
class StageSettings:
    """
    Settings of one population stage on top of the ablation run settings.

    Attributes
    ----------
    run : base.RunSettings
        Ablation settings; ``results_dir`` is the stage directory.
    population : str
        ``"validation"`` or ``"test"``.
    freeze : Optional[Dict[str, object]]
        Verified freeze record (test stage), else ``None``.
    """

    run: base.RunSettings
    population: str
    freeze: Optional[Dict[str, object]] = None


def serialization_block(condition: AblationCondition, dataset_id: str, stage: StageSettings,
                        fingerprints: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, object]:
    """Record the population, orders, logical conditions and freeze behind one run."""
    logical = [c for c in LOGICAL_CONDITIONS if c.condition_id == condition.condition_id and c.dataset == dataset_id]
    rows = [[side, uri, fp] for side, fps in sorted(fingerprints["set"].items()) for uri, fp in sorted(fps.items())]
    freeze = stage.freeze or {}
    return {
        "population": stage.population,
        "selection_order": condition.order_mode,
        "render_order": condition.render_order,
        "logical_conditions": [c.name for c in logical],
        "order_definitions_sha256": order_definitions_sha256(),
        "item_set_sha256": hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest(),
        "freeze": {"sha256": freeze.get("_sha256"), "chosen": freeze.get("chosen"),
                   "commit": freeze.get("_commit")} if freeze else None,
    }


def run_condition(condition: AblationCondition, context: base.PairContext, full: AlignmentPair,
                  stage: StageSettings) -> Dict[str, object]:
    """
    Run one ``SO-*`` condition on one pair and write its result and sidecar.

    Parameters
    ----------
    condition : AblationCondition
        A #134 condition.
    context : base.PairContext
        Items, candidates and cache of the scored pair.
    full : AlignmentPair
        The full pair, for populations and references.
    stage : StageSettings
        Stage settings.

    Returns
    -------
    Dict[str, object]
        The written result payload.
    """
    selector = base.condition_selector(condition, context.count_tokens)
    selections, embeddings, runtime = base._select_and_encode(condition, context)
    index = {uri: i for i, uri in enumerate(context.candidate.uris)}
    ranked = _build_ranked_lists(context.pair, embeddings[base._SOURCE], context.candidates, index,
                                 embeddings[base._CANDIDATE])
    metrics = _evaluate_pair(full, ranked) if stage.population == TEST else validation_metrics(full, ranked)
    fingerprints = {"text": {}, "set": {}}
    for name, side in ((base._SOURCE, context.source), (base._CANDIDATE, context.candidate)):
        fingerprints["text"][name], fingerprints["set"][name] = side_fingerprints(selector, side, selections[name])
    summaries = {s: {**base.side_summary(selections[s], stage.run.max_seq_length),
                     "budget": base.budget_summary(selections[s], condition.token_budget),
                     "role_removal": base.role_removal_summary(side, base.role_filter_for(condition),
                                                               context.count_tokens)}
                 for s, side in ((base._SOURCE, context.source), (base._CANDIDATE, context.candidate))}
    uris = {base._SOURCE: context.source.uris, base._CANDIDATE: context.candidate.uris}
    summaries.update(selection_sha256=base.selection_digest(uris, selections), runtime=runtime)
    payload = base._result_payload(condition, context, metrics,
                                   base.ablation_block(condition, stage.run, summaries), stage.run)
    payload["serialization"] = serialization_block(condition, full.dataset_id, stage, fingerprints)
    refs = population_refs(full, stage.population)
    sidecar = {"condition_id": condition.condition_id, "population": stage.population,
               "reference_sources": {p: reference_records(r, ranked, fingerprints) for p, r in refs.items()},
               "selection_fingerprints": fingerprints["text"], "item_set_fingerprints": fingerprints["set"]}
    _write(condition, full, payload, sidecar, stage.run.results_dir)
    return payload


def _write(condition: AblationCondition, pair: AlignmentPair, payload: Dict[str, object],
           sidecar: Dict[str, object], results_dir: Path) -> None:
    """Write the result JSON and the sidecar of a run."""
    result_path, sidecar_path = base._result_paths(results_dir, condition.condition_id, pair)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command in the working directory without raising."""
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def verify_freeze(path: Path = FREEZE_PATH) -> Dict[str, object]:
    """
    Verify the freeze record before any test rank is computed.

    The record must exist, match the current order definitions and selection
    rule, be tracked and committed on a clean tree, be pushed to a remote
    branch, follow its validation commit, list validation files that are
    unchanged on disk, and no :data:`FROZEN_IMPLEMENTATION` file may differ
    from the validation commit.

    Parameters
    ----------
    path : Path
        Freeze record path.

    Returns
    -------
    Dict[str, object]
        The record, plus ``_sha256`` (file digest) and ``_commit`` (freezing commit).

    Raises
    ------
    FreezeError
        If any condition fails.
    """
    if not path.exists():
        raise FreezeError(f"No freeze record at {path}; run the validation stage and commit the freeze first.")
    record = json.loads(path.read_text(encoding="utf-8"))
    _check_definitions(record)
    commit = _check_committed(path, str(record.get("validation_commit")))
    _check_implementation(str(record["validation_commit"]))
    _check_validation_files(record.get("validation_files_sha256") or {})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**record, "_sha256": digest, "_commit": commit}


def _check_definitions(record: Dict[str, object]) -> None:
    """Refuse a record whose orders, rule constants or chosen order differ from the code."""
    if record.get("order_definitions_sha256") != order_definitions_sha256():
        raise FreezeError("The freeze record's order definitions differ from the current code.")
    rule = record.get("rule") or {}
    expected = {"replicates": order_selection.SELECTION_REPLICATES, "seed": order_selection.SELECTION_SEED,
                "precedence": list(order_selection.PRECEDENCE), "reference": order_selection.REFERENCE_ORDER}
    if any(rule.get(k) != v for k, v in expected.items()):
        raise FreezeError("The freeze record's selection rule differs from the current code.")
    if record.get("chosen") not in SERIALIZATION_ORDERS or record.get("chosen") == order_selection.REFERENCE_ORDER:
        raise FreezeError(f"The freeze record's chosen order {record.get('chosen')!r} is not a candidate.")


def _check_committed(path: Path, validation_commit: str) -> str:
    """Require a tracked, pushed record on a clean tree after its validation commit; return its commit."""
    if _git("ls-files", "--error-unmatch", str(path)).returncode != 0:
        raise FreezeError(f"{path} is not tracked by git.")
    if _git("status", "--porcelain").stdout.strip():
        raise FreezeError("The working tree is not clean; test evaluation needs a committed state.")
    commit = _git("log", "-n", "1", "--format=%H", "--", str(path)).stdout.strip()
    if not _git("branch", "-r", "--contains", commit).stdout.strip():
        raise FreezeError(f"The freezing commit {commit[:12]} is not on any remote branch; push it first.")
    if _git("merge-base", "--is-ancestor", validation_commit, commit).returncode != 0:
        raise FreezeError("The validation commit recorded in the freeze does not precede the freezing commit.")
    return commit


def _check_implementation(validation_commit: str) -> None:
    """Refuse when a frozen implementation file differs between the validation commit and HEAD."""
    diff = _git("diff", "--name-only", validation_commit, "HEAD", "--", *FROZEN_IMPLEMENTATION)
    if diff.returncode != 0 or diff.stdout.strip():
        changed = diff.stdout.split() or [diff.stderr.strip()]
        raise FreezeError(f"Implementation changed since the validation commit: {changed}.")


def _check_validation_files(digests: Dict[str, str]) -> None:
    """Refuse when a validation result the freeze was computed from is missing or changed."""
    if not digests:
        raise FreezeError("The freeze record lists no validation files.")
    for name, digest in sorted(digests.items()):
        path = Path(name)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise FreezeError(f"Validation file {name} is missing or differs from the freeze record.")


def guard_test_results(stage_dir: Path, freeze: Dict[str, object]) -> None:
    """
    Refuse existing test results written under another freeze or commit.

    Parameters
    ----------
    stage_dir : Path
        ``<results_dir>/test``.
    freeze : Dict[str, object]
        Verified freeze record.

    Raises
    ------
    FreezeError
        If a stored test result names a different freeze digest or commit,
        or was written from a dirty tree.
    """
    head = _git("rev-parse", "HEAD").stdout.strip()
    for path in sorted(stage_dir.glob("*/*/*_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        recorded = (result.get("serialization") or {}).get("freeze") or {}
        if recorded.get("sha256") != freeze["_sha256"] or result.get("git_commit") != head or result.get("git_dirty"):
            raise FreezeError(f"{path} was written under another freeze or commit; move it aside first.")


def guard_results_dir(results_dir: Path) -> None:
    """
    Refuse every historical results tree.

    Parameters
    ----------
    results_dir : Path
        Requested results root.

    Raises
    ------
    ValueError
        If it is a #122, #128, #131 or main-pipeline results tree.
    """
    resolved = results_dir.resolve()
    if any(resolved == Path(d).resolve() for d in _FORBIDDEN_DIRS):
        raise ValueError(f"#134 results must not be written to {resolved}; use {SERIALIZATION_RESULTS_DIR}.")


def run_dataset(dataset_id: str, conditions: List[AblationCondition], encoder: EmbeddingEncoder,
                stage: StageSettings) -> int:
    """
    Run every pending condition on a dataset's pairs; return how many ran.

    Parameters
    ----------
    dataset_id : str
        ``"D1"`` or ``"D2"``.
    conditions : List[AblationCondition]
        Requested conditions.
    encoder : EmbeddingEncoder
        Shared M2 encoder.
    stage : StageSettings
        Stage settings.

    Returns
    -------
    int
        Number of conditions run.
    """
    count_tokens = make_token_counter(encoder.model.tokenizer)
    applicable = [c for c in conditions if dataset_id in c.datasets]
    ran = 0
    for full in load_dataset(dataset_id, stage.run.data_dir):
        full = base.restrict_sources(full, stage.run.max_sources)
        pending = base._pending(applicable, full, stage.run)
        if not pending:
            continue
        context = base.build_pair_context(scored_pair(full, stage.population), encoder, count_tokens,
                                          stage.run.data_dir)
        base.check_role_coverage(context, pending)
        for condition in pending:
            _seed_random_state()
            _LOGGER.info("Running %s on %s (%s)", condition.condition_id, full.dataset_id, stage.population)
            run_condition(condition, context, full, stage)
            ran += 1
    return ran


def _select(condition_ids: Optional[List[str]]) -> List[AblationCondition]:
    """Return the requested #134 conditions in matrix order; all when ``None``."""
    if condition_ids is None:
        return list(SERIALIZATION_CONDITIONS)
    wanted = {get_serialization_condition(cid).condition_id for cid in condition_ids}
    return [c for c in SERIALIZATION_CONDITIONS if c.condition_id in wanted]


def verify_test_stage(args: argparse.Namespace, root: Path) -> Dict[str, object]:
    """
    Check every test-stage precondition before the model is loaded.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    root : Path
        Results root.

    Returns
    -------
    Dict[str, object]
        The verified freeze record.

    Raises
    ------
    FreezeError
        On a smoke subset, a failed freeze check or mixed test results.
    """
    if args.max_sources is not None:
        raise FreezeError("The test stage runs on full pairs only; --max_sources is refused.")
    freeze = verify_freeze(Path(args.freeze))
    guard_test_results(root / TEST, freeze)
    return freeze


def run_stage(args: argparse.Namespace) -> int:
    """
    Verify the stage's preconditions, load M2 once, and run the requested conditions.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Number of runs written.
    """
    root = Path(args.results_dir)
    guard_results_dir(root)
    freeze = verify_test_stage(args, root) if args.population == TEST else None
    model, model_info = load_sentence_transformer(ABLATION_MODEL_KEY)
    if not str(model_info["device"]).startswith("mps"):
        raise RuntimeError(f"Expected an MPS device, got {model_info['device']}.")
    encoder = EmbeddingEncoder(ABLATION_MODEL_KEY, model)
    run = base.RunSettings(
        data_dir=Path(args.data_dir), results_dir=root / args.population, force_recompute=args.force_recompute,
        model_info=model_info, tokenizer_id=model.tokenizer.name_or_path, max_seq_length=model.max_seq_length,
        encoder_settings=base._encoder_settings(encoder, model.max_seq_length), max_sources=args.max_sources,
    )
    stage = StageSettings(run, args.population, freeze)
    conditions = _select(args.conditions)
    return sum(run_dataset(d, conditions, encoder, stage) for d in args.datasets)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the #134 serialization-order experiment.")
    parser.add_argument("--population", required=True, choices=POPULATIONS, help="Stage to run.")
    parser.add_argument("--datasets", nargs="+", default=["D1", "D2"], help="Dataset IDs.")
    parser.add_argument("--conditions", nargs="*", default=None, help="SO-* condition IDs (default: all).")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    parser.add_argument("--results_dir", default=SERIALIZATION_RESULTS_DIR, help="Results root.")
    parser.add_argument("--freeze", default=str(FREEZE_PATH), help="Freeze record (test stage).")
    parser.add_argument("--force_recompute", action="store_true", help="Overwrite existing results.")
    parser.add_argument("--max_sources", type=int, default=None, help="Smoke run: stable crc32 source subset.")
    return parser


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run one population stage."""
    _seed_random_state()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ran = run_stage(_build_arg_parser().parse_args(argv))
    _LOGGER.info("Wrote %d runs.", ran)


if __name__ == "__main__":
    cli_main()
