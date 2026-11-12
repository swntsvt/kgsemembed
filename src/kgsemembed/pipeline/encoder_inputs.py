"""What the encoder received for one condition and pair, and the guards on it (#183).

Every text is counted as the encoder receives it: the model's tokenizer, the
role's prompt prefix (M5's source instruction) and special tokens included.
The result records, per role, how many inputs the encoder truncated at its
limit and the tokens discarded, and how many texts the verbalisation budget
cut; the per-source sidecar records every entity's length.

Two guards make a violation an error rather than a silent truncation: the
loaded model must use the registered input limit, and a budgeted condition's
texts must all fit the budget (so the encoder truncates none of them).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from kgsemembed.embeddings.encoder import EmbeddingEncoder
from kgsemembed.embeddings.models import get_model_config
from kgsemembed.pipeline.conditions import ExperimentCondition
from kgsemembed.utils.errors import PipelineStageError
from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.budget import (
    BudgetedVerbaliser,
    budget_enforcement_digest,
    budget_tokens,
    strategy_is_budgeted,
)

RoleInputs = Tuple[VerbaliserBase, List[str], List[str]]


class EncoderInputError(PipelineStageError):
    """The encoder's input broke the registered limit or a condition's budget."""


def condition_budget(condition: ExperimentCondition) -> Optional[int]:
    """
    Return the token budget a condition's texts are held to.

    Parameters
    ----------
    condition : ExperimentCondition
        The condition.

    Returns
    -------
    Optional[int]
        The model's budget for a budgeted strategy, else ``None``.
    """
    if not strategy_is_budgeted(condition.strategy_name, condition.model_key):
        return None
    return budget_tokens(condition.model_key)


def length_summary(lengths: List[int], limit: int, budget: Optional[int]) -> Dict[str, int]:
    """
    Summarise the encoder-input lengths of one role.

    Parameters
    ----------
    lengths : List[int]
        Encoder-input length of every text.
    limit : int
        The model's input limit.
    budget : Optional[int]
        The condition's budget, or ``None``.

    Returns
    -------
    Dict[str, int]
        ``n_inputs``, ``n_truncated``, ``tokens_discarded``, ``max_length``
        and ``n_over_budget`` (0 without a budget).
    """
    return {
        "n_inputs": len(lengths),
        "n_truncated": sum(length > limit for length in lengths),
        "tokens_discarded": sum(max(0, length - limit) for length in lengths),
        "max_length": max(lengths, default=0),
        "n_over_budget": 0 if budget is None else sum(length > budget for length in lengths),
    }


def budget_cuts(verbaliser: VerbaliserBase) -> Dict[str, int]:
    """
    Return how many texts a verbaliser cut to its budget, and the tokens cut.

    Parameters
    ----------
    verbaliser : VerbaliserBase
        A verbaliser after use; only a :class:`BudgetedVerbaliser` cuts.

    Returns
    -------
    Dict[str, int]
        ``n_budget_cut`` and ``budget_tokens_cut``.
    """
    if not isinstance(verbaliser, BudgetedVerbaliser):
        return {"n_budget_cut": 0, "budget_tokens_cut": 0}
    return {"n_budget_cut": verbaliser.budget.n_cut, "budget_tokens_cut": verbaliser.budget.tokens_cut}


def check_limit(condition: ExperimentCondition, encoder: EmbeddingEncoder) -> int:
    """
    Return the loaded model's input limit, refusing one other than the registered.

    Parameters
    ----------
    condition : ExperimentCondition
        The condition run.
    encoder : EmbeddingEncoder
        Encoder of the loaded model.

    Returns
    -------
    int
        The input limit the model encodes at.

    Raises
    ------
    EncoderInputError
        If the model's ``max_seq_length`` differs from the registry's ``max_tokens``.
    """
    limit, registered = encoder.model.max_seq_length, get_model_config(condition.model_key).max_tokens
    if limit != registered:
        raise EncoderInputError(f"{condition.condition_id}: model encodes at {limit}, registry says {registered}")
    return limit


def check_budget(condition: ExperimentCondition, summaries: Dict[str, Dict[str, int]]) -> None:
    """
    Refuse a budgeted condition whose texts exceed the budget or are truncated.

    Parameters
    ----------
    condition : ExperimentCondition
        The condition run.
    summaries : Dict[str, Dict[str, int]]
        :func:`length_summary` of each role.

    Raises
    ------
    EncoderInputError
        If the condition has a budget and any text exceeds it or the limit.
    """
    if condition_budget(condition) is None:
        return
    broken = {role: s for role, s in summaries.items() if s["n_over_budget"] or s["n_truncated"]}
    if broken:
        raise EncoderInputError(f"{condition.condition_id}: budgeted texts exceed the budget or limit: {broken}")


def encoder_input_record(
    condition: ExperimentCondition, encoder: EmbeddingEncoder, roles: Dict[str, RoleInputs]
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Measure every text the encoder receives, check it, and build the records.

    Parameters
    ----------
    condition : ExperimentCondition
        The condition run.
    encoder : EmbeddingEncoder
        Encoder of the loaded model.
    roles : Dict[str, RoleInputs]
        Per role, the verbaliser used, the entity URIs and their texts.

    Returns
    -------
    Tuple[Dict[str, object], Dict[str, object]]
        The result's ``encoder_input`` block and the sidecar's ``inputs``
        block (per role, ``{uri: [length, tokens_discarded]}``).

    Raises
    ------
    EncoderInputError
        From :func:`check_limit` or :func:`check_budget`.
    """
    limit, budget = check_limit(condition, encoder), condition_budget(condition)
    lengths = {role: encoder.input_lengths(texts, role) for role, (_, _, texts) in roles.items()}
    summaries = {role: length_summary(lengths[role], limit, budget) for role in roles}
    check_budget(condition, summaries)
    overhead = {role: encoder.input_lengths([""], role)[0] for role in roles}
    record = {
        "max_seq_length": limit, "token_budget": budget, "budget_enforcement_sha256": budget_enforcement_digest(),
        **{role: {**summaries[role], **budget_cuts(verbaliser), "overhead_tokens": overhead[role]}
           for role, (verbaliser, _, _) in roles.items()},
    }
    inputs = {"max_seq_length": limit, "token_budget": budget, "overhead_tokens": overhead,
              **{role: {uri: [n, max(0, n - limit)] for uri, n in zip(uris, lengths[role])}
                 for role, (_, uris, _) in roles.items()}}
    return record, inputs
