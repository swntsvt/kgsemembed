"""Token budgets counted on the encoder's own input (#183).

A verbalisation budget is a number of tokens of the text the encoder receives:
the model's tokenizer at its pinned revision counts it, special tokens and the
role's prompt prefix (M5's source instruction) included.  The budget is
separate from the encoder's input limit (``ModelConfig.max_tokens``) and never
exceeds it.

Budgeted strategies select items in their own priority order while the
items' individually counted tokens fit, then check the joined text exactly
and drop items from the end until it fits (:func:`fit_items`).  Inside a
composite strategy, each component fits what the earlier components left
(``preceding``).  A whole text that still exceeds its budget, because an item
cannot be split (V7 never drops the annotation), is cut at the last whole
word that fits (:meth:`TokenBudget.cut`); :class:`BudgetedVerbaliser` does that
for every budgeted strategy and counts the texts it cut.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, List, Optional

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS

PART_SEPARATOR = "\n"
BUDGETED_COMPONENTS = frozenset({"V3", "V4", "V5", "V6", "V7"})
BUDGET_ENFORCEMENT = {
    "version": 1,
    "counting": "model tokenizer at the pinned revision, special tokens and prompt prefix included",
    "budget_rule": "min(PPAS_BUDGETS[model], ModelConfig.max_tokens)",
    "composites": "components filled in registered order; each fits what earlier ones left",
    "selection": "items in priority order while individual counts fit; joined text checked exactly, items dropped from the end",
    "overflow": "a text still over budget is cut at the last whole word that fits",
}
_WORD_END = re.compile(r"\S+")

TokenCounter = Callable[[str], int]


def budget_enforcement_digest() -> str:
    """
    Return the SHA-256 of :data:`BUDGET_ENFORCEMENT`, recorded in every result.

    Returns
    -------
    str
        Hex digest of the canonical JSON of the enforcement rules.
    """
    return hashlib.sha256(json.dumps(BUDGET_ENFORCEMENT, sort_keys=True).encode()).hexdigest()


def join_parts(parts: Iterable[str]) -> str:
    """
    Join non-empty stripped parts as a composite strategy joins its components.

    Parameters
    ----------
    parts : Iterable[str]
        Component texts, possibly empty.

    Returns
    -------
    str
        The stripped non-empty parts joined with :data:`PART_SEPARATOR`.
    """
    return PART_SEPARATOR.join(stripped for part in parts if (stripped := part.strip()))


@dataclass
class TokenBudget:
    """
    A token budget on the encoder input, with the counters that measure it.

    Attributes
    ----------
    tokens : int
        Maximum encoder-input tokens of the whole text.
    count : TokenCounter
        Encoder-input length of a text: prompt prefix and special tokens included.
    pieces : TokenCounter
        Length of a text fragment alone: no prefix, no special tokens.
    n_cut, tokens_cut : int
        Texts cut by :meth:`cut`, and the tokens removed from them.
    """

    tokens: int
    count: TokenCounter
    pieces: TokenCounter
    n_cut: int = field(default=0)
    tokens_cut: int = field(default=0)

    def fits(self, text: str, preceding: str = "") -> bool:
        """
        Report whether ``text`` after ``preceding`` fits the budget.

        Parameters
        ----------
        text : str
            Text of the component being fitted.
        preceding : str
            Text of the earlier components of a composite, or ``""``.

        Returns
        -------
        bool
            Whether the joined text's encoder input is within the budget.
        """
        return self.count(join_parts([preceding, text])) <= self.tokens

    def cut(self, text: str) -> str:
        """
        Cut ``text`` at the last whole word whose prefix fits, and count the cut.

        Parameters
        ----------
        text : str
            Text that may exceed the budget.

        Returns
        -------
        str
            ``text`` unchanged if it fits, else its longest fitting word prefix.
        """
        if self.fits(text):
            return text
        ends = [match.end() for match in _WORD_END.finditer(text)]
        low, high = 0, len(ends)
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if self.fits(text[: ends[middle - 1]]) else (low, middle - 1)
        kept = text[: ends[low - 1]] if low else ""
        self.n_cut += 1
        self.tokens_cut += self.count(text) - self.count(kept)
        return kept


def fit_items(items: List[str], separator: str, budget: Optional[TokenBudget], preceding: str = "") -> List[str]:
    """
    Keep items in order while they fit, skipping any that does not.

    Each item's tokens are counted alone (plus the separator); the joined text
    is then counted exactly and items are dropped from the end until it fits.

    Parameters
    ----------
    items : List[str]
        Rendered items in priority order.
    separator : str
        String the strategy joins its items with.
    budget : Optional[TokenBudget]
        The budget, or ``None`` to keep every item.
    preceding : str
        Text of the earlier components of a composite, or ``""``.

    Returns
    -------
    List[str]
        The kept items, in their original order.
    """
    if budget is None or budget.fits(separator.join(items), preceding):
        return items
    used, kept = budget.count(join_parts([preceding, "x"])) - budget.pieces("x"), []
    for item in items:
        cost = budget.pieces(item) + (budget.pieces(separator) if kept else 0)
        if used + cost <= budget.tokens:
            kept, used = kept + [item], used + cost
    while kept and not budget.fits(separator.join(kept), preceding):
        kept.pop()
    return kept


def fit_prefix(items: List[str], separator: str, budget: Optional[TokenBudget], preceding: str = "") -> List[str]:
    """
    Keep the longest leading run of items whose joined text fits.

    Parameters
    ----------
    items : List[str]
        Rendered items in order.
    separator : str
        String the strategy joins its items with.
    budget : Optional[TokenBudget]
        The budget, or ``None`` to keep every item.
    preceding : str
        Text of the earlier components of a composite, or ``""``.

    Returns
    -------
    List[str]
        The first items, stopping before the first that does not fit.
    """
    if budget is None:
        return items
    kept: List[str] = []
    for item in items:
        if not budget.fits(separator.join(kept + [item]), preceding):
            break
        kept.append(item)
    return kept


@lru_cache(maxsize=None)
def _tokenizer(model_key: str) -> object:
    from transformers import AutoTokenizer

    from kgsemembed.embeddings.models import get_model_config

    config = get_model_config(model_key)
    return AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)


def encoder_input_counters(model_key: str, role: str) -> tuple[TokenCounter, TokenCounter]:
    """
    Build the encoder-input and fragment counters of a model and role.

    Parameters
    ----------
    model_key : str
        Registered model key.
    role : str
        ``"source"`` or ``"candidate"``; selects the prompt prefix.

    Returns
    -------
    tuple[TokenCounter, TokenCounter]
        Encoder-input length (prefix and special tokens included) and the
        bare length of a fragment.
    """
    from kgsemembed.embeddings.models import get_model_config

    tokenizer = _tokenizer(model_key)
    config = get_model_config(model_key)
    prefix = {"source": config.query_prefix, "candidate": config.doc_prefix}[role]

    def count(text: str) -> int:
        prompted = prefix.format(text) if prefix else text
        return len(tokenizer(prompted, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])

    def pieces(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])

    return count, pieces


def budget_tokens(model_key: str) -> Optional[int]:
    """
    Return the model's verbalisation budget: its PPAS budget, capped at its input limit.

    Parameters
    ----------
    model_key : str
        Registered model key.

    Returns
    -------
    Optional[int]
        Budget in encoder-input tokens, or ``None`` when the model is uncapped.
    """
    from kgsemembed.embeddings.models import get_model_config

    budget = PPAS_BUDGETS.get(model_key)
    return None if budget is None else min(budget, get_model_config(model_key).max_tokens)


def token_budget(model_key: str, role: str) -> Optional[TokenBudget]:
    """
    Build the token budget of a model and role.

    Parameters
    ----------
    model_key : str
        Registered model key.
    role : str
        ``"source"`` or ``"candidate"``.

    Returns
    -------
    Optional[TokenBudget]
        The budget, or ``None`` when the model is uncapped.
    """
    tokens = budget_tokens(model_key)
    if tokens is None:
        return None
    count, pieces = encoder_input_counters(model_key, role)
    return TokenBudget(tokens, count, pieces)


def strategy_is_budgeted(strategy_name: str, model_key: str) -> bool:
    """
    Report whether a strategy's text is held to a token budget under a model.

    Parameters
    ----------
    strategy_name : str
        Registered strategy, possibly composite.
    model_key : str
        Registered model key.

    Returns
    -------
    bool
        Whether a component is budgeted and the model has a budget.
    """
    return PPAS_BUDGETS.get(model_key) is not None and bool(set(strategy_name.split("+")) & BUDGETED_COMPONENTS)


class BudgetedVerbaliser(VerbaliserBase):
    """
    Hold a budgeted strategy's whole text to its budget, cutting any overflow.

    Parameters
    ----------
    inner : VerbaliserBase
        The strategy, whose components already fit their own budgets.
    budget : TokenBudget
        The strategy's budget; it counts the texts this class cuts.
    """

    def __init__(self, inner: VerbaliserBase, budget: TokenBudget) -> None:
        self.inner = inner
        self.budget = budget

    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        """
        Verbalise with the inner strategy and cut the result to the budget.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.

        Returns
        -------
        str
            The inner text, cut at the last whole word that fits if needed.
        """
        return self.budget.cut(self.inner.verbalise(graph, entity_uri, entity_type))
