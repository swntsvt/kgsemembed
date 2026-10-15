"""Budgeted triple selection for the controlled PPAS ablation (#122).

Every selection policy draws from the same eligible item set, renders it the
same way, and enforces the same budget, so the policy is the only difference
between capped representations:

* An entity's eligible items are its structured key-value pairs (V6
  rendering), built from :meth:`VerbaliserBase._collect_triples`.  OpenEA's
  inlined literal datatype suffix is stripped, items that render empty are
  dropped, and exact duplicate renderings are kept once.  Items stay in the
  canonical lexicographic ``(predicate URI, object term)`` order.
* The budget counts the final encoded input of the joined text, special tokens
  included, with the embedding model's own tokenizer.  An entity whose full
  text fits is never sampled, so its text is identical under every policy.
* Otherwise items are visited in policy order and each is kept when the joined
  text still fits; an item that does not fit is skipped, never ends selection.
* Selected items are rendered in canonical order, so a policy changes which
  items appear but not the order in which they are presented.
"""

import hashlib
import json
import random
import zlib
from dataclasses import dataclass
from typing import Callable, Optional

from rdflib import Graph, Literal, URIRef

from kgsemembed.candidates.ngram import _literal_text
from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
)

FULL = "full"
PPAS = "ppas"
DETERMINISTIC = "deterministic"
RANDOM = "random"
SELECTION_POLICIES = (FULL, PPAS, DETERMINISTIC, RANDOM)
CAPPED_POLICIES = (PPAS, DETERMINISTIC, RANDOM)

ITEM_SEPARATOR = " | "
DETERMINISTIC_ORDER = "lexicographic (predicate URI, object term)"

TokenCounter = Callable[[str], int]

_TIER_LISTS: dict[str, list[list[str]]] = {
    "class": CLASS_TIER_LIST,
    "instance": INSTANCE_TIER_LIST,
    "predicate": PREDICATE_TIER_LIST,
}


@dataclass(frozen=True)
class EligibleItem:
    """
    One serialisable triple of an entity.

    Attributes
    ----------
    predicate : str
        Predicate URI, used for PPAS tier lookup.
    text : str
        Rendered ``"key: value"`` text.
    """

    predicate: str
    text: str


@dataclass(frozen=True)
class Selection:
    """
    The representation a policy produced for one entity.

    Attributes
    ----------
    text : str
        Selected items joined in canonical order.
    selected : tuple[int, ...]
        Canonical indices of the selected items.
    n_eligible : int
        Number of eligible items.
    n_tokens : int
        Encoded length of ``text``, special tokens included.
    sampling_invoked : bool
        Whether the full text exceeded the budget, so the policy chose items.
    n_oversized : int
        Eligible items whose own text exceeds the budget.
    """

    text: str
    selected: tuple[int, ...]
    n_eligible: int
    n_tokens: int
    sampling_invoked: bool
    n_oversized: int


def tier_config_digest() -> str:
    """
    Fingerprint the PPAS predicate-tier configuration.

    Returns
    -------
    str
        SHA-256 hex digest of the class, instance and predicate tier lists.
    """
    encoded = json.dumps(_TIER_LISTS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tier_list_for(entity_type: str) -> list[list[str]]:
    """
    Return the PPAS tier list for an entity type.

    Parameters
    ----------
    entity_type : str
        One of ``"class"``, ``"instance"``, ``"predicate"``.

    Returns
    -------
    list[list[str]]
        Predicate URIs grouped by tier, highest priority first.

    Raises
    ------
    ValueError
        If ``entity_type`` is not supported.
    """
    if entity_type not in _TIER_LISTS:
        raise ValueError(
            f"Unsupported entity type: {entity_type!r}. Expected one of "
            f"{sorted(_TIER_LISTS)}."
        )
    return _TIER_LISTS[entity_type]


def ppas_order(items: list[EligibleItem], tier_list: list[list[str]]) -> list[int]:
    """
    Order item indices by PPAS tier, untiered predicates last.

    Within a tier, and among untiered items, canonical order is kept, matching
    :func:`kgsemembed.verbalisation.ppas.ppas_sample` with
    :func:`~kgsemembed.verbalisation.ppas.untiered_predicates` appended.

    Parameters
    ----------
    items : list[EligibleItem]
        Eligible items in canonical order.
    tier_list : list[list[str]]
        Predicate URIs grouped by tier, highest priority first.

    Returns
    -------
    list[int]
        Item indices in PPAS priority order.
    """
    rank: dict[str, int] = {}
    for tier_index, tier in enumerate(tier_list):
        for predicate in tier:
            rank.setdefault(predicate, tier_index)
    lowest = len(tier_list)
    return sorted(
        range(len(items)), key=lambda i: (rank.get(items[i].predicate, lowest), i)
    )


def deterministic_order(items: list[EligibleItem]) -> list[int]:
    """
    Order item indices canonically, without consulting any predicate tier.

    Parameters
    ----------
    items : list[EligibleItem]
        Eligible items in canonical order.

    Returns
    -------
    list[int]
        ``0 .. len(items) - 1``: lexicographic ``(predicate URI, object)``.
    """
    return list(range(len(items)))


def random_order(n_items: int, seed: int, entity_uri: str) -> list[int]:
    """
    Shuffle item indices with an RNG derived from ``seed`` and the entity.

    ``zlib.crc32`` rather than ``hash()`` keeps the stream stable across
    processes, and per-entity seeding makes it independent of visiting order.

    Parameters
    ----------
    n_items : int
        Number of eligible items.
    seed : int
        Experiment-level random seed.
    entity_uri : str
        Entity whose items are shuffled.

    Returns
    -------
    list[int]
        A seeded permutation of ``0 .. n_items - 1``.
    """
    digest = zlib.crc32(entity_uri.encode("utf-8"))
    order = list(range(n_items))
    random.Random(seed * 2**32 + digest).shuffle(order)
    return order


def join_items(texts: list[str], indices: list[int]) -> str:
    """
    Join the selected item texts in canonical order.

    Parameters
    ----------
    texts : list[str]
        Item texts in canonical order.
    indices : list[int]
        Selected canonical indices, in any order.

    Returns
    -------
    str
        The texts at ``indices``, sorted, joined with :data:`ITEM_SEPARATOR`.
    """
    return ITEM_SEPARATOR.join(texts[i] for i in sorted(indices))


def select_within_budget(
    texts: list[str], order: list[int], budget: int, count_tokens: TokenCounter
) -> list[int]:
    """
    Greedily keep items, in ``order``, whose addition keeps the text in budget.

    The joined text is re-tokenised for every candidate addition, so the budget
    holds for the exact final input rather than for a sum of per-item costs.

    Parameters
    ----------
    texts : list[str]
        Item texts in canonical order.
    order : list[int]
        Indices in the policy's priority order.
    budget : int
        Maximum encoded length of the joined text.
    count_tokens : TokenCounter
        Encoded length of a text, special tokens included.

    Returns
    -------
    list[int]
        Selected indices in canonical order.
    """
    chosen: list[int] = []
    for index in order:
        if count_tokens(join_items(texts, chosen + [index])) <= budget:
            chosen.append(index)
    return sorted(chosen)


class PolicySelector(VerbaliserBase):
    """
    Build an entity's eligible items and select them under one policy.

    Parameters
    ----------
    policy : str
        One of :data:`SELECTION_POLICIES`.
    count_tokens : TokenCounter
        Encoded length of a text under the embedding model's tokenizer.
    budget : Optional[int]
        Maximum encoded length; ``None`` only for the ``"full"`` policy.
    seed : Optional[int]
        Random seed; required by, and only accepted for, ``"random"``.
    """

    def __init__(
        self,
        policy: str,
        count_tokens: TokenCounter,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        _validate_policy(policy, budget, seed)
        self.policy = policy
        self.count_tokens = count_tokens
        self.budget = budget
        self.seed = seed

    def eligible_items(self, graph: Graph, entity_uri: URIRef) -> list[EligibleItem]:
        """
        Return an entity's distinct, non-empty rendered items, canonically ordered.

        Parameters
        ----------
        graph : Graph
            Graph holding the entity's triples.
        entity_uri : URIRef
            Entity to render.

        Returns
        -------
        list[EligibleItem]
            Eligible items shared by every selection policy.
        """
        items: list[EligibleItem] = []
        seen: set[str] = set()
        for _subj, pred, obj in self._collect_triples(graph, entity_uri):
            text = self._item_text(graph, pred, obj)
            if text and text not in seen:
                seen.add(text)
                items.append(EligibleItem(str(pred), text))
        return items

    def _item_text(self, graph: Graph, pred: URIRef, obj: object) -> str:
        """Render one triple, stripping an inlined literal datatype suffix."""
        if isinstance(obj, Literal):
            obj = Literal(_literal_text(obj))
        return self._format_pair(graph, pred, obj)

    def select(
        self, items: list[EligibleItem], entity_uri: str, entity_type: str
    ) -> Selection:
        """
        Select an entity's items under this policy and budget.

        Parameters
        ----------
        items : list[EligibleItem]
            The entity's eligible items, from :meth:`eligible_items`.
        entity_uri : str
            Entity URI, which seeds the random policy.
        entity_type : str
            Entity type, which picks the PPAS tier list.

        Returns
        -------
        Selection
            The selected representation and its accounting.
        """
        texts = [item.text for item in items]
        everything = list(range(len(items)))
        full_tokens = self.count_tokens(join_items(texts, everything))
        if self.budget is None or full_tokens <= self.budget:
            return Selection(
                join_items(texts, everything), tuple(everything), len(items),
                full_tokens, False, 0,
            )
        return self._sampled(items, entity_uri, entity_type)

    def _sampled(
        self, items: list[EligibleItem], entity_uri: str, entity_type: str
    ) -> Selection:
        """Select within budget an entity whose full text exceeds it."""
        texts = [item.text for item in items]
        order = self._order(items, entity_uri, entity_type)
        chosen = select_within_budget(texts, order, self.budget, self.count_tokens)
        text = join_items(texts, chosen)
        n_tokens = self.count_tokens(text)
        if n_tokens > self.budget:
            raise AssertionError(
                f"Selection for {entity_uri} uses {n_tokens} tokens, budget {self.budget}."
            )
        oversized = sum(self.count_tokens(t) > self.budget for t in texts)
        return Selection(text, tuple(chosen), len(items), n_tokens, True, oversized)

    def _order(
        self, items: list[EligibleItem], entity_uri: str, entity_type: str
    ) -> list[int]:
        """Return the item visiting order of this policy."""
        if self.policy == PPAS:
            return ppas_order(items, tier_list_for(entity_type))
        if self.policy == RANDOM:
            return random_order(len(items), self.seed, entity_uri)
        return deterministic_order(items)

    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        """
        Return the entity's selected representation.

        Parameters
        ----------
        graph : Graph
            Graph holding the entity's triples.
        entity_uri : URIRef
            Entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.

        Returns
        -------
        str
            Selected items joined in canonical order; empty when none qualify.
        """
        items = self.eligible_items(graph, entity_uri)
        return self.select(items, str(entity_uri), entity_type).text


def _validate_policy(policy: str, budget: Optional[int], seed: Optional[int]) -> None:
    """Raise if the policy, budget and seed do not form a valid combination."""
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"Unknown selection policy {policy!r}; expected {SELECTION_POLICIES}.")
    if (policy == FULL) != (budget is None):
        raise ValueError(f"Policy {policy!r} requires budget to be None only for 'full'.")
    if (policy == RANDOM) != (seed is not None):
        raise ValueError(f"Policy {policy!r} takes a seed only when it is 'random'.")
    if budget is not None and budget <= 0:
        raise ValueError(f"Token budget must be positive, got {budget}.")


def make_token_counter(tokenizer: object) -> TokenCounter:
    """
    Wrap a Hugging Face tokenizer as an encoded-length counter.

    Special tokens are included and nothing is truncated, so the count is the
    length the model would see before its own ``max_seq_length`` cut.

    Parameters
    ----------
    tokenizer : object
        A ``transformers`` tokenizer, e.g. ``SentenceTransformer.tokenizer``.

    Returns
    -------
    TokenCounter
        Function mapping a text to its encoded length.
    """

    def count_tokens(text: str) -> int:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)
        return len(encoded["input_ids"])

    return count_tokens
