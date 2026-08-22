"""Predicate-Priority Adaptive Sampling (PPAS) for entity verbalisation.

Caps verbalised entity triple sets to fit within a model token budget by
selecting triples greedily in predicate-tier priority order.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Token budgets keyed by model key; every key of MODEL_REGISTRY must appear here
# ---------------------------------------------------------------------------
PPAS_BUDGETS: Dict[str, Optional[int]] = {
    "M1": 200,
    "M2": 420,
    "M2_uncapped": None,  # No cap -- controlled PPAS ablation twin of M2
    "M3": None,  # No cap -- long-context ablation model (bge-m3)
    "M4": 420,
    "M5": 420,
}

PPAS_TRIGGER_THRESHOLD: int = 20  # Apply PPAS when entity has more triples than this

# ---------------------------------------------------------------------------
# Predicate URI constants (full URIs for comparison with str(triple[1]))
# ---------------------------------------------------------------------------
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_OBO = "http://purl.obolibrary.org/obo/"
_OWL = "http://www.w3.org/2002/07/owl#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# ---------------------------------------------------------------------------
# Tier lists -- each inner list = one priority tier (tier 0 = highest)
# ---------------------------------------------------------------------------

CLASS_TIER_LIST: List[List[str]] = [
    [  # Tier 0: core lexical
        f"{_RDFS}label",
        f"{_SKOS}prefLabel",
        f"{_SKOS}altLabel",
        f"{_OBO}hasExactSynonym",
        f"{_OBO}hasSynonym",
        f"{_OBO}IAO_0000115",
        f"{_RDFS}comment",
        f"{_SKOS}definition",
    ],
    [  # Tier 1: structural type
        f"{_RDF}type",
        f"{_RDFS}subClassOf",
        f"{_RDFS}domain",
        f"{_RDFS}range",
    ],
    [  # Tier 2: equivalence
        f"{_OWL}equivalentClass",
        f"{_SKOS}exactMatch",
    ],
    [],  # Tier 3 -- populated dynamically per dataset at runtime
    [],  # Tier 4
    [],  # Tier 5
]

INSTANCE_TIER_LIST: List[List[str]] = [
    [  # Tier 0
        f"{_RDF}type",
        f"{_RDFS}label",
        f"{_SKOS}prefLabel",
    ],
    [],  # Tier 1 -- datatype properties / functional object properties
    [],  # Tier 2
    [  # Tier 3
        f"{_OWL}sameAs",
        f"{_SKOS}exactMatch",
    ],
]

PREDICATE_TIER_LIST: List[List[str]] = [
    [  # Tier 0
        f"{_RDFS}label",
        f"{_SKOS}prefLabel",
        f"{_RDFS}comment",
    ],
    [  # Tier 1
        f"{_RDFS}domain",
        f"{_RDFS}range",
    ],
    [  # Tier 2
        f"{_RDFS}subPropertyOf",
        f"{_OWL}inverseOf",
        f"{_OWL}equivalentProperty",
    ],
]


def estimate_tokens(text: str) -> int:
    """Approximate subword token count without a real tokenizer.

    Uses the formula ``ceil(len(text.split()) * 1.3)``.

    Parameters
    ----------
    text : str
        Text to estimate token count for.

    Returns
    -------
    int
        Estimated number of subword tokens.
    """
    return math.ceil(len(text.split()) * 1.3)


def untiered_predicates(
    triples: List[Tuple], tier_list: List[List[str]]
) -> List[str]:
    """Return predicates present in *triples* but absent from every tier.

    The result is intended to be appended to *tier_list* as a synthetic
    lowest-priority tier, so that :func:`ppas_sample` selects untiered
    triples within the token budget instead of discarding them.

    Predicate URIs are returned as :class:`str`, matching the tier-list
    convention, because :func:`ppas_sample` compares them against
    ``str(triple[1])``.

    Parameters
    ----------
    triples : list of tuple
        RDF triples as ``(subject, predicate, object)`` rdflib tuples.
    tier_list : list of list of str
        Predicate URI strings grouped by tier priority.

    Returns
    -------
    list of str
        Untiered predicate URIs, deduplicated in first-occurrence order.
    """
    tiered = {predicate for tier in tier_list for predicate in tier}
    untiered: List[str] = []
    for _subj, pred, _obj in triples:
        key = str(pred)
        if key not in tiered and key not in untiered:
            untiered.append(key)
    return untiered


def ppas_sample(
    triples: List[Tuple],
    tier_list: List[List[str]],
    token_budget: int,
    verbalise_triple_fn: Callable[[Tuple, Tuple, Tuple], str],
) -> List[Tuple]:
    """Select triples greedily in predicate-tier priority order.

    Parameters
    ----------
    triples : list of tuple
        RDF triples as ``(subject, predicate, object)`` rdflib tuples.
    tier_list : list of list of str
        Predicate URI strings grouped by tier priority; outer list is tiers
        in priority order (tier 0 = highest).
    token_budget : int
        Maximum tokens allowed for the verbalised output.
    verbalise_triple_fn : callable
        Function ``(subj, pred, obj) -> str`` used for cost estimation.

    Returns
    -------
    list of tuple
        Subset of *triples* that fit within *token_budget*.
    """
    if not tier_list or token_budget <= 0:
        return []

    selected: List[Tuple] = []
    tokens_used = 0

    for tier in tier_list:
        # Skip empty tiers gracefully
        if not tier:
            continue

        tier_predicate_set = set(tier)

        for triple in triples:
            pred_str = str(triple[1])
            if pred_str not in tier_predicate_set:
                continue

            cost = estimate_tokens(verbalise_triple_fn(*triple))
            if tokens_used + cost <= token_budget:
                selected.append(triple)
                tokens_used += cost
            # Do NOT break on budget miss -- skip and continue to next triple

        # Only break outer tier loop when budget is fully exhausted
        if tokens_used >= token_budget:
            break

    return selected


def should_apply_ppas(triples: List[Tuple], model_key: str) -> bool:
    """Decide whether PPAS should be applied for a given entity.

    Parameters
    ----------
    triples : list of tuple
        RDF triples for the entity.
    model_key : str
        Model key (e.g. ``"M1"``).

    Returns
    -------
    bool
        ``True`` if PPAS should be applied.
    """
    budget = PPAS_BUDGETS.get(model_key)
    if budget is None:
        return False
    return len(triples) > PPAS_TRIGGER_THRESHOLD
