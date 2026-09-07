"""Deterministic character n-gram candidate generation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from kgsemembed.utils.errors import ConfigurationError


@dataclass(frozen=True)
class CandidatePair:
    source_uri: str
    target_uri: str
    score: float
    rank: int


def _normalize_text(text: str, strip_punctuation: bool) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if strip_punctuation:
        normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def _char_ngrams(text: str, n: int) -> list[str]:
    if n <= 0:
        raise ConfigurationError("candidates.n must be a positive integer")
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _cosine_score(source_tokens: list[str], target_tokens: list[str]) -> float:
    s_counts = Counter(source_tokens)
    t_counts = Counter(target_tokens)
    if not s_counts or not t_counts:
        return 0.0
    dot = sum(s_counts[t] * t_counts.get(t, 0) for t in s_counts)
    s_norm = math.sqrt(sum(v * v for v in s_counts.values()))
    t_norm = math.sqrt(sum(v * v for v in t_counts.values()))
    if s_norm == 0 or t_norm == 0:
        return 0.0
    return dot / (s_norm * t_norm)


def _jaccard_score(source_tokens: list[str], target_tokens: list[str]) -> float:
    s_set = set(source_tokens)
    t_set = set(target_tokens)
    union = s_set | t_set
    if not union:
        return 0.0
    return len(s_set & t_set) / len(union)


def _score_pair(source_tokens: list[str], target_tokens: list[str], metric: str) -> float:
    if metric == "cosine":
        return _cosine_score(source_tokens, target_tokens)
    if metric == "jaccard":
        return _jaccard_score(source_tokens, target_tokens)
    raise ConfigurationError(f"Unsupported candidates.metric: {metric}")


def _build_tokens(
    labels: dict[str, str],
    n: int,
    strip_punctuation: bool,
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    token_map: dict[str, list[str]] = {}
    aliases = aliases or {}
    for uri, label in labels.items():
        texts = [label]
        texts.extend(aliases.get(uri, []))
        all_tokens: list[str] = []
        for text in texts:
            normalized = _normalize_text(text, strip_punctuation=strip_punctuation)
            all_tokens.extend(_char_ngrams(normalized, n=n))
        token_map[uri] = all_tokens
    return token_map


def generate_candidates(
    source_labels: dict[str, str],
    target_labels: dict[str, str],
    n: int,
    metric: str,
    top_k: int,
    strip_punctuation: bool = True,
    source_aliases: dict[str, list[str]] | None = None,
    target_aliases: dict[str, list[str]] | None = None,
) -> list[CandidatePair]:
    if top_k <= 0:
        raise ConfigurationError("candidates.top_k must be a positive integer")

    source_tokens = _build_tokens(source_labels, n=n, strip_punctuation=strip_punctuation, aliases=source_aliases)
    target_tokens = _build_tokens(target_labels, n=n, strip_punctuation=strip_punctuation, aliases=target_aliases)

    pairs: list[CandidatePair] = []

    for source_uri in sorted(source_tokens.keys()):
        scored: list[tuple[str, float]] = []
        s_tokens = source_tokens[source_uri]
        for target_uri in sorted(target_tokens.keys()):
            t_tokens = target_tokens[target_uri]
            score = _score_pair(s_tokens, t_tokens, metric=metric)
            scored.append((target_uri, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        for rank, (target_uri, score) in enumerate(scored[:top_k], start=1):
            pairs.append(
                CandidatePair(
                    source_uri=source_uri,
                    target_uri=target_uri,
                    score=float(score),
                    rank=rank,
                )
            )

    return pairs


def build_candidate_table(candidates: list[CandidatePair]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_uri": c.source_uri,
                "target_uri": c.target_uri,
                "score": c.score,
                "rank": c.rank,
            }
            for c in candidates
        ],
        columns=["source_uri", "target_uri", "score", "rank"],
    )
