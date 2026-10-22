"""What Figures 1 and 2 plot, shared by the plotting scripts and their claims (#149).

Kept free of plotting imports, so ``scripts/figure_claims.py`` can read the
figures' pairs and bars without loading matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BarSpec:
    """One model group on the x-axis and the condition that supplies it."""

    label: str
    condition_id: str
    strategy: str
    model_key: str
    colour: str


@dataclass(frozen=True)
class DatasetSpec:
    """One bar within each model group; ``not_run`` lists conditions never run on it."""

    dataset_id: str
    pair_name: str
    label: str
    alpha: float
    not_run: frozenset[str] = field(default_factory=frozenset)


SPARSE_PAIRS = frozenset({
    "cmt-conference", "cmt-confof", "cmt-edas", "cmt-ekaw",
    "cmt-iasted", "cmt-sigkdd",
    "confof-edas", "conference-edas", "edas-ekaw",
    "edas-iasted", "edas-sigkdd",
})
CONSTANT_STRATEGY = "V2+V8"
BARS = (
    BarSpec("M1\n(MiniLM)", "C18", CONSTANT_STRATEGY, "M1", "#7fb3d3"),
    BarSpec("M2\n(bge-large)", "C10", CONSTANT_STRATEGY, "M2", "#5b7fa6"),
    BarSpec("M3\n(bge-m3)", "C15", CONSTANT_STRATEGY, "M3", "#3a5f8a"),
    BarSpec("M4\n(BioLORD)", "C11", CONSTANT_STRATEGY, "M4", "#e07b54"),
    BarSpec("M5\n(stella)", "C13", CONSTANT_STRATEGY, "M5", "#9b59b6"),
    BarSpec("M4 V1\n(BioLORD,\nC17 ref)", "C17", "V1", "M4", "#c0392b"),
)
DATASETS = (
    DatasetSpec("D1", "d1_snomed_fma", "D1 (SNOMED-FMA)", 0.45),
    DatasetSpec("D2", "d2_anatomy", "D2 (Anatomy MA-NCI)", 1.0, frozenset({"C15"})),
)


def check_condition(result: dict, bar: BarSpec) -> None:
    """
    Raise if a result file was produced by a different strategy or model.

    Parameters
    ----------
    result : dict
        A parsed ``*_results.json``.
    bar : BarSpec
        Bar the result is plotted or tested under.

    Raises
    ------
    ValueError
        If the result's ``strategy`` or ``model_key`` differs from the bar's,
        or is missing.
    """
    found = (result.get("strategy"), result.get("model_key"))
    if found != (bar.strategy, bar.model_key):
        raise ValueError(
            f"{bar.condition_id} result has strategy/model {found}, "
            f"expected {(bar.strategy, bar.model_key)}"
        )
