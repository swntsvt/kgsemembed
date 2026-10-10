"""Statistical significance testing for Phase 2 experiment results.

This module compares experimental conditions using the Wilcoxon signed-rank
test with Bonferroni correction applied independently within each ablation
group.  It operates exclusively on previously generated result files under the
``data/results/{condition_id}/{dataset_id}/{pair_name}_results.json`` convention
written by :mod:`kgsemembed.pipeline.run_experiment`; it never reruns
experiments, modifies results, or produces plots.

Experimental unit
-----------------
The independent unit of every test is a dataset column: D1, D2, D3,
D4_schema, D4_instance, and D5, as in the main results table.  The 21 D3
alignment pairs are not independent observations — the Conference track has
seven ontologies and each one takes part in six of the pairs — so they are
averaged into one D3 value before testing.  Pooling them as 21 observations
would let one correlated track dominate the sample and overstate the evidence.
``"D4"`` in a comparison expands to its schema and instance columns, which
score disjoint reference sets.

With at most six units, the smallest attainable two-sided p-value is
``2 / 2**n``: ``0.0625`` for five units and ``0.03125`` for six.  Each result
reports that floor and flags a comparison whose floor exceeds its corrected
alpha as ``underpowered``, since no outcome of such a test could be called
significant.  Pair-level win/tie/loss counts are reported alongside as a
description of the data, not as an inferential test.

Within each comparison, observations are paired by ``AlignmentPair`` name, so
repeated executions produce identical paired samples regardless of filesystem
traversal order.  Only results written under the current
:data:`~kgsemembed.evaluation.metrics.EVALUATION_PROTOCOL_VERSION` are
accepted.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import PermutationMethod, rankdata, wilcoxon

from kgsemembed.evaluation.metrics import EVALUATION_PROTOCOL_VERSION
from kgsemembed.utils.errors import DataError

_RESULT_SUFFIX = "_results.json"
_MIN_PAIRED_OBSERVATIONS = 5
_DEFAULT_ALPHA = 0.05
_DEFAULT_RESULTS_DIR = "data/results/"
_PERMUTATION_SEED = 42
_DIFFERENCE_DECIMALS = 12
_SUB_DATASETS: Dict[str, Tuple[str, ...]] = {"D4": ("D4_schema", "D4_instance")}

_TABLE_COLUMNS: Tuple[str, ...] = (
    "Condition A",
    "Condition B",
    "n",
    "pairs (W/T/L)",
    "p-value",
    "min p",
    "Corrected sig.",
    "delta-F1",
    "effect_size_r",
    "Warning",
)
_TABLE_HEADER = "| " + " | ".join(_TABLE_COLUMNS) + " |"
_TABLE_SEPARATOR = "| " + " | ".join(["---"] * len(_TABLE_COLUMNS)) + " |"
_NO_WARNING_CELL = "-"
_NOT_TESTED_CELL = "n/a"

_RELIABILITY_WARNINGS = (RuntimeWarning, UserWarning)


@dataclass(frozen=True)
class _Comparison:
    """A single canonical comparison between two conditions over datasets."""

    condition_a: str
    condition_b: str
    datasets: Tuple[str, ...]


_GROUP_COMPARISONS: Dict[str, Tuple[_Comparison, ...]] = {
    "A": (
        _Comparison("C1", "C2", ("D1", "D2", "D3", "D4", "D5")),
        _Comparison("C1", "C17", ("D1", "D2")),
        _Comparison("C2", "C17", ("D1", "D2")),
    ),
    "B": (
        _Comparison("C3", "C10", ("D1", "D2", "D3", "D4", "D5")),
        _Comparison("C9", "C10", ("D1", "D2", "D3", "D4", "D5")),
        _Comparison("C12", "C1", ("D3", "D4")),
        _Comparison("C18", "C10", ("D1", "D2", "D3", "D4", "D5")),
    ),
    "C": (
        _Comparison("C10", "C13", ("D1", "D2", "D3", "D4", "D5")),
        _Comparison("C11", "C13", ("D1", "D2")),
        _Comparison("C6", "C16", ("D1", "D2")),
    ),
    "D": (
        _Comparison("C5", "C14", ("D5",)),
        _Comparison("C10", "C15", ("D1",)),
    ),
    "E": (
        _Comparison("C20", "C10", ("D1", "D2", "D3", "D4", "D5")),
    ),
}


def _result_dir(results_dir: str, condition_id: str, dataset_id: str) -> Path:
    """Return the directory holding a condition's result files for a dataset."""
    return Path(results_dir) / condition_id / dataset_id


def _load_result_file(path: Path) -> dict:
    """
    Parse a single result JSON file.

    Parameters
    ----------
    path : Path
        Location of the result JSON file.

    Returns
    -------
    dict
        The parsed result payload.

    Raises
    ------
    DataError
        If the file does not contain valid JSON.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Malformed result JSON at {path}: {exc}") from exc


def _extract_pair_name(payload: dict, path: Path) -> str:
    """Return the ``pair_name`` recorded in a result payload."""
    pair_name = payload.get("pair_name")
    if not isinstance(pair_name, str) or not pair_name:
        raise DataError(f"Result file {path} is missing a valid 'pair_name'.")
    return pair_name


def _extract_f1(payload: dict, pair_name: str) -> float:
    """Return the F1 metric recorded for an alignment pair."""
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or "f1" not in metrics:
        raise DataError(f"Result for pair {pair_name!r} is missing the F1 metric.")
    try:
        return float(metrics["f1"])
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"Result for pair {pair_name!r} has a non-numeric F1 metric: "
            f"{metrics['f1']!r}."
        ) from exc


def _require_current_protocol(payload: dict, pair_name: str) -> None:
    """
    Reject a result computed under an older evaluation protocol.

    Parameters
    ----------
    payload : dict
        Parsed result payload.
    pair_name : str
        Alignment pair the payload belongs to, used in the error message.

    Raises
    ------
    DataError
        If ``evaluation_protocol`` is absent or differs from the current one.
    """
    protocol = payload.get("evaluation_protocol")
    if protocol != EVALUATION_PROTOCOL_VERSION:
        raise DataError(
            f"Result for pair {pair_name!r} uses evaluation protocol {protocol!r}, "
            f"expected {EVALUATION_PROTOCOL_VERSION}; regenerate it with "
            "--force_recompute before testing."
        )


def load_results_for_condition(
    condition_id: str,
    dataset_id: str,
    results_dir: str = _DEFAULT_RESULTS_DIR,
) -> Optional[dict]:
    """
    Load every alignment-pair result for a condition and dataset directory.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    dataset_id : str
        Result directory name, e.g. ``"D3"`` or ``"D4_schema"``.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    Optional[dict]
        Mapping from ``pair_name`` to the parsed result payload, ordered by
        ``pair_name``, or ``None`` when no result files exist.

    Raises
    ------
    DataError
        If a result file is malformed, lacks a ``pair_name``, or a
        ``pair_name`` is duplicated within the dataset.
    """
    directory = _result_dir(results_dir, condition_id, dataset_id)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(f"*{_RESULT_SUFFIX}"))
    if not files:
        return None
    results: Dict[str, dict] = {}
    for path in files:
        payload = _load_result_file(path)
        pair_name = _extract_pair_name(payload, path)
        if pair_name in results:
            raise DataError(f"Duplicate AlignmentPair {pair_name!r} in {directory}.")
        results[pair_name] = payload
    return results


def expand_dataset_ids(dataset_ids: List[str]) -> List[str]:
    """
    Replace each dataset identifier by the result directories it writes.

    Parameters
    ----------
    dataset_ids : List[str]
        Dataset identifiers as conditions list them, e.g. ``["D3", "D4"]``.

    Returns
    -------
    List[str]
        Result directory names in order, e.g. ``["D3", "D4_schema",
        "D4_instance"]``.
    """
    expanded: List[str] = []
    for dataset_id in dataset_ids:
        expanded.extend(_SUB_DATASETS.get(dataset_id, (dataset_id,)))
    return expanded


def _f1_by_unit(
    condition_id: str,
    dataset_ids: List[str],
    results_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Map each result directory to the F1 of every alignment pair it holds.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier.
    dataset_ids : List[str]
        Datasets whose results contribute F1 values; ``"D4"`` is expanded.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping from result directory to ``{pair_name: f1}``; directories
        without results are omitted.

    Raises
    ------
    DataError
        If an F1 metric is missing, a result uses an older evaluation
        protocol, or a ``pair_name`` recurs across datasets.
    """
    units: Dict[str, Dict[str, float]] = {}
    seen: set = set()
    for dataset_id in expand_dataset_ids(dataset_ids):
        results = load_results_for_condition(condition_id, dataset_id, results_dir)
        if results is None:
            continue
        if seen & set(results):
            raise DataError(
                f"Duplicate AlignmentPair {sorted(seen & set(results))} for condition "
                f"{condition_id!r} across datasets."
            )
        seen |= set(results)
        units[dataset_id] = _unit_scores(results)
    return units


def _unit_scores(results: Dict[str, dict]) -> Dict[str, float]:
    """Return ``{pair_name: f1}`` for one result directory, checking the protocol."""
    scores: Dict[str, float] = {}
    for pair_name in sorted(results):
        _require_current_protocol(results[pair_name], pair_name)
        scores[pair_name] = _extract_f1(results[pair_name], pair_name)
    return scores


def collect_f1_scores(
    condition_id: str,
    dataset_ids: List[str],
    results_dir: str = _DEFAULT_RESULTS_DIR,
) -> List[float]:
    """
    Collect one F1 value per alignment pair across datasets.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    dataset_ids : List[str]
        Datasets whose alignment pairs contribute F1 values; ``"D4"`` expands
        to ``D4_schema`` and ``D4_instance``.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    List[float]
        F1 values ordered deterministically by dataset then ``pair_name``.

    Raises
    ------
    DataError
        If an F1 metric is missing, a result uses an older evaluation
        protocol, or a ``pair_name`` is duplicated.
    """
    units = _f1_by_unit(condition_id, dataset_ids, results_dir)
    return [f1 for scores in units.values() for f1 in scores.values()]


def _matched_pair_names(
    f1_a: Dict[str, float],
    f1_b: Dict[str, float],
) -> List[str]:
    """
    Return the shared alignment-pair names, requiring identical name sets.

    Parameters
    ----------
    f1_a, f1_b : Dict[str, float]
        F1 mappings keyed by ``pair_name`` for the two conditions.

    Returns
    -------
    List[str]
        The common ``pair_name`` keys sorted deterministically.

    Raises
    ------
    DataError
        If the two conditions do not cover an identical set of pair names.
    """
    names_a, names_b = set(f1_a), set(f1_b)
    if names_a != names_b:
        raise DataError(
            "AlignmentPair names differ between conditions; "
            f"only in A: {sorted(names_a - names_b)}, "
            f"only in B: {sorted(names_b - names_a)}."
        )
    return sorted(names_a)


def _paired_units(
    units_a: Dict[str, Dict[str, float]],
    units_b: Dict[str, Dict[str, float]],
) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Pair the two conditions' F1 values within every shared dataset column.

    Parameters
    ----------
    units_a, units_b : Dict[str, Dict[str, float]]
        Per-directory ``{pair_name: f1}`` mappings of the two conditions.

    Returns
    -------
    Dict[str, Tuple[List[float], List[float]]]
        Mapping from dataset column to aligned F1 lists of A and B.

    Raises
    ------
    DataError
        If the conditions cover different columns or different pair names.
    """
    if set(units_a) != set(units_b):
        raise DataError(
            "Dataset columns differ between conditions; "
            f"only in A: {sorted(set(units_a) - set(units_b))}, "
            f"only in B: {sorted(set(units_b) - set(units_a))}."
        )
    paired: Dict[str, Tuple[List[float], List[float]]] = {}
    for unit, scores_a in units_a.items():
        names = _matched_pair_names(scores_a, units_b[unit])
        paired[unit] = ([scores_a[n] for n in names], [units_b[unit][n] for n in names])
    return paired


def _reliability_message(caught: List[warnings.WarningMessage]) -> Optional[str]:
    """
    Join the warnings that question a result's reliability into one message.

    Only categories SciPy uses to flag a degenerate test — ``RuntimeWarning``
    for numerical failures and ``UserWarning`` for zeros, ties, or an
    approximation fallback — describe the reliability of the comparison.  Any
    other category is re-issued so that intercepting the test does not hide it.

    Parameters
    ----------
    caught : List[warnings.WarningMessage]
        Warnings recorded while the test ran, in the order they were raised.

    Returns
    -------
    Optional[str]
        Reliability warning messages joined by ``"; "``, or ``None`` when none
        were raised.
    """
    messages: List[str] = []
    for entry in caught:
        if issubclass(entry.category, _RELIABILITY_WARNINGS):
            messages.append(str(entry.message))
        else:
            warnings.warn_explicit(
                entry.message, entry.category, entry.filename, entry.lineno
            )
    return "; ".join(messages) or None


def _paired_differences(scores_a: List[float], scores_b: List[float]) -> np.ndarray:
    """
    Return the paired differences ``B - A`` rounded to suppress float noise.

    Rounding makes differences that are zero in exact arithmetic exactly zero,
    so SciPy drops them, and equal differences exactly equal, so they tie.

    Parameters
    ----------
    scores_a, scores_b : List[float]
        Paired observations in matching order.

    Returns
    -------
    np.ndarray
        Rounded differences.
    """
    differences = np.asarray(scores_b, dtype=float) - np.asarray(scores_a, dtype=float)
    return np.round(differences, _DIFFERENCE_DECIMALS)


def _paired_wilcoxon(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, float, Optional[str]]:
    """
    Run the two-sided paired Wilcoxon signed-rank test on aligned scores.

    Zero differences are dropped (``zero_method="wilcox"``).  The p-value is
    computed by sign-flip permutation, which is exact whenever ``2**n`` falls
    within the resample budget — always the case for dataset-level units — and
    handles tied differences correctly, unlike SciPy's exact table.  A sample
    whose differences are all zero carries no evidence of a difference and
    returns ``p = 1`` with a warning rather than failing.  Warnings raised by
    SciPy are captured and returned rather than suppressed; capture is local to
    this call and leaves the global warning configuration untouched.

    Parameters
    ----------
    scores_a, scores_b : List[float]
        Paired observations in matching order.

    Returns
    -------
    Tuple[float, float, Optional[str]]
        Test statistic, p-value, and the captured warning messages joined into
        one string, or ``None`` when SciPy raised no warning.
    """
    differences = _paired_differences(scores_a, scores_b)
    if not np.any(differences):
        return 0.0, 1.0, "All paired differences are zero; no test was run."
    method = PermutationMethod(rng=np.random.default_rng(_PERMUTATION_SEED))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wilcoxon(differences, alternative="two-sided", method=method)
    return float(result.statistic), float(result.pvalue), _reliability_message(caught)


def _rank_biserial_effect_size(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Compute the matched-pairs rank-biserial correlation of ``B`` over ``A``.

    ``r = (R+ - R-) / (R+ + R-)``, where ``R+`` and ``R-`` are the sums of the
    ranks of ``|B - A|`` over the positive and negative differences.  Zero
    differences are dropped, as in the test itself, and tied magnitudes take
    their average rank.  The sign matches ``delta_f1``: ``+1`` when B wins every
    non-tied unit, ``-1`` when A does, ``0`` when the rank sums balance or every
    difference is zero.

    Parameters
    ----------
    scores_a, scores_b : List[float]
        Paired observations in matching order.

    Returns
    -------
    float
        Effect size in ``[-1, 1]``.
    """
    differences = _paired_differences(scores_a, scores_b)
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    r_plus = float(ranks[nonzero > 0].sum())
    r_minus = float(ranks[nonzero < 0].sum())
    return (r_plus - r_minus) / (r_plus + r_minus)


def minimum_attainable_p(n_nonzero: int) -> float:
    """
    Return the smallest two-sided p-value an exact signed-rank test can reach.

    Parameters
    ----------
    n_nonzero : int
        Number of units with a non-zero paired difference.

    Returns
    -------
    float
        ``min(1, 2 / 2**n_nonzero)``, attained when every difference has the
        same sign.
    """
    return min(1.0, 2.0 / 2.0**n_nonzero)


def _pair_outcomes(paired: Dict[str, Tuple[List[float], List[float]]]) -> Dict[str, int]:
    """
    Count alignment pairs on which B beats, ties, or loses to A.

    Parameters
    ----------
    paired : Dict[str, Tuple[List[float], List[float]]]
        Aligned per-column F1 lists of A and B.

    Returns
    -------
    Dict[str, int]
        ``pair_wins``, ``pair_ties``, and ``pair_losses`` from B's side.
    """
    differences = np.concatenate(
        [_paired_differences(a, b) for a, b in paired.values()]
    )
    return {
        "pair_wins": int(np.sum(differences > 0)),
        "pair_ties": int(np.sum(differences == 0)),
        "pair_losses": int(np.sum(differences < 0)),
    }


def _unit_means(
    paired: Dict[str, Tuple[List[float], List[float]]],
) -> Tuple[List[float], List[float]]:
    """Return each column's mean F1 for A and for B, in column order."""
    means_a = [float(np.mean(a)) for a, _ in paired.values()]
    means_b = [float(np.mean(b)) for _, b in paired.values()]
    return means_a, means_b


def _test_fields(means_a: List[float], means_b: List[float]) -> dict:
    """
    Run the signed-rank test on unit means, or mark it as not testable.

    Parameters
    ----------
    means_a, means_b : List[float]
        Per-unit mean F1 of A and B.

    Returns
    -------
    dict
        ``testable``, ``statistic``, ``p_value``, ``wilcoxon_warning``, and
        ``n_nonzero``; the first three are ``False``/``None`` below
        :data:`_MIN_PAIRED_OBSERVATIONS` units.
    """
    n_nonzero = int(np.count_nonzero(_paired_differences(means_a, means_b)))
    fields = {"testable": False, "statistic": None, "p_value": None,
              "wilcoxon_warning": None, "n_nonzero": n_nonzero}
    if len(means_a) < _MIN_PAIRED_OBSERVATIONS:
        fields["wilcoxon_warning"] = (
            f"Not tested: {len(means_a)} independent units, "
            f"minimum is {_MIN_PAIRED_OBSERVATIONS}."
        )
        return fields
    statistic, p_value, message = _paired_wilcoxon(means_a, means_b)
    fields.update(testable=True, statistic=statistic, p_value=p_value,
                  wilcoxon_warning=message)
    return fields


def wilcoxon_comparison(
    condition_a_id: str,
    condition_b_id: str,
    dataset_ids: List[str],
    results_dir: str = _DEFAULT_RESULTS_DIR,
) -> dict:
    """
    Compare two conditions with a paired Wilcoxon signed-rank test over datasets.

    The test unit is the dataset column (see the module notes): pairs are
    matched by ``AlignmentPair`` name within each column, averaged into one
    value per column, and the column means are tested.  A comparison with
    fewer than five columns is reported but not tested.

    Parameters
    ----------
    condition_a_id : str
        Baseline condition identifier.
    condition_b_id : str
        Comparison condition identifier.
    dataset_ids : List[str]
        Datasets whose alignment pairs are compared; ``"D4"`` expands.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    dict
        ``condition_a``, ``condition_b``, ``units`` (column names), ``n_units``,
        ``n_pairs``, ``n_nonzero``, ``testable``, ``statistic`` and ``p_value``
        (``None`` when not testable), ``significant``, ``min_attainable_p``,
        ``mean_f1_a`` and ``mean_f1_b`` (means of the column means),
        ``delta_f1`` (``mean_f1_b - mean_f1_a``), signed ``effect_size_r``,
        ``pair_wins``/``pair_ties``/``pair_losses``, and ``wilcoxon_warning``.

    Raises
    ------
    DataError
        If pair names or columns differ between the conditions, a result is
        malformed, a result uses an older evaluation protocol, or no result
        exists for either condition.
    """
    paired = _paired_units(
        _f1_by_unit(condition_a_id, dataset_ids, results_dir),
        _f1_by_unit(condition_b_id, dataset_ids, results_dir),
    )
    if not paired:
        raise DataError(
            f"No results for {condition_a_id} or {condition_b_id} on {dataset_ids}."
        )
    means_a, means_b = _unit_means(paired)
    fields = _test_fields(means_a, means_b)
    return {
        "condition_a": condition_a_id,
        "condition_b": condition_b_id,
        "units": list(paired),
        "n_units": len(paired),
        "n_pairs": sum(len(a) for a, _ in paired.values()),
        **fields,
        "significant": bool(fields["testable"] and fields["p_value"] < _DEFAULT_ALPHA),
        "min_attainable_p": minimum_attainable_p(fields["n_nonzero"]),
        "mean_f1_a": float(np.mean(means_a)),
        "mean_f1_b": float(np.mean(means_b)),
        "delta_f1": float(np.mean(means_b) - np.mean(means_a)),
        "effect_size_r": _rank_biserial_effect_size(means_a, means_b),
        **_pair_outcomes(paired),
    }


def _group_comparisons(group: str) -> Tuple[_Comparison, ...]:
    """Return the canonical comparisons for an ablation group."""
    try:
        return _GROUP_COMPARISONS[group]
    except KeyError:
        raise ValueError(
            f"Unknown ablation group {group!r}; "
            f"expected one of {sorted(_GROUP_COMPARISONS)}."
        ) from None


def _apply_correction(result: dict, alpha_corrected: float) -> dict:
    """
    Add the group's Bonferroni verdict and power flag to one comparison.

    Parameters
    ----------
    result : dict
        A :func:`wilcoxon_comparison` result.
    alpha_corrected : float
        Bonferroni-corrected significance level of the group.

    Returns
    -------
    dict
        ``result`` extended with ``alpha_corrected``, ``corrected_significant``,
        and ``underpowered`` (the attainable p floor exceeds the corrected
        alpha, so no outcome could be significant).
    """
    tested = result["testable"]
    result["alpha_corrected"] = alpha_corrected
    result["corrected_significant"] = bool(tested and result["p_value"] < alpha_corrected)
    result["underpowered"] = bool(result["min_attainable_p"] >= alpha_corrected)
    return result


def run_group_comparisons(
    group: str,
    results_dir: str = _DEFAULT_RESULTS_DIR,
    alpha: float = _DEFAULT_ALPHA,
) -> List[dict]:
    """
    Run every canonical comparison for an ablation group with Bonferroni correction.

    Bonferroni correction is applied only within the requested group: the
    corrected threshold is ``alpha`` divided by the number of comparisons in
    that group, untestable ones included, and never shared across groups.

    Parameters
    ----------
    group : str
        Ablation group identifier, one of ``"A"`` to ``"E"``.
    results_dir : str
        Root directory of previously generated result files.
    alpha : float
        Family-wise significance level to correct within the group.

    Returns
    -------
    List[dict]
        One :func:`wilcoxon_comparison` result per comparison, each extended
        by :func:`_apply_correction`.

    Raises
    ------
    ValueError
        If ``group`` is not a recognised ablation group.
    """
    comparisons = _group_comparisons(group)
    alpha_corrected = alpha / len(comparisons)
    return [
        _apply_correction(
            wilcoxon_comparison(
                comparison.condition_a,
                comparison.condition_b,
                list(comparison.datasets),
                results_dir,
            ),
            alpha_corrected,
        )
        for comparison in comparisons
    ]


def _sanitise_warning(message: object) -> str:
    """
    Collapse a warning into text that occupies exactly one Markdown table cell.

    Shared with :mod:`kgsemembed.evaluation.aggregator` so both statistical
    outputs escape warnings identically; only the placeholder for an absent
    warning differs between them.

    Parameters
    ----------
    message : object
        Warning text captured from SciPy, or any object carrying it.

    Returns
    -------
    str
        The message on a single line, with cell-splitting pipes escaped.
    """
    return " ".join(str(message).split()).replace("|", "\\|")


def _warning_cell(result: dict) -> str:
    """Render a captured Wilcoxon warning as a single Markdown table cell."""
    message = result.get("wilcoxon_warning")
    if not message:
        return _NO_WARNING_CELL
    return _sanitise_warning(message)


def _p_value_cell(result: dict) -> str:
    """Render a p-value, or ``n/a`` for a comparison that was not tested."""
    p_value = result.get("p_value")
    return _NOT_TESTED_CELL if p_value is None else f"{p_value:.4f}"


def pair_outcome_cell(result: dict) -> str:
    """Render pair-level wins, ties, and losses of B as ``W/T/L``."""
    return f"{result['pair_wins']}/{result['pair_ties']}/{result['pair_losses']}"


def _format_row(result: dict) -> str:
    """Render a single comparison result as a Markdown table row."""
    cells = (
        result["condition_a"],
        result["condition_b"],
        str(result["n_units"]),
        pair_outcome_cell(result),
        _p_value_cell(result),
        f"{result['min_attainable_p']:.4f}",
        str(result["corrected_significant"]),
        f"{result['delta_f1']:.4f}",
        f"{result['effect_size_r']:.4f}",
        _warning_cell(result),
    )
    return "| " + " | ".join(cells) + " |"


def export_stats_table(
    comparison_results: List[dict],
    output_path: str,
) -> None:
    """
    Write comparison results to a Markdown table.

    Parameters
    ----------
    comparison_results : List[dict]
        Results produced by :func:`run_group_comparisons`.  ``n`` is the number
        of independent dataset units; ``pairs (W/T/L)`` counts alignment pairs
        B wins, ties, and loses; ``min p`` is the smallest p-value the test
        could reach.  A comparison that was not tested shows ``n/a`` as its
        p-value and its reason in the warning column, which reads ``"-"`` when
        SciPy was silent.
    output_path : str
        Destination Markdown file; parent directories are created as needed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_TABLE_HEADER, _TABLE_SEPARATOR]
    lines.extend(_format_row(result) for result in comparison_results)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
