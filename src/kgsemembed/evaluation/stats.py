"""Statistical significance testing for Phase 2 experiment results.

This module compares experimental conditions using the Wilcoxon signed-rank
test with Bonferroni correction applied independently within each ablation
group.  It operates exclusively on previously generated result files under the
``data/results/{condition_id}/{dataset_id}/{pair_name}_results.json`` convention
written by :mod:`kgsemembed.pipeline.run_experiment`; it never reruns
experiments, modifies results, or produces plots.

Observations are paired by ``AlignmentPair`` name so that repeated executions
produce identical paired samples regardless of filesystem traversal order.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.stats import wilcoxon

from kgsemembed.utils.errors import DataError

_RESULT_SUFFIX = "_results.json"
_MIN_PAIRED_OBSERVATIONS = 5
_DEFAULT_ALPHA = 0.05
_DEFAULT_RESULTS_DIR = "data/results/"

_TABLE_COLUMNS: Tuple[str, ...] = (
    "Condition A",
    "Condition B",
    "n",
    "p-value",
    "Corrected sig.",
    "delta-F1",
    "effect_size_r",
    "Warning",
)
_TABLE_HEADER = "| " + " | ".join(_TABLE_COLUMNS) + " |"
_TABLE_SEPARATOR = "| " + " | ".join(["---"] * len(_TABLE_COLUMNS)) + " |"
_NO_WARNING_CELL = "-"

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


def load_results_for_condition(
    condition_id: str,
    dataset_id: str,
    results_dir: str = _DEFAULT_RESULTS_DIR,
) -> Optional[dict]:
    """
    Load every alignment-pair result for a condition and dataset.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    dataset_id : str
        Dataset identifier, e.g. ``"D3"``.
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


def _f1_by_pair(
    condition_id: str,
    dataset_ids: List[str],
    results_dir: str,
) -> Dict[str, float]:
    """
    Map each alignment pair to its F1 across the requested datasets.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier.
    dataset_ids : List[str]
        Datasets whose results contribute F1 values.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    Dict[str, float]
        Mapping from ``pair_name`` to F1, ordered deterministically by dataset
        then ``pair_name``.

    Raises
    ------
    DataError
        If an F1 metric is missing or a ``pair_name`` recurs across datasets.
    """
    f1_by_pair: Dict[str, float] = {}
    for dataset_id in dataset_ids:
        results = load_results_for_condition(condition_id, dataset_id, results_dir)
        if results is None:
            continue
        for pair_name in sorted(results):
            if pair_name in f1_by_pair:
                raise DataError(
                    f"Duplicate AlignmentPair {pair_name!r} for condition "
                    f"{condition_id!r} across datasets."
                )
            f1_by_pair[pair_name] = _extract_f1(results[pair_name], pair_name)
    return f1_by_pair


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
        Datasets whose alignment pairs contribute F1 values.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    List[float]
        F1 values ordered deterministically by dataset then ``pair_name``.

    Raises
    ------
    DataError
        If an F1 metric is missing or a ``pair_name`` is duplicated.
    """
    return list(_f1_by_pair(condition_id, dataset_ids, results_dir).values())


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


def _paired_wilcoxon(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, float, Optional[str]]:
    """
    Run the two-sided paired Wilcoxon signed-rank test on aligned scores.

    Warnings raised by SciPy — zero differences, ties, or a sample too small
    for the normal approximation — mark a result as potentially unreliable, so
    they are captured and returned rather than suppressed.  Capture is local to
    this call and leaves the global warning configuration untouched.

    Parameters
    ----------
    scores_a, scores_b : List[float]
        Paired F1 observations in matching order.

    Returns
    -------
    Tuple[float, float, Optional[str]]
        Test statistic, p-value, and the captured warning messages joined into
        one string, or ``None`` when SciPy raised no warning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wilcoxon(scores_a, scores_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue), _reliability_message(caught)


def _rank_biserial_effect_size(statistic: float, n_pairs: int) -> float:
    """
    Convert a Wilcoxon statistic into a rank-biserial effect size.

    The magnitude of the rank difference is reported, not its direction: the
    two-sided test returns the smaller signed-rank sum, so the value falls in
    ``[0.5, 1]`` and rises as the conditions diverge.  ``delta_f1`` carries the
    direction of the difference.

    Parameters
    ----------
    statistic : float
        Wilcoxon's W as returned by the signed-rank test.
    n_pairs : int
        Number of paired observations the test consumed.

    Returns
    -------
    float
        ``1 - (2 * statistic) / (n_pairs * (n_pairs + 1))``.
    """
    return 1.0 - (2.0 * statistic) / (n_pairs * (n_pairs + 1))


def wilcoxon_comparison(
    condition_a_id: str,
    condition_b_id: str,
    dataset_ids: List[str],
    results_dir: str = _DEFAULT_RESULTS_DIR,
) -> dict:
    """
    Compare two conditions with a paired Wilcoxon signed-rank test.

    Observations are paired by ``AlignmentPair`` name, so the two conditions
    must cover an identical set of pair names across ``dataset_ids``.

    Parameters
    ----------
    condition_a_id : str
        Baseline condition identifier.
    condition_b_id : str
        Comparison condition identifier.
    dataset_ids : List[str]
        Datasets whose alignment pairs are compared.
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    dict
        Comparison summary with ``condition_a``, ``condition_b``, ``n_pairs``,
        ``statistic``, ``p_value``, ``significant``, ``mean_f1_a``,
        ``mean_f1_b``, ``delta_f1`` (``mean_f1_b - mean_f1_a``),
        ``effect_size_r`` holding the rank-biserial effect size derived from
        ``statistic`` and ``n_pairs``, and ``wilcoxon_warning`` holding any
        warning SciPy raised, else ``None``.

    Raises
    ------
    ValueError
        If fewer than five paired observations are available.
    DataError
        If pair names differ between the conditions or a result is malformed.
    """
    f1_a = _f1_by_pair(condition_a_id, dataset_ids, results_dir)
    f1_b = _f1_by_pair(condition_b_id, dataset_ids, results_dir)
    pair_names = _matched_pair_names(f1_a, f1_b)
    if len(pair_names) < _MIN_PAIRED_OBSERVATIONS:
        raise ValueError(
            f"Wilcoxon test requires at least {_MIN_PAIRED_OBSERVATIONS} paired "
            f"observations; found {len(pair_names)}."
        )
    scores_a = [f1_a[name] for name in pair_names]
    scores_b = [f1_b[name] for name in pair_names]
    statistic, p_value, wilcoxon_warning = _paired_wilcoxon(scores_a, scores_b)
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    return {
        "condition_a": condition_a_id,
        "condition_b": condition_b_id,
        "n_pairs": len(pair_names),
        "statistic": statistic,
        "p_value": p_value,
        "significant": bool(p_value < _DEFAULT_ALPHA),
        "mean_f1_a": mean_a,
        "mean_f1_b": mean_b,
        "delta_f1": mean_b - mean_a,
        "effect_size_r": _rank_biserial_effect_size(statistic, len(pair_names)),
        "wilcoxon_warning": wilcoxon_warning,
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


def run_group_comparisons(
    group: str,
    results_dir: str = _DEFAULT_RESULTS_DIR,
    alpha: float = _DEFAULT_ALPHA,
) -> List[dict]:
    """
    Run every canonical comparison for an ablation group with Bonferroni correction.

    Bonferroni correction is applied only within the requested group: the
    corrected threshold is ``alpha`` divided by the number of comparisons in
    that group, and never shared across groups.

    Parameters
    ----------
    group : str
        Ablation group identifier, one of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
    results_dir : str
        Root directory of previously generated result files.
    alpha : float
        Family-wise significance level to correct within the group.

    Returns
    -------
    List[dict]
        One :func:`wilcoxon_comparison` result per comparison, each extended
        with a ``corrected_significant`` boolean.

    Raises
    ------
    ValueError
        If ``group`` is not a recognised ablation group.
    """
    comparisons = _group_comparisons(group)
    alpha_corrected = alpha / len(comparisons)
    results: List[dict] = []
    for comparison in comparisons:
        result = wilcoxon_comparison(
            comparison.condition_a,
            comparison.condition_b,
            list(comparison.datasets),
            results_dir,
        )
        result["corrected_significant"] = bool(result["p_value"] < alpha_corrected)
        results.append(result)
    return results


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


def _format_row(result: dict) -> str:
    """Render a single comparison result as a Markdown table row."""
    cells = (
        result["condition_a"],
        result["condition_b"],
        str(result["n_pairs"]),
        f"{result['p_value']:.4f}",
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
        Results produced by :func:`run_group_comparisons`; each must carry a
        ``corrected_significant`` flag and an ``effect_size_r`` value, which is
        rendered in its own column.  Any ``wilcoxon_warning`` is rendered in a
        dedicated warning column, which reads ``"-"`` when SciPy was silent.
    output_path : str
        Destination Markdown file; parent directories are created as needed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_TABLE_HEADER, _TABLE_SEPARATOR]
    lines.extend(_format_row(result) for result in comparison_results)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
