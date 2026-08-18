"""Statistical analysis utilities for publication-quality experiments.

Provides functions for computing confidence intervals, significance tests,
effect sizes, and multiple comparison corrections.
"""

import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from alberta_framework.core._float32_scalars import validated_float32_scalar

if TYPE_CHECKING:
    from alberta_framework.utils.experiments import AggregatedResults


_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


class StatisticalSummary(NamedTuple):
    """Summary statistics for a set of values.

    Attributes:
        mean: Arithmetic mean
        std: Standard deviation
        sem: Standard error of the mean
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        median: Median value
        iqr: Interquartile range
        n_seeds: Number of samples
    """

    mean: float
    std: float
    sem: float
    ci_lower: float
    ci_upper: float
    median: float
    iqr: float
    n_seeds: int


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """Result of a statistical significance test.

    Attributes:
        test_name: Name of the test performed
        statistic: Test statistic value
        p_value: P-value of the test
        significant: Whether the result is significant at the given alpha
        alpha: Significance level used
        effect_size: Effect size (e.g., Cohen's d)
        method_a: Name of first method
        method_b: Name of second method
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    method_a: str
    method_b: str

    def __post_init__(self) -> None:
        """Reject leftover bool/int/string identities before they become a verdict."""
        _require_exact_str("test_name", self.test_name)
        _require_exact_bool("significant", self.significant)
        _require_exact_str("method_a", self.method_a)
        _require_exact_str("method_b", self.method_b)

    def _asdict(self) -> dict[str, object]:
        return asdict(self)

    def _replace(self, **changes: Any) -> "SignificanceResult":
        return replace(self, **changes)


def _validate_confidence_level(confidence_level: object) -> None:
    if type(confidence_level) not in _ALLOWED_REAL_TYPES:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    try:
        validated = validated_float32_scalar(
            "confidence_level",
            confidence_level,
            positive=True,
            upper=1.0,
            upper_inclusive=False,
        )
    except ValueError:
        raise ValueError("confidence_level must be strictly between 0 and 1") from None
    # validated ensures (0,1); keep for mypy
    _ = validated


def _require_finite_values(values: NDArray[np.floating], *, name: str) -> None:
    """Reject NaN/inf samples so summaries cannot gold-plate a poisoned seed."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def _require_sample_vector(values: object, *, name: str) -> NDArray[np.float64]:
    """Return ``values`` as a rank-1 array: exactly one sample per seed.

    A ``(n_seeds, n_steps)`` matrix would otherwise be flattened by the
    spread estimators while ``n_seeds`` still reported the row count.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional sample vector (one value per seed), "
            f"got shape {arr.shape}; reduce per seed first or use "
            "compute_timeseries_statistics"
        )
    return arr


def _require_probability(value: object, *, name: str, strict: bool) -> float:
    if type(value) not in _ALLOWED_REAL_TYPES:
        domain = "strictly between 0 and 1" if strict else "in [0, 1]"
        raise ValueError(f"{name} must be a finite real {domain}")
    try:
        if strict:
            return validated_float32_scalar(
                name, value, positive=True, upper=1.0, upper_inclusive=False
            )
        return validated_float32_scalar(
            name, value, lower=0.0, upper=1.0, upper_inclusive=True
        )
    except ValueError:
        domain = "strictly between 0 and 1" if strict else "in [0, 1]"
        raise ValueError(f"{name} must be a finite real {domain}") from None


def _require_alpha(alpha: object) -> float:
    return _require_probability(alpha, name="alpha", strict=True)


def _require_positive_int(name: str, value: object) -> int:
    """Reject bool/float aliases that ordered comparisons treat as legal counts."""
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a positive integer")
    count = operator.index(cast(SupportsIndex, value))
    if count <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return count


def _require_p_value(value: object, *, name: str) -> float:
    return _require_probability(value, name=name, strict=False)


def compute_statistics(
    values: NDArray[np.float64] | list[float],
    confidence_level: float = 0.95,
) -> StatisticalSummary:
    """Compute comprehensive statistics for a set of values.

    Args:
        values: Array of values (e.g., final performance across seeds)
        confidence_level: Confidence level for CI (default 0.95)

    Returns:
        StatisticalSummary with all statistics

    Raises:
        ValueError: If values is empty, any sample is non-finite, or
            ``confidence_level`` is not strictly between 0 and 1.
    """
    arr = _require_sample_vector(values, name="values")
    n = len(arr)
    if n == 0:
        raise ValueError("values must be non-empty")
    _require_finite_values(arr, name="values")
    _validate_confidence_level(confidence_level)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)

    # Compute confidence interval
    try:
        from scipy import stats

        if n > 1:
            t_value = float(stats.t.ppf((1 + confidence_level) / 2, n - 1))
            margin = t_value * sem
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = ci_upper = mean
    except ImportError:
        # Fallback without scipy: normal approximation.  Only two quantiles
        # are wired in — 0.95 -> z=1.96; every other confidence level silently
        # gets the 99% quantile z=2.576.  The normal approximation also
        # understates the t-based interval width at small n.
        z_value = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin = z_value * sem
        ci_lower = mean - margin
        ci_upper = mean + margin

    return StatisticalSummary(
        mean=mean,
        std=std,
        sem=sem,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        median=median,
        iqr=iqr,
        n_seeds=n,
    )


def compute_timeseries_statistics(
    metric_array: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and confidence intervals for timeseries data.

    Args:
        metric_array: Array of shape (n_seeds, n_steps)
        confidence_level: Confidence level for CI

    Returns:
        Tuple of (mean, ci_lower, ci_upper) arrays of shape (n_steps,)

    Raises:
        ValueError: If metric_array has no seed rows, any sample is
            non-finite, or ``confidence_level`` is not strictly between 0 and 1.
    """
    n_seeds = metric_array.shape[0]
    if n_seeds == 0:
        raise ValueError("metric_array must contain at least one seed row")
    _require_finite_values(metric_array, name="metric_array")
    _validate_confidence_level(confidence_level)
    mean = np.mean(metric_array, axis=0)

    if n_seeds == 1:
        # One seed has no between-seed spread estimate; return the degenerate
        # point interval, matching compute_statistics([x]). The ddof=1 /
        # Student-t path below would produce all-NaN bounds (df=0).
        return mean, mean.copy(), mean.copy()

    std = np.std(metric_array, axis=0, ddof=1)
    sem = std / np.sqrt(n_seeds)

    try:
        from scipy import stats

        t_value = stats.t.ppf((1 + confidence_level) / 2, n_seeds - 1)
    except ImportError:
        # Same limitation as compute_statistics: 0.95 -> z=1.96, any other
        # level silently gets the 99% quantile z=2.576.
        t_value = 1.96 if confidence_level == 0.95 else 2.576

    margin = t_value * sem
    ci_lower = mean - margin
    ci_upper = mean + margin

    return mean, ci_lower, ci_upper


def cohens_d(
    values_a: NDArray[np.float64] | list[float],
    values_b: NDArray[np.float64] | list[float],
) -> float:
    """Compute Cohen's d effect size.

    Args:
        values_a: Values for first group
        values_b: Values for second group

    Returns:
        Cohen's d (positive means a > b). Unequal groups with zero pooled
        variance have a signed-infinite standardized difference.

    Raises:
        ValueError: If either group is not a finite one-dimensional sample
            vector, is empty, or the pooled degrees of freedom
            ``n_a + n_b - 2`` are not positive.
    """
    a = _require_sample_vector(values_a, name="values_a")
    b = _require_sample_vector(values_b, name="values_b")
    _require_finite_values(a, name="values_a")
    _require_finite_values(b, name="values_b")

    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"cohens_d requires non-empty groups (got n_a={n_a}, n_b={n_b})"
        )

    pooled_df = n_a + n_b - 2
    if pooled_df <= 0:
        raise ValueError(
            "cohens_d requires positive pooled degrees of freedom "
            f"(got n_a={n_a}, n_b={n_b}, pooled_df={pooled_df})"
        )

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    # Pooled standard deviation
    var_a = np.var(a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0.0

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / pooled_df)

    if pooled_std == 0:
        mean_difference = mean_a - mean_b
        if mean_difference == 0:
            return 0.0
        return float(np.copysign(np.inf, mean_difference))

    return float((mean_a - mean_b) / pooled_std)


def ttest_comparison(
    values_a: NDArray[np.float64] | list[float],
    values_b: NDArray[np.float64] | list[float],
    paired: bool = True,
    alpha: float = 0.05,
    method_a: str = "A",
    method_b: str = "B",
) -> SignificanceResult:
    """Perform t-test comparison between two methods.

    Args:
        values_a: Values for first method
        values_b: Values for second method
        paired: Whether to use paired t-test (default True for same seeds)
        alpha: Significance level
        method_a: Name of first method
        method_b: Name of second method

    Returns:
        SignificanceResult with test results

    Raises:
        ValueError: If ``alpha`` is not a finite probability strictly between
            0 and 1, scipy returns an invalid p-value, ``paired`` samples
            differ in length or hold fewer than 2 pairs or are identical, or
            an unpaired group is empty or has no positive pooled degrees of
            freedom.
    """
    alpha_value = _require_alpha(alpha)
    a = _require_sample_vector(values_a, name="values_a")
    b = _require_sample_vector(values_b, name="values_b")
    _require_finite_values(a, name="values_a")
    _require_finite_values(b, name="values_b")

    if paired:
        if len(a) != len(b):
            raise ValueError(
                f"paired t-test requires equal-length samples (got {len(a)} and {len(b)})"
            )
        if len(a) < 2:
            raise ValueError(f"paired t-test requires at least 2 pairs (got {len(a)})")
        _require_exact_str("method_a", method_a)
        _require_exact_str("method_b", method_b)
        if np.array_equal(a, b):
            raise ValueError(
                f"Paired comparison '{method_a}' vs '{method_b}' has identical "
                "samples; the paired t statistic is undefined"
            )
    elif len(a) == 0 or len(b) == 0:
        raise ValueError(
            f"independent t-test requires non-empty groups (got {len(a)} and {len(b)})"
        )
    elif len(a) + len(b) - 2 <= 0:
        raise ValueError(
            "independent t-test requires positive pooled degrees of freedom "
            f"(got {len(a)} and {len(b)})"
        )

    try:
        from scipy import stats

        if paired:
            result = stats.ttest_rel(a, b)
            test_name = "paired t-test"
        else:
            result = stats.ttest_ind(a, b)
            test_name = "independent t-test"
        # scipy returns (statistic, pvalue) tuple
        stat_val = float(result[0])
        p_val = _require_p_value(
            result[1], name=f"p_value returned by {test_name}"
        )
    except ImportError:
        raise ImportError("scipy is required for t-test. Install with: pip install scipy")

    effect = cohens_d(a, b)

    return SignificanceResult(
        test_name=test_name,
        statistic=stat_val,
        p_value=p_val,
        significant=p_val < alpha_value,
        alpha=alpha_value,
        effect_size=effect,
        method_a=method_a,
        method_b=method_b,
    )


def mann_whitney_comparison(
    values_a: NDArray[np.float64] | list[float],
    values_b: NDArray[np.float64] | list[float],
    alpha: float = 0.05,
    method_a: str = "A",
    method_b: str = "B",
) -> SignificanceResult:
    """Perform Mann-Whitney U test (non-parametric).

    Args:
        values_a: Values for first method
        values_b: Values for second method
        alpha: Significance level
        method_a: Name of first method
        method_b: Name of second method

    Returns:
        SignificanceResult with test results

    Raises:
        ValueError: If ``alpha`` is not a finite probability strictly between
            0 and 1, either sample is empty, or scipy returns an invalid
            p-value.
    """
    alpha_value = _require_alpha(alpha)
    a = _require_sample_vector(values_a, name="values_a")
    b = _require_sample_vector(values_b, name="values_b")
    _require_finite_values(a, name="values_a")
    _require_finite_values(b, name="values_b")

    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            f"independent Mann-Whitney test requires non-empty groups (got {len(a)} and {len(b)})"
        )

    if np.all(a == a[0]) and np.all(b == a[0]):
        # Every cross-group pair is a tie.  The exact permutation null puts
        # all mass at U=n_a*n_b/2, hence p=1; scipy's asymptotic tie variance
        # is zero and some supported versions return NaN instead.
        stat_val = len(a) * len(b) / 2.0
        p_val = 1.0
    else:
        try:
            from scipy import stats

            result = stats.mannwhitneyu(a, b, alternative="two-sided")
            # scipy returns (statistic, pvalue) tuple
            stat_val = float(result[0])
            p_val = _require_p_value(
                result[1], name="p_value returned by Mann-Whitney U"
            )
        except ImportError:
            raise ImportError(
                "scipy is required for Mann-Whitney test. Install with: pip install scipy"
            )

    # Compute rank-biserial correlation as effect size (Kerby 2014):
    # r = 2*U1/(n_a*n_b) - 1, where scipy's statistic is U1 (pairs favoring a).
    # Positive means a > b, matching the cohens_d sign convention used by the
    # parametric tests in this module.
    n_a, n_b = len(a), len(b)
    r = (2 * stat_val) / (n_a * n_b) - 1

    return SignificanceResult(
        test_name="Mann-Whitney U",
        statistic=stat_val,
        p_value=p_val,
        significant=p_val < alpha_value,
        alpha=alpha_value,
        effect_size=r,
        method_a=method_a,
        method_b=method_b,
    )


def wilcoxon_comparison(
    values_a: NDArray[np.float64] | list[float],
    values_b: NDArray[np.float64] | list[float],
    alpha: float = 0.05,
    method_a: str = "A",
    method_b: str = "B",
) -> SignificanceResult:
    """Perform Wilcoxon signed-rank test (paired non-parametric).

    Caveat: the reported ``effect_size`` is Cohen's d computed on the raw
    values, not a rank-based effect size.  The standard companion to this
    test is the matched-pairs rank-biserial correlation; interpret the
    parametric d alongside a rank test with care.

    Args:
        values_a: Values for first method
        values_b: Values for second method
        alpha: Significance level
        method_a: Name of first method
        method_b: Name of second method

    Returns:
        SignificanceResult with test results

    Raises:
        ValueError: If ``alpha`` is not a finite probability strictly between
            0 and 1, scipy returns an invalid p-value, or the paired samples
            differ in length, hold fewer than 2 pairs, or are identical, for
            which the Wilcoxon signed-rank statistic is undefined.
    """
    alpha_value = _require_alpha(alpha)
    a = _require_sample_vector(values_a, name="values_a")
    b = _require_sample_vector(values_b, name="values_b")
    _require_finite_values(a, name="values_a")
    _require_finite_values(b, name="values_b")

    if len(a) != len(b):
        raise ValueError(
            f"Wilcoxon signed-rank test requires equal-length samples "
            f"(got {len(a)} and {len(b)})"
        )
    _require_exact_str("method_a", method_a)
    _require_exact_str("method_b", method_b)
    if a.size > 0 and np.array_equal(a, b):
        raise ValueError(
            f"Paired comparison '{method_a}' vs '{method_b}' has identical "
            "samples; the Wilcoxon signed-rank statistic is undefined"
        )
    if len(a) < 2:
        raise ValueError(
            f"Wilcoxon signed-rank test requires at least 2 pairs (got {len(a)})"
        )

    try:
        from scipy import stats

        result = stats.wilcoxon(a, b, alternative="two-sided")
        # scipy returns (statistic, pvalue) tuple
        stat_val = float(result[0])
        p_val = _require_p_value(
            result[1], name="p_value returned by Wilcoxon signed-rank"
        )
    except ImportError:
        raise ImportError("scipy is required for Wilcoxon test. Install with: pip install scipy")

    effect = cohens_d(a, b)

    return SignificanceResult(
        test_name="Wilcoxon signed-rank",
        statistic=stat_val,
        p_value=p_val,
        significant=p_val < alpha_value,
        alpha=alpha_value,
        effect_size=effect,
        method_a=method_a,
        method_b=method_b,
    )


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], float]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise significance level

    Returns:
        Tuple of (list of significant booleans, corrected alpha)

    Raises:
        ValueError: If ``alpha`` is not strictly between 0 and 1 or any
            p-value is not finite and inside [0, 1].
    """
    alpha_value = _require_alpha(alpha)
    validated_p_values = [
        _require_p_value(p_value, name=f"p_values[{index}]")
        for index, p_value in enumerate(p_values)
    ]
    n_tests = len(validated_p_values)
    if n_tests == 0:
        return [], alpha_value
    corrected_alpha = alpha_value / n_tests
    significant = [p < corrected_alpha for p in validated_p_values]
    return significant, corrected_alpha


def holm_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply Holm-Bonferroni step-down correction.

    More powerful than Bonferroni while still controlling FWER.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise significance level

    Returns:
        List of significant booleans

    Raises:
        ValueError: If ``alpha`` is not strictly between 0 and 1 or any
            p-value is not finite and inside [0, 1].
    """
    alpha_value = _require_alpha(alpha)
    validated_p_values = [
        _require_p_value(p_value, name=f"p_values[{index}]")
        for index, p_value in enumerate(p_values)
    ]
    n_tests = len(validated_p_values)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(validated_p_values)
    sorted_p = [validated_p_values[i] for i in sorted_indices]

    # Apply Holm correction
    significant_sorted = []
    for i, p in enumerate(sorted_p):
        corrected_alpha = alpha_value / (n_tests - i)
        if p < corrected_alpha:
            significant_sorted.append(True)
        else:
            # Once we fail to reject, all subsequent are not significant
            significant_sorted.extend([False] * (n_tests - i))
            break

    # Restore original order
    significant = [False] * n_tests
    for orig_idx, sig in zip(sorted_indices, significant_sorted, strict=False):
        significant[orig_idx] = sig

    return significant


def common_final_window(step_counts: Mapping[str, int], window: int, metric: str) -> int:
    """Return the number of final steps every method averages, or fail closed.

    The documented ``min(window, n_steps)`` convention only holds when it
    yields the same window for every method. When ``window`` exceeds the
    shortest trace and the traces differ in length, a per-method minimum
    would silently compare methods over different horizons.

    Raises:
        ValueError: If ``step_counts`` is empty or the per-method
            ``min(window, n_steps)`` values disagree.
    """
    window = _require_positive_int("window", window)
    if not step_counts:
        raise ValueError("at least one method is required to derive a final window")
    final_windows = {min(window, n_steps) for n_steps in step_counts.values()}
    if len(final_windows) != 1:
        described = ", ".join(
            f"{name}: {n_steps} steps" for name, n_steps in sorted(step_counts.items())
        )
        _require_exact_str("metric", metric)
        raise ValueError(
            f"window={window} exceeds the shortest '{metric}' trace and the traces differ "
            f"in length ({described}); every method must average the same number of "
            "final steps"
        )
    return final_windows.pop()


def pairwise_comparisons(
    results: "dict[str, AggregatedResults]",  # noqa: F821
    metric: str = "squared_error",
    test: str = "ttest",
    correction: str = "bonferroni",
    alpha: float = 0.05,
    window: int = 100,
) -> dict[tuple[str, str], SignificanceResult]:
    """Perform all pairwise comparisons between methods.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        metric: Metric to compare
        test: Test to use ("ttest", "mann_whitney", or "wilcoxon")
        correction: Multiple comparison correction ("bonferroni" or "holm")
        alpha: Significance level
        window: Number of final steps to average

    Returns:
        Dictionary mapping (method_a, method_b) to SignificanceResult

    Raises:
        ValueError: If ``alpha`` is not a finite probability strictly between
            0 and 1, ``window`` is not positive, a metric has no steps, seed
            identities are duplicated, seeds do not match metric rows, seeds
            differ between methods used by a paired test, or ``window`` exceeds
            the shortest trace while trace lengths differ between methods.
            Paired rows are aligned by seed identity; Mann-Whitney samples
            remain unpaired.
    """
    from alberta_framework.utils.experiments import AggregatedResults

    alpha_value = _require_alpha(alpha)
    window = _require_positive_int("window", window)

    names = list(results.keys())
    n = len(names)

    # Extract final values for each method
    metric_arrays: dict[str, NDArray[np.float64]] = {}
    seeds_by_name: dict[str, list[int]] = {}
    for name, agg in results.items():
        if not isinstance(agg, AggregatedResults):
            raise TypeError(f"Expected AggregatedResults, got {type(agg)}")
        arr = agg.metric_arrays[metric]
        _require_exact_str("name", name)
        if len(set(agg.seeds)) != len(agg.seeds):
            raise ValueError(f"AggregatedResults '{name}' contains duplicate seeds")
        _require_exact_str("name", name)
        _require_exact_str("metric", metric)
        if len(agg.seeds) != arr.shape[0]:
            raise ValueError(
                f"AggregatedResults '{name}' seed count ({len(agg.seeds)}) does not match "
                f"metric rows ({arr.shape[0]}) for '{metric}'"
            )
        _require_exact_str("name", name)
        _require_exact_str("metric", metric)
        if arr.shape[1] == 0:
            raise ValueError(
                f"AggregatedResults '{name}' must contain at least one metric step "
                f"for '{metric}'"
            )
        metric_arrays[name] = arr
        seeds_by_name[name] = agg.seeds

    if n < 2:
        return {}

    final_window = common_final_window(
        {name: arr.shape[1] for name, arr in metric_arrays.items()}, window, metric
    )
    final_values: dict[str, NDArray[np.float64]] = {
        name: np.mean(arr[:, -final_window:], axis=1) for name, arr in metric_arrays.items()
    }

    if test not in ("ttest", "mann_whitney", "wilcoxon"):
        raise ValueError(f"Unknown test: {test}")

    # Perform all pairwise comparisons
    comparisons: dict[tuple[str, str], SignificanceResult] = {}
    p_values: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            name_a, name_b = names[i], names[j]
            values_a = final_values[name_a]
            values_b = final_values[name_b]

            if test in ("ttest", "wilcoxon"):
                seeds_a = seeds_by_name[name_a]
                seeds_b = seeds_by_name[name_b]
                if set(seeds_a) != set(seeds_b):
                    raise ValueError(
                        f"Paired comparison '{name_a}' vs '{name_b}' requires equal seed sets"
                    )
                index_b_by_seed = {seed: index for index, seed in enumerate(seeds_b)}
                values_b = values_b[[index_b_by_seed[seed] for seed in seeds_a]]

            if test == "ttest":
                result = ttest_comparison(
                    values_a,
                    values_b,
                    paired=True,
                    alpha=alpha_value,
                    method_a=name_a,
                    method_b=name_b,
                )
            elif test == "mann_whitney":
                result = mann_whitney_comparison(
                    values_a,
                    values_b,
                    alpha=alpha_value,
                    method_a=name_a,
                    method_b=name_b,
                )
            else:  # wilcoxon
                result = wilcoxon_comparison(
                    values_a,
                    values_b,
                    alpha=alpha_value,
                    method_a=name_a,
                    method_b=name_b,
                )

            comparisons[(name_a, name_b)] = result
            p_values.append(result.p_value)

    # Apply multiple comparison correction
    if correction == "bonferroni":
        significant_list, _ = bonferroni_correction(p_values, alpha_value)
    elif correction == "holm":
        significant_list = holm_correction(p_values, alpha_value)
    else:
        raise ValueError(f"Unknown correction: {correction}")

    # Update significance based on correction
    corrected_comparisons: dict[tuple[str, str], SignificanceResult] = {}
    for (key, result), sig in zip(comparisons.items(), significant_list, strict=False):
        corrected_comparisons[key] = SignificanceResult(
            test_name=f"{result.test_name} ({correction})",
            statistic=result.statistic,
            p_value=result.p_value,
            significant=sig,
            alpha=alpha_value,
            effect_size=result.effect_size,
            method_a=result.method_a,
            method_b=result.method_b,
        )

    return corrected_comparisons


def bootstrap_ci(
    values: NDArray[np.float64] | list[float],
    statistic: str = "mean",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Array of values
        statistic: Statistic to bootstrap ("mean" or "median")
        confidence_level: Confidence level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)

    Raises:
        ValueError: If ``values`` is empty or contains a non-finite sample,
            ``statistic`` is not ``"mean"`` or ``"median"``, ``confidence_level``
            is not strictly between 0 and 1, or ``n_bootstrap`` is not positive.
    """
    arr = _require_sample_vector(values, name="values")
    if len(arr) == 0:
        raise ValueError(
            "bootstrap_ci requires at least one value; got an empty array "
            "(a NaN interval would be indistinguishable from a real CI)"
        )
    _require_finite_values(arr, name="values")
    if type(statistic) is not str or statistic not in ("mean", "median"):
        raise ValueError("statistic must be either 'mean' or 'median'")
    _validate_confidence_level(confidence_level)
    n_bootstrap = _require_positive_int("n_bootstrap", n_bootstrap)
    rng = np.random.default_rng(seed)

    stat_func = np.mean if statistic == "mean" else np.median
    point_estimate = float(stat_func(arr))

    # Generate bootstrap samples
    bootstrap_stats_list: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_stats_list.append(float(stat_func(sample)))

    bootstrap_stats = np.array(bootstrap_stats_list)

    # Percentile method
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    ci_lower = float(np.percentile(bootstrap_stats, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_stats, upper_percentile))

    return point_estimate, ci_lower, ci_upper
