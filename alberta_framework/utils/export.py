"""Export utilities for experiment results.

Provides functions for exporting results to CSV, JSON, LaTeX tables,
and markdown, suitable for academic publications.
"""

import csv
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, SupportsFloat

import numpy as np
from numpy.typing import NDArray

from alberta_framework._seed_validation import require_unique_jax_seeds

if TYPE_CHECKING:
    import pandas as pd

    from alberta_framework.utils.experiments import AggregatedResults, MetricSummary
    from alberta_framework.utils.statistics import SignificanceResult


def _exported_number(value: SupportsFloat) -> str:
    """Return the shortest text that round-trips one finite binary64 value."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"refusing to export non-finite measurement: {number!r}")
    return repr(number)


def _require_finite_array(values: NDArray[np.float64], *, name: str) -> None:
    """Reject an array that cannot serve as finite numeric export data."""
    try:
        all_finite = bool(np.all(np.isfinite(values)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric measurements") from exc
    if not all_finite:
        raise ValueError(f"{name} contains a non-finite measurement")


def _preflight_export_results(
    results: dict[str, "AggregatedResults"],
    *,
    metric: str | None = None,
) -> None:
    """Validate the complete aggregate before any export touches the filesystem."""
    if not results:
        raise ValueError("export results must be non-empty")

    for name, aggregate in results.items():
        seeds = require_unique_jax_seeds(
            aggregate.seeds,
            name=f"AggregatedResults {name!r} seeds",
        )
        seed_count = len(seeds)

        if not aggregate.metric_arrays:
            raise ValueError(f"AggregatedResults {name!r} metric_arrays must be non-empty")
        if not aggregate.summary:
            raise ValueError(f"AggregatedResults {name!r} summary must be non-empty")
        if metric is not None and (
            metric not in aggregate.metric_arrays or metric not in aggregate.summary
        ):
            raise ValueError(
                f"AggregatedResults {name!r} must contain requested metric {metric!r} "
                "in both metric_arrays and summary"
            )
        if set(aggregate.metric_arrays) != set(aggregate.summary):
            raise ValueError(
                f"AggregatedResults {name!r} metric_arrays and summary must contain "
                "the same metric names"
            )
        for metric_name, values in aggregate.metric_arrays.items():
            qualified_name = f"AggregatedResults {name!r} metric {metric_name!r}"
            if values.ndim != 2:
                raise ValueError(
                    f"{qualified_name} must be a two-dimensional seed-by-step array"
                )
            if values.shape[0] != seed_count:
                raise ValueError(
                    f"AggregatedResults {name!r} seed count ({seed_count}) does not match "
                    f"metric rows ({values.shape[0]}) for {metric_name!r}"
                )
            if values.shape[1] == 0:
                raise ValueError(f"{qualified_name} must contain at least one metric step")
            _require_finite_array(values, name=qualified_name)

        for metric_name, summary in aggregate.summary.items():
            qualified_name = f"AggregatedResults {name!r} summary {metric_name!r}"
            if summary.values.ndim != 1:
                raise ValueError(f"{qualified_name} values must be one-dimensional")
            if type(summary.n_seeds) is not int:
                raise ValueError(f"{qualified_name} n_seeds must be a built-in integer")
            value_count = summary.values.shape[0]
            if summary.n_seeds != seed_count or value_count != seed_count:
                raise ValueError(
                    f"{qualified_name} seed counts disagree: seeds={seed_count}, "
                    f"n_seeds={summary.n_seeds}, values={value_count}"
                )
            for statistic in (summary.mean, summary.std, summary.min, summary.max):
                _exported_number(statistic)
            _require_finite_array(summary.values, name=f"{qualified_name} values")


def _write_text_atomically(filepath: Path, payload: str) -> None:
    """Replace ``filepath`` only after complete text is staged beside it."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filepath.name}.",
        suffix=".tmp",
        dir=filepath.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    open_descriptor: int | None = descriptor
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        open_descriptor = None
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, filepath)
    finally:
        if open_descriptor is not None:
            os.close(open_descriptor)
        temporary.unlink(missing_ok=True)


def export_to_csv(
    results: dict[str, "AggregatedResults"],
    filepath: str | Path,
    metric: str = "squared_error",
    include_timeseries: bool = False,
) -> None:
    """Export results to CSV file.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        filepath: Path to output CSV file
        metric: Metric to export
        include_timeseries: Whether to include full timeseries (large!)
    """
    filepath = Path(filepath)

    if include_timeseries:
        _export_timeseries_csv(results, filepath, metric)
    else:
        _export_summary_csv(results, filepath, metric)


def _export_summary_csv(
    results: dict[str, "AggregatedResults"],
    filepath: Path,
    metric: str,
) -> None:
    """Export summary statistics to CSV."""
    _preflight_export_results(results, metric=metric)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["config", "mean", "std", "min", "max", "n_seeds"])

    for name, agg in results.items():
        summary = agg.summary[metric]
        mean = _exported_number(summary.mean)
        std = _exported_number(summary.std)
        minimum = _exported_number(summary.min)
        maximum = _exported_number(summary.max)
        writer.writerow(
            [
                name,
                mean,
                std,
                minimum,
                maximum,
                summary.n_seeds,
            ]
        )

    _write_text_atomically(filepath, output.getvalue())


def _export_timeseries_csv(
    results: dict[str, "AggregatedResults"],
    filepath: Path,
    metric: str,
) -> None:
    """Export full timeseries to CSV."""
    _preflight_export_results(results, metric=metric)

    output = io.StringIO(newline="")
    writer = csv.writer(output)

    max_steps = max(agg.metric_arrays[metric].shape[1] for agg in results.values())

    headers = ["step"]
    for name, agg in results.items():
        for seed in agg.seeds:
            headers.append(f"{name}_seed{seed}")
    writer.writerow(headers)

    for step in range(max_steps):
        row: list[str | int] = [step]
        for agg in results.values():
            arr = agg.metric_arrays[metric]
            n_seeds = arr.shape[0]
            n_steps = arr.shape[1]
            for seed_idx in range(n_seeds):
                if step < n_steps:
                    row.append(_exported_number(arr[seed_idx, step]))
                else:
                    row.append("")
        writer.writerow(row)

    _write_text_atomically(filepath, output.getvalue())


def export_to_json(
    results: dict[str, "AggregatedResults"],
    filepath: str | Path,
    include_timeseries: bool = False,
) -> None:
    """Export results to JSON file.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        filepath: Path to output JSON file
        include_timeseries: Whether to include full timeseries (large!)
    """
    _preflight_export_results(results)
    filepath = Path(filepath)

    from typing import Any

    data: dict[str, Any] = {}
    for name, agg in results.items():
        summary_data: dict[str, dict[str, Any]] = {}
        for metric_name, summary in agg.summary.items():
            summary_data[metric_name] = {
                "mean": summary.mean,
                "std": summary.std,
                "min": summary.min,
                "max": summary.max,
                "n_seeds": summary.n_seeds,
                "values": summary.values.tolist(),
            }

        config_data: dict[str, Any] = {
            "seeds": agg.seeds,
            "summary": summary_data,
        }

        if include_timeseries:
            config_data["timeseries"] = {
                metric: arr.tolist() for metric, arr in agg.metric_arrays.items()
            }

        data[name] = config_data

    payload = json.dumps(data, indent=2, allow_nan=False)
    _write_text_atomically(filepath, payload)


def _finite_table_summaries(
    results: dict[str, "AggregatedResults"],
    metric: str,
) -> dict[str, "MetricSummary"]:
    """Return display summaries only after validating their rendered statistics."""
    if not results:
        raise ValueError("table results must be non-empty")

    summaries: dict[str, MetricSummary] = {}
    for name, aggregate in results.items():
        summary = aggregate.summary[metric]
        mean = float(_exported_number(summary.mean))
        std = float(_exported_number(summary.std))
        summaries[name] = summary._replace(mean=mean, std=std)
    return summaries


def generate_latex_table(
    results: dict[str, "AggregatedResults"],
    significance_results: dict[tuple[str, str], "SignificanceResult"] | None = None,
    metric: str = "squared_error",
    caption: str = "Experimental Results",
    label: str = "tab:results",
    metric_label: str = "Error",
    lower_is_better: bool = True,
) -> str:
    """Generate a LaTeX table of results.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        significance_results: Optional pairwise significance test results
        metric: Metric to display
        caption: Table caption
        label: LaTeX label for the table
        metric_label: Human-readable name for the metric
        lower_is_better: Whether lower metric values are better

    Returns:
        LaTeX table as a string
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & " + metric_label + r" & Seeds \\")
    lines.append(r"\midrule")

    # Find best result
    summaries = _finite_table_summaries(results, metric)
    if lower_is_better:
        best_name = min(summaries.keys(), key=lambda k: summaries[k].mean)
    else:
        best_name = max(summaries.keys(), key=lambda k: summaries[k].mean)

    for name in results:
        summary = summaries[name]
        mean_str = f"{summary.mean:.4f}"
        std_str = f"{summary.std:.4f}"

        # Bold if best
        if name == best_name:
            value_str = rf"\textbf{{{mean_str}}} $\pm$ {std_str}"
        else:
            value_str = rf"{mean_str} $\pm$ {std_str}"

        # Add significance marker if provided
        if significance_results:
            sig_marker = _get_significance_marker(name, best_name, significance_results)
            value_str += sig_marker

        # Escape underscores in method name
        escaped_name = name.replace("_", r"\_")
        lines.append(rf"{escaped_name} & {value_str} & {summary.n_seeds} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if significance_results:
        lines.append(r"\vspace{0.5em}")
        lines.append(r"\footnotesize{$^*$ $p < 0.05$, $^{**}$ $p < 0.01$, $^{***}$ $p < 0.001$}")

    lines.append(r"\end{table}")

    return "\n".join(lines)


def _get_significance_marker(
    name: str,
    best_name: str,
    significance_results: dict[tuple[str, str], "SignificanceResult"],
) -> str:
    """Get significance marker for comparison with best method."""
    if name == best_name:
        return ""

    # Find comparison result
    key1 = (name, best_name)
    key2 = (best_name, name)

    if key1 in significance_results:
        result = significance_results[key1]
    elif key2 in significance_results:
        result = significance_results[key2]
    else:
        return ""

    if not result.significant:
        return ""

    p = result.p_value
    if p < 0.001:
        return r"$^{***}$"
    elif p < 0.01:
        return r"$^{**}$"
    elif p < 0.05:
        return r"$^{*}$"
    return ""


def generate_markdown_table(
    results: dict[str, "AggregatedResults"],
    significance_results: dict[tuple[str, str], "SignificanceResult"] | None = None,
    metric: str = "squared_error",
    metric_label: str = "Error",
    lower_is_better: bool = True,
) -> str:
    """Generate a markdown table of results.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        significance_results: Optional pairwise significance test results
        metric: Metric to display
        metric_label: Human-readable name for the metric
        lower_is_better: Whether lower metric values are better

    Returns:
        Markdown table as a string
    """
    lines = []
    lines.append(f"| Method | {metric_label} (mean ± std) | Seeds |")
    lines.append("|--------|-------------------------|-------|")

    # Find best result
    summaries = _finite_table_summaries(results, metric)
    if lower_is_better:
        best_name = min(summaries.keys(), key=lambda k: summaries[k].mean)
    else:
        best_name = max(summaries.keys(), key=lambda k: summaries[k].mean)

    for name in results:
        summary = summaries[name]
        mean_str = f"{summary.mean:.4f}"
        std_str = f"{summary.std:.4f}"

        # Bold if best
        if name == best_name:
            value_str = f"**{mean_str}** ± {std_str}"
        else:
            value_str = f"{mean_str} ± {std_str}"

        # Add significance marker if provided
        if significance_results:
            sig_marker = _get_md_significance_marker(name, best_name, significance_results)
            value_str += sig_marker

        lines.append(f"| {name} | {value_str} | {summary.n_seeds} |")

    if significance_results:
        lines.append("")
        lines.append("\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001")

    return "\n".join(lines)


def _get_md_significance_marker(
    name: str,
    best_name: str,
    significance_results: dict[tuple[str, str], "SignificanceResult"],
) -> str:
    """Get significance marker for markdown."""
    if name == best_name:
        return ""

    key1 = (name, best_name)
    key2 = (best_name, name)

    if key1 in significance_results:
        result = significance_results[key1]
    elif key2 in significance_results:
        result = significance_results[key2]
    else:
        return ""

    if not result.significant:
        return ""

    p = result.p_value
    if p < 0.001:
        return " ***"
    elif p < 0.01:
        return " **"
    elif p < 0.05:
        return " *"
    return ""


def generate_significance_table(
    significance_results: dict[tuple[str, str], "SignificanceResult"],
    format: str = "latex",
) -> str:
    """Generate a table of pairwise significance results.

    Args:
        significance_results: Pairwise significance test results
        format: Output format ("latex" or "markdown")

    Returns:
        Formatted table as string
    """
    if format == "latex":
        return _generate_significance_latex(significance_results)
    else:
        return _generate_significance_markdown(significance_results)


def _generate_significance_latex(
    significance_results: dict[tuple[str, str], "SignificanceResult"],
) -> str:
    """Generate LaTeX significance table."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pairwise Significance Tests}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method A & Method B & Statistic & p-value & Effect Size & Sig. \\")
    lines.append(r"\midrule")

    for (name_a, name_b), result in significance_results.items():
        sig_str = "Yes" if result.significant else "No"
        escaped_a = name_a.replace("_", r"\_")
        escaped_b = name_b.replace("_", r"\_")
        lines.append(
            rf"{escaped_a} & {escaped_b} & {result.statistic:.3f} & "
            rf"{result.p_value:.4f} & {result.effect_size:.3f} & {sig_str} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _generate_significance_markdown(
    significance_results: dict[tuple[str, str], "SignificanceResult"],
) -> str:
    """Generate markdown significance table."""
    lines = []
    lines.append("| Method A | Method B | Statistic | p-value | Effect Size | Sig. |")
    lines.append("|----------|----------|-----------|---------|-------------|------|")

    for (name_a, name_b), result in significance_results.items():
        sig_str = "Yes" if result.significant else "No"
        lines.append(
            f"| {name_a} | {name_b} | {result.statistic:.3f} | "
            f"{result.p_value:.4f} | {result.effect_size:.3f} | {sig_str} |"
        )

    return "\n".join(lines)


def save_experiment_report(
    results: dict[str, "AggregatedResults"],
    output_dir: str | Path,
    experiment_name: str,
    significance_results: dict[tuple[str, str], "SignificanceResult"] | None = None,
    metric: str = "squared_error",
) -> dict[str, Path]:
    """Save a complete experiment report with all artifacts.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        output_dir: Directory to save artifacts
        experiment_name: Name for the experiment (used in filenames)
        significance_results: Optional pairwise significance test results
        metric: Primary metric to report

    Returns:
        Dictionary mapping artifact type to file path
    """
    _preflight_export_results(results, metric=metric)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # Export summary CSV
    csv_path = output_dir / f"{experiment_name}_summary.csv"
    export_to_csv(results, csv_path, metric=metric)
    artifacts["summary_csv"] = csv_path

    # Export JSON
    json_path = output_dir / f"{experiment_name}_results.json"
    export_to_json(results, json_path, include_timeseries=False)
    artifacts["json"] = json_path

    # Generate LaTeX table
    latex_path = output_dir / f"{experiment_name}_table.tex"
    latex_content = generate_latex_table(
        results,
        significance_results=significance_results,
        metric=metric,
        caption=f"{experiment_name} Results",
        label=f"tab:{experiment_name}",
    )
    with open(latex_path, "w") as f:
        f.write(latex_content)
    artifacts["latex_table"] = latex_path

    # Generate markdown table
    md_path = output_dir / f"{experiment_name}_table.md"
    md_content = generate_markdown_table(
        results,
        significance_results=significance_results,
        metric=metric,
    )
    with open(md_path, "w") as f:
        f.write(md_content)
    artifacts["markdown_table"] = md_path

    # If significance results provided, save those too
    if significance_results:
        sig_latex_path = output_dir / f"{experiment_name}_significance.tex"
        sig_latex = generate_significance_table(significance_results, format="latex")
        with open(sig_latex_path, "w") as f:
            f.write(sig_latex)
        artifacts["significance_latex"] = sig_latex_path

        sig_md_path = output_dir / f"{experiment_name}_significance.md"
        sig_md = generate_significance_table(significance_results, format="markdown")
        with open(sig_md_path, "w") as f:
            f.write(sig_md)
        artifacts["significance_md"] = sig_md_path

    return artifacts


def results_to_dataframe(
    results: dict[str, "AggregatedResults"],
    metric: str = "squared_error",
) -> "pd.DataFrame":
    """Convert results to a pandas DataFrame.

    Requires pandas to be installed.

    Args:
        results: Dictionary mapping config name to AggregatedResults
        metric: Metric to include

    Returns:
        DataFrame with results
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas")

    rows = []
    for name, agg in results.items():
        summary = agg.summary[metric]
        rows.append(
            {
                "method": name,
                "mean": summary.mean,
                "std": summary.std,
                "min": summary.min,
                "max": summary.max,
                "n_seeds": summary.n_seeds,
            }
        )

    return pd.DataFrame(rows)
