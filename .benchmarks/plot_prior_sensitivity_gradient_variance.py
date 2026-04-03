#!/usr/bin/env python
"""Plot saved results from prior_sensitivity_gradient_variance.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PARAMETER_NAMES = ("a0", "b0", "lambda")
ESTIMATOR_COLORS = {
    "pathwise": "#1d4ed8",
    "reinforce_raw_joint_score": "#b91c1c",
    "reinforce_centered_covariance": "#047857",
}
ESTIMATOR_STYLES = {
    "pathwise": ("o", "-"),
    "reinforce_raw_joint_score": ("s", "--"),
    "reinforce_centered_covariance": ("^", ":"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot variance and mean summaries from saved benchmark JSON."
    )
    parser.add_argument("input", type=Path, help="Saved JSON benchmark results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots into. Defaults to a sibling directory.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def records_to_arrays(payload: dict):
    configurations = payload["configurations"]
    functional_names = payload["functional_names"]
    estimator_names = payload["estimator_names"]
    parameter_names = payload["parameter_names"]
    fd_reference = payload.get("mean_w1_fd_reference", {})

    records = []
    for record in configurations:
        records.append(
            {
                "dimension": int(record["dimension"]),
                "num_samples": int(record["num_samples"]),
                "mean": np.asarray(record["mean"], dtype=float),
                "variance": np.asarray(record["variance"], dtype=float),
            }
        )
    return records, functional_names, estimator_names, parameter_names, fd_reference


def output_dir_for(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return input_path.with_suffix("")


def plot_dimension_summary(
    records: list[dict],
    estimator_names: list[str],
    parameter_names: list[str],
    fd_reference: dict,
    dimension: int,
    output_path: Path,
) -> None:
    records = sorted(
        [record for record in records if record["dimension"] == dimension],
        key=lambda record: record["num_samples"],
    )
    mean_w1_index = 0
    x_values = [record["num_samples"] for record in records]
    reference = np.asarray(fd_reference.get(str(dimension), []), dtype=float)
    fig, axes = plt.subplots(
        2,
        len(parameter_names),
        figsize=(4.8 * len(parameter_names), 6.0),
        squeeze=False,
        constrained_layout=True,
    )

    for col, parameter_name in enumerate(parameter_names):
        mean_ax = axes[0, col]
        var_ax = axes[1, col]
        for estimator_idx, estimator_name in enumerate(estimator_names):
            marker, linestyle = ESTIMATOR_STYLES.get(estimator_name, ("o", "-"))
            mean_y = [
                record["mean"][estimator_idx, mean_w1_index, col] for record in records
            ]
            var_y = [
                record["variance"][estimator_idx, mean_w1_index, col]
                for record in records
            ]
            label = estimator_name
            mean_ax.plot(
                x_values,
                mean_y,
                color=ESTIMATOR_COLORS.get(estimator_name, None),
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6.0,
                label=label,
            )
            var_ax.plot(
                x_values,
                var_y,
                color=ESTIMATOR_COLORS.get(estimator_name, None),
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6.0,
                label=label,
            )

        if reference.size:
            mean_ax.axhline(
                reference[col],
                color="black",
                linestyle="-.",
                linewidth=1.8,
                label="finite_difference_reference",
            )

        mean_ax.set_xscale("log")
        var_ax.set_xscale("log")
        var_ax.set_yscale("log")
        mean_ax.set_title(f"mean_w1 / {parameter_name}")
        mean_ax.grid(alpha=0.25)
        var_ax.grid(alpha=0.25)
        mean_ax.set_xlabel("Retained samples M")
        var_ax.set_xlabel("Retained samples M")

    axes[0, 0].set_ylabel("Mean gradient estimate")
    axes[1, 0].set_ylabel("Gradient estimator variance")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="outside right center",
        frameon=False,
    )
    fig.suptitle(
        f"Pathwise vs REINFORCE gradient benchmark (D={dimension}, mean_w1)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)
    records, _, estimator_names, parameter_names, fd_reference = records_to_arrays(payload)
    output_dir = output_dir_for(args.input, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dimension in sorted({record["dimension"] for record in records}):
        plot_dimension_summary(
            records=records,
            estimator_names=estimator_names,
            parameter_names=parameter_names,
            fd_reference=fd_reference,
            dimension=dimension,
            output_path=output_dir / f"gradient_summary_D{dimension}.png",
        )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
