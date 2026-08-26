#!/usr/bin/env python3
"""Plot Natural Stories reproductions, including a paper-style layer figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_INFO = {
    "xglm_small": {
        "display": "XGLM-small",
        "comparison_label": "XGLM-small (causal)",
        "color": "#0072B2",
    },
    "mt5_large": {
        "display": "mT5-large",
        "comparison_label": "mT5-large (bidirectional)",
        "color": "#D55E00",
    },
    "gpt2_xl": {
        "display": "GPT-2 XL",
        "comparison_label": "GPT-2 XL (causal)",
        "color": "#009E73",
    },
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=here / "runs")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xglm_small", "mt5_large"],
        choices=sorted(MODEL_INFO),
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(directory / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results" / "reference_layer_summary.csv"
    if not path.exists():
        path = run_dir / "results" / "benchmark_layer_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed run summary under {run_dir / 'results'}")
    frame = pd.read_csv(path).sort_values("layer").reset_index(drop=True)
    frame["normalized_layer"] = frame["layer"] / frame["layer"].max()
    if {"reference_std", "reference_count"}.issubset(frame.columns):
        frame["reference_sem"] = frame["reference_std"] / np.sqrt(
            frame["reference_count"]
        )
    frame["reproduced_sem"] = frame["reproduced_std"] / np.sqrt(
        frame["reproduced_count"]
    )
    return frame


def plot_model_diagnostic(model_key: str, run_dir: Path, summary: pd.DataFrame) -> None:
    info = MODEL_INFO[model_key]
    figure_dir = run_dir / "figures"
    x = summary["layer"].to_numpy()
    has_reference = "reference_mean" in summary.columns
    if not has_reference:
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        ax.fill_between(
            x,
            (summary["reproduced_mean"] - summary["reproduced_sem"]).to_numpy(),
            (summary["reproduced_mean"] + summary["reproduced_sem"]).to_numpy(),
            color=info["color"],
            alpha=0.18,
            linewidth=0,
            label="Across-story SEM",
        )
        ax.plot(
            x,
            summary["reproduced_mean"],
            color=info["color"],
            linewidth=2.6,
            marker="o",
            markersize=3.8,
            markeredgewidth=0,
            label="New benchmark",
        )
        peak_index = int(summary["reproduced_mean"].idxmax())
        peak_layer = int(summary.loc[peak_index, "layer"])
        peak_value = float(summary.loc[peak_index, "reproduced_mean"])
        ax.scatter([peak_layer], [peak_value], s=48, color=info["color"], zorder=4)
        ax.set_xlabel("Representation level")
        ax.set_ylabel("Mean held-out-story Pearson r")
        ax.set_title(f"Natural Stories {info['display']} benchmark")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=10.5,
        )
        fig.subplots_adjust(bottom=0.23)
        save_figure(fig, figure_dir, f"{model_key}_benchmark_layer_curve")
        return

    mean_error = summary["mean_absolute_difference"].to_numpy()
    std_error = summary["std_absolute_difference"].to_numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.10},
    )
    ax, ax_error = axes
    ax.fill_between(
        x,
        (summary["reference_mean"] - summary["reference_sem"]).to_numpy(),
        (summary["reference_mean"] + summary["reference_sem"]).to_numpy(),
        color="#4D4D4D",
        alpha=0.14,
        linewidth=0,
        label="Released ± SEM",
    )
    ax.plot(
        x,
        summary["reference_mean"],
        color="#4D4D4D",
        linewidth=3.0,
        label="Released result",
    )
    ax.plot(
        x,
        summary["reproduced_mean"],
        color=info["color"],
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=4.2,
        markeredgewidth=0,
        label="Reproduction",
    )
    peak_index = int(summary["reproduced_mean"].idxmax())
    peak_layer = int(summary.loc[peak_index, "layer"])
    peak_value = float(summary.loc[peak_index, "reproduced_mean"])
    ax.scatter([peak_layer], [peak_value], s=52, color=info["color"], zorder=4)
    ax.annotate(
        f"Peak: layer {peak_layer}, r = {peak_value:.3f}",
        xy=(peak_layer, peak_value),
        xytext=(peak_layer + 1.0, peak_value + 0.018),
        arrowprops={"arrowstyle": "-", "color": info["color"], "lw": 1.2},
        color=info["color"],
        fontsize=11,
    )
    ax.set_ylabel("Mean held-out-story Pearson r")
    ax.set_title(f"Natural Stories {info['display']} reproduction")
    ax.legend(frameon=False, loc="best", ncol=3, fontsize=10.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)

    floor = np.finfo(float).tiny
    ax_error.plot(
        x,
        np.maximum(mean_error, floor),
        color="#56B4E9",
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label="Layer mean",
    )
    ax_error.plot(
        x,
        np.maximum(std_error, floor),
        color="#009E73",
        linewidth=2.0,
        label="Layer SD",
    )
    ax_error.axhline(
        1e-5,
        color="#999999",
        linestyle=":",
        linewidth=1.5,
        label="Tolerance (10⁻⁵)",
    )
    ax_error.set_yscale("log")
    ax_error.set_ylabel("Absolute difference")
    ax_error.set_xlabel("Representation level")
    ax_error.set_xticks(np.arange(0, int(x.max()) + 1, 2))
    ax_error.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7, which="both")
    ax_error.legend(frameon=False, fontsize=10, ncol=3, loc="best")
    save_figure(fig, figure_dir, f"{model_key}_reproduction_layer_curve")


def plot_xglm_story_heatmaps(run_dir: Path) -> None:
    comparison_path = run_dir / "results" / "reference_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    required = {"layer", "lang", "r_reference", "r_reproduced", "absolute_difference"}
    if not required.issubset(comparison.columns):
        return
    comparison["layer"] = comparison["layer"].astype(int)
    comparison["lang"] = comparison["lang"].astype(int)
    layers = np.sort(comparison["layer"].unique())
    stories = np.sort(comparison["lang"].unique())
    reference = comparison.pivot(index="layer", columns="lang", values="r_reference").loc[
        layers, stories
    ]
    reproduced = comparison.pivot(
        index="layer", columns="lang", values="r_reproduced"
    ).loc[layers, stories]
    absolute_difference = comparison.pivot(
        index="layer", columns="lang", values="absolute_difference"
    ).loc[layers, stories]
    shared_min = float(min(reference.to_numpy().min(), reproduced.to_numpy().min()))
    shared_max = float(max(reference.to_numpy().max(), reproduced.to_numpy().max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 8.0), gridspec_kw={"wspace": 0.34})
    for ax_heat, matrix, title in zip(
        axes[:2],
        (reference, reproduced),
        ("Released result", "Reproduction"),
    ):
        image = ax_heat.imshow(
            matrix.to_numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=shared_min,
            vmax=shared_max,
        )
        ax_heat.set_title(title)
        ax_heat.set_xlabel("Held-out story")
        ax_heat.set_xticks(np.arange(len(stories)), stories)
        ax_heat.set_yticks(np.arange(0, len(layers), 2), layers[::2])
        ax_heat.spines[["top", "right"]].set_visible(True)
        colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
        colorbar.set_label("Pearson r")
    axes[0].set_ylabel("XGLM representation level")

    error_image = axes[2].imshow(
        absolute_difference.to_numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=float(absolute_difference.to_numpy().max()),
    )
    axes[2].set_title("Absolute difference")
    axes[2].set_xlabel("Held-out story")
    axes[2].set_xticks(np.arange(len(stories)), stories)
    axes[2].set_yticks(np.arange(0, len(layers), 2), layers[::2])
    axes[2].spines[["top", "right"]].set_visible(True)
    colorbar = fig.colorbar(error_image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("|reproduced − released r|")
    fig.suptitle("Natural Stories XGLM-small: all 225 held-out-story scores", y=0.995)
    save_figure(
        fig,
        run_dir / "figures",
        "xglm_small_reproduction_layer_story_heatmaps",
    )


def interpolate_curve(frame: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, frame["normalized_layer"], frame[column])


def plot_paper_style(
    runs_root: Path,
    model_keys: list[str],
    summaries: dict[str, pd.DataFrame],
) -> None:
    number_of_panels = len(model_keys) + (1 if len(model_keys) > 1 else 0)
    fig, axes = plt.subplots(
        1,
        number_of_panels,
        figsize=(4.2 * number_of_panels, 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for index, model_key in enumerate(model_keys):
        frame = summaries[model_key]
        info = MODEL_INFO[model_key]
        ax = axes[index]
        x = frame["normalized_layer"].to_numpy()
        ax.fill_between(
            x,
            (frame["reproduced_mean"] - frame["reproduced_sem"]).to_numpy(),
            (frame["reproduced_mean"] + frame["reproduced_sem"]).to_numpy(),
            color=info["color"],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            x,
            frame["reference_mean"],
            color="#4D4D4D",
            linewidth=2.8,
            label="Released",
        )
        ax.plot(
            x,
            frame["reproduced_mean"],
            color=info["color"],
            linewidth=2.2,
            linestyle="--",
            marker="o",
            markersize=3.8,
            markeredgewidth=0,
            label="Reproduced",
        )
        ax.set_title(info["display"], color=info["color"], fontweight="bold")
        ax.set_xlabel("Normalized layer position")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        if index == 0:
            ax.set_ylabel("Mean held-out-story Pearson r")

    if len(model_keys) > 1:
        grid = np.linspace(0.0, 1.0, 101)
        reference_curves = np.vstack(
            [interpolate_curve(summaries[key], "reference_mean", grid) for key in model_keys]
        )
        reproduced_curves = np.vstack(
            [interpolate_curve(summaries[key], "reproduced_mean", grid) for key in model_keys]
        )
        ax = axes[-1]
        reproduced_mean = reproduced_curves.mean(axis=0)
        reproduced_sem = reproduced_curves.std(axis=0, ddof=1) / np.sqrt(len(model_keys))
        ax.fill_between(
            grid,
            reproduced_mean - reproduced_sem,
            reproduced_mean + reproduced_sem,
            color="#6A3D9A",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(grid, reference_curves.mean(axis=0), color="#4D4D4D", linewidth=2.8)
        ax.plot(
            grid,
            reproduced_mean,
            color="#6A3D9A",
            linewidth=2.2,
            linestyle="--",
        )
        ax.set_title("Average across models", color="#6A3D9A", fontweight="bold")
        ax.set_xlabel("Normalized layer position")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
    )
    fig.suptitle("Natural Stories encoding performance across model depth", y=1.02)
    fig.subplots_adjust(wspace=0.12, bottom=0.20)
    save_figure(fig, runs_root / "figures", "paper_style_normalized_layer_reproduction")


def plot_reproduced_model_comparison(
    runs_root: Path,
    model_keys: list[str],
    summaries: dict[str, pd.DataFrame],
) -> None:
    """Compare all benchmarked models without implying a released reference."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for model_key in model_keys:
        frame = summaries[model_key]
        info = MODEL_INFO[model_key]
        x = frame["normalized_layer"].to_numpy()
        mean = frame["reproduced_mean"].to_numpy()
        sem = frame["reproduced_sem"].to_numpy()
        ax.fill_between(x, mean - sem, mean + sem, color=info["color"], alpha=0.12)
        ax.plot(
            x,
            mean,
            color=info["color"],
            linewidth=2.5,
            label=info["comparison_label"],
        )
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Mean held-out-story Pearson r")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Natural Stories encoding across model depth")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=10.5,
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.subplots_adjust(bottom=0.24)
    save_figure(fig, runs_root / "figures", "normalized_layer_model_comparison")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    runs_root = args.runs_root.resolve()
    summaries = {}
    for model_key in args.models:
        run_dir = runs_root / f"{model_key}_3shift"
        summary = load_summary(run_dir)
        summaries[model_key] = summary
        plot_model_diagnostic(model_key, run_dir, summary)
        if model_key == "xglm_small":
            plot_xglm_story_heatmaps(run_dir)
    reference_models = [
        key for key in args.models if "reference_mean" in summaries[key].columns
    ]
    if reference_models:
        plot_paper_style(runs_root, reference_models, summaries)
    if len(args.models) > 1:
        plot_reproduced_model_comparison(runs_root, list(args.models), summaries)
    print(f"Saved figures under {runs_root}")


if __name__ == "__main__":
    main()
