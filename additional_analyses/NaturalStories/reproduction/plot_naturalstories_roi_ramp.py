#!/usr/bin/env python3
"""Plot ramp-corrected Natural Stories encoding for individual language fROIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROIS = {
    "AntTemp": ("Anterior temporal", "#D55E00"),
    "IFG": ("IFG", "#0072B2"),
    "IFGorb": ("Orbital IFG", "#56B4E9"),
    "MFG": ("MFG", "#CC79A7"),
    "PostTemp": ("Posterior temporal", "#009E73"),
}
MODELS_100M = [
    (f"AuriStream100M_{horizon}Pred_BigAudioDataset_500k", f"{horizon}Pred")
    for horizon in (1, 10, 20, 40, 60, 80)
]
MODELS_7B = [
    ("AuriStream7BDeep_1Pred_BigAudioDataset_500k", "7B–1Pred"),
    ("AuriStream7BDeep_40Pred_BigAudioDataset_500k", "7B–40Pred"),
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=here / "runs")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.runs_root / "figures"
    return args


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_roi_curve(runs_root: Path, model: str, roi: str) -> pd.DataFrame:
    is_7b = model.startswith("AuriStream7B")
    layer_tag = "every3" if is_7b else "alllayers"
    run = runs_root / f"{model}_roi-{roi}_ramptrim100s_{layer_tag}_3shift"
    frame = pd.read_csv(run / "results" / "layer_means.csv").sort_values("layer")
    maximum = int(frame["layer"].max())
    frame["relative_layer"] = frame["layer"] / maximum
    return frame


def draw_model(ax: plt.Axes, runs_root: Path, model: str, title: str) -> None:
    for roi, (label, color) in ROIS.items():
        frame = load_roi_curve(runs_root, model, roi)
        x = frame["relative_layer"].to_numpy()
        mean = frame["mean"].to_numpy()
        sem = frame["sem"].to_numpy()
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.08, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.65, label=label)
    ax.set_title(title)
    ax.set_xlabel("Relative layer")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_100m(runs_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 8.0), sharex=True, sharey=True)
    for ax, (model, label) in zip(axes.flat, MODELS_100M):
        draw_model(ax, runs_root, model, f"100M–{label}")
    axes[0, 0].set_ylabel("Mean held-out-story Pearson r")
    axes[1, 0].set_ylabel("Mean held-out-story Pearson r")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=5, fontsize=10.5)
    fig.suptitle(
        "Natural Stories encoding by language fROI: AuriStream 100M horizons",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        "20.48-s maximum context; first 100 s excluded only when scoring each held-out story",
        ha="center",
        fontsize=10.2,
        color="#4D4D4D",
    )
    fig.subplots_adjust(bottom=0.13, top=0.88, hspace=0.28, wspace=0.12)
    save(fig, output_dir, "naturalstories_roi_ramp_corrected_100m_horizons")


def plot_7b(runs_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True, sharey=True)
    for ax, (model, label) in zip(axes, MODELS_7B):
        draw_model(ax, runs_root, model, label)
    axes[0].set_ylabel("Mean held-out-story Pearson r")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=5, fontsize=10.2)
    fig.suptitle(
        "Natural Stories encoding by language fROI: AuriStream 7B",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "20.48-s maximum context; first 100 s excluded only when scoring each held-out story",
        ha="center",
        fontsize=10.2,
        color="#4D4D4D",
    )
    fig.subplots_adjust(bottom=0.20, top=0.82, wspace=0.12)
    save(fig, output_dir, "naturalstories_roi_ramp_corrected_7b_models")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    plot_100m(args.runs_root.resolve(), args.output_dir.resolve())
    plot_7b(args.runs_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
