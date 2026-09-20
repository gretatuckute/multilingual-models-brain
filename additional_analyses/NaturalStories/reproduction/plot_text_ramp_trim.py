#!/usr/bin/env python3
"""Plot a completed ramp-trimmed Natural Stories text-model layer curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--color", default="#009E73")
    parser.add_argument("--output-stem", default=None)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "results" / "layer_means.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path).sort_values("layer").reset_index(drop=True)

    required = {"layer", "mean", "sem", "count"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing layer-summary columns: {sorted(missing)}")
    if int(manifest.get("test_trim_start_seconds", -1)) != 100:
        raise ValueError("This plot requires a 100-second ramp-trimmed run")
    if not manifest.get("correlation_scored_after_test_trim_only", False):
        raise ValueError("Manifest does not confirm scoring-only ramp trimming")
    if summary["count"].nunique() != 1 or int(summary["count"].iloc[0]) != 9:
        raise ValueError("Expected nine held-out-story scores at every layer")
    if not np.isfinite(summary[["mean", "sem"]].to_numpy()).all():
        raise ValueError("Layer summary contains non-finite values")

    peak_index = int(summary["mean"].idxmax())
    peak_layer = int(summary.loc[peak_index, "layer"])
    peak_mean = float(summary.loc[peak_index, "mean"])

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    layer = summary["layer"].to_numpy(dtype=int)
    mean = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].to_numpy(dtype=float)
    ax.fill_between(
        layer,
        mean - sem,
        mean + sem,
        color=args.color,
        alpha=0.17,
        linewidth=0,
        label="Across-story SEM",
    )
    ax.plot(layer, mean, color=args.color, linewidth=2.4)
    ax.scatter(
        [peak_layer],
        [peak_mean],
        s=48,
        color=args.color,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=f"Peak: layer {peak_layer}, r = {peak_mean:.3f}",
    )
    representation_label = args.model_label.partition(" (")[0]
    ax.set_xlabel(f"{representation_label} representation level")
    ax.set_ylabel("Mean held-out-story Pearson r")
    fig.suptitle(
        f"Natural Stories encoding — {args.model_label}",
        x=0.125,
        y=0.98,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.125,
        0.925,
        "First 100 seconds excluded from held-out scoring",
        ha="left",
        va="top",
        fontsize=11.5,
        color="#555555",
    )
    ax.set_xlim(int(layer.min()), int(layer.max()))
    ax.set_xticks(np.arange(0, int(layer.max()) + 1, 8))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.9, alpha=0.75)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2)
    fig.subplots_adjust(bottom=0.25, top=0.84)

    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or f"{manifest['model_key']}_ramptrim100s_layer_curve"
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(output_dir / f"{stem}.png")
    print(output_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
