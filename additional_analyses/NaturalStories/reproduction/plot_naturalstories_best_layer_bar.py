#!/usr/bin/env python3
"""Plot each trained AuriStream model's peak Natural Stories encoding score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHA_ROOT = Path("/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/AuriStream-alpha")
if str(ALPHA_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHA_ROOT))

from auristream.plotting import model_color  # noqa: E402


MODELS = [
    (f"AuriStream100M_{horizon}Pred_BigAudioDataset_500k", f"100M–{horizon}Pred", "100M", horizon)
    for horizon in (1, 10, 20, 40, 60, 80)
] + [
    ("AuriStream7BDeep_1Pred_BigAudioDataset_500k", "7B–1Pred", "7B", 1),
    ("AuriStream7BDeep_40Pred_BigAudioDataset_500k", "7B–40Pred", "7B", 40),
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
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_best_layers(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    story_rows = []
    for model, label, family, horizon in MODELS:
        layer_tag = "every3" if family == "7B" else "alllayers"
        run_dir = runs_root / f"{model}_ramptrim100s_{layer_tag}_3shift"
        means = pd.read_csv(run_dir / "results" / "layer_means.csv").sort_values("layer")
        long_path = next((run_dir / "results").glob("*ramptrim_reproduced.csv"))
        long_frame = pd.read_csv(long_path)

        best = means.loc[means["mean"].idxmax()]
        selected_layer = int(best["layer"])
        selected = long_frame.loc[long_frame["layer"] == selected_layer].copy()
        if len(selected) != 9 or selected["lang"].nunique() != 9:
            raise ValueError(
                f"Expected nine held-out stories for {model} layer {selected_layer}; "
                f"found {len(selected)} rows and {selected['lang'].nunique()} story IDs"
            )
        if not np.isfinite(selected["r"]).all():
            raise ValueError(f"Non-finite held-out score for {model} layer {selected_layer}")
        recomputed_mean = float(selected["r"].mean())
        recomputed_sem = float(selected["r"].sem())
        if not np.isclose(recomputed_mean, float(best["mean"]), atol=1e-12):
            raise ValueError(f"Layer mean mismatch for {model}")
        if not np.isclose(recomputed_sem, float(best["sem"]), atol=1e-12):
            raise ValueError(f"Layer SEM mismatch for {model}")

        maximum_layer = int(means["layer"].max())
        summary_rows.append(
            {
                "model": model,
                "label": label,
                "family": family,
                "prediction_heads": horizon,
                "selected_layer": selected_layer,
                "relative_layer": selected_layer / maximum_layer,
                "mean_r": recomputed_mean,
                "sem_r": recomputed_sem,
                "n_held_out_stories": len(selected),
            }
        )
        for row in selected.itertuples(index=False):
            story_rows.append(
                {
                    "model": model,
                    "label": label,
                    "selected_layer": selected_layer,
                    "held_out_story_id": row.lang,
                    "story": row.story,
                    "r": float(row.r),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(story_rows)


def plot(summary: pd.DataFrame, story_scores: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("mean_r").reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = [model_color(model) for model in ordered["model"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.9))
    bars = ax.bar(
        x,
        ordered["mean_r"],
        width=0.72,
        color=colors,
        alpha=0.78,
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        ordered["mean_r"],
        yerr=ordered["sem_r"],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=4,
    )

    rng = np.random.default_rng(20260830)
    for position, row in ordered.iterrows():
        values = story_scores.loc[story_scores["model"] == row["model"], "r"].to_numpy()
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=35,
            color="#222222",
            alpha=0.48,
            edgecolors="none",
            zorder=5,
        )
        label_y = float(row["mean_r"] + row["sem_r"] + 0.012)
        ax.text(
            position,
            label_y,
            f"r={row['mean_r']:.3f}\nL{int(row['selected_layer'])}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color="#333333",
        )

    all_story_values = story_scores["r"].to_numpy()
    lower = min(-0.10, float(all_story_values.min()) - 0.025)
    upper = max(0.40, float((ordered["mean_r"] + ordered["sem_r"]).max()) + 0.075)
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, ordered["label"], rotation=35, ha="right")
    ax.set_ylabel("Peak mean held-out-story Pearson r")
    ax.set_xlabel("AuriStream model")
    ax.set_title("Natural Stories: peak language-network encoding by model", pad=24)
    ax.text(
        0.5,
        1.01,
        "20.48-s maximum context; first 100 s excluded only when scoring each held-out story",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color="#555555",
    )
    ax.axhline(0.0, color="#AFAFAF", linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.7, zorder=0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "naturalstories_average_language_network_best_layer_bar"
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)

    ordered.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    story_order = {model: index for index, model in enumerate(ordered["model"])}
    detailed = story_scores.assign(
        model_order=story_scores["model"].map(story_order)
    ).sort_values(["model_order", "held_out_story_id"])
    detailed.drop(columns="model_order").to_csv(
        output_dir / f"{stem}_held_out_story_scores.csv", index=False
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    summary, story_scores = load_best_layers(args.runs_root.resolve())
    plot(summary, story_scores, args.output_dir.resolve())


if __name__ == "__main__":
    main()
