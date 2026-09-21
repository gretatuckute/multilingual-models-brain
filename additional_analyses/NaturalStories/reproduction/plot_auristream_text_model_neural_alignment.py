#!/usr/bin/env python3
"""Compare AuriStream-MTP with text models across two neural benchmarks.

Baseline 200 uses the canonical last-token text-model representations and
noise-ceiling-normalized language-network predictivity. Natural Stories uses
the 20.48-second text contexts matched to AuriStream's maximum input duration
and raw held-out-story Pearson correlation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auristream.plotting import (  # noqa: E402
    LAYER_UNTRAINED_7B_COLOR,
    LAYER_UNTRAINED_ERROR_ALPHA,
    LAYER_UNTRAINED_LINE_ALPHA,
    LAYER_UNTRAINED_MARKER_ALPHA,
    add_layer_grid,
    init_rcparams,
    model_color,
)


EXPECTED_UIDS = (
    "cvn7002",
    "cvn7006",
    "cvn7007",
    "cvn7009",
    "cvn7011",
    "cvn7012",
    "cvn7013",
    "cvn7016",
)
LANGUAGE_FROIS = (1, 2, 3, 4, 5)
ROI_COLUMN = "top10_tstat_langloc_SN_parc_lang_int"
RELATIVE_DEPTH_TICKS = np.linspace(0.0, 1.0, 6)
RELATIVE_DEPTH_LIMITS = (-0.015, 1.015)

AURISTREAM_LABEL = "AuriStream-MTP 7B 40Pred"
UNTRAINED_LABEL = "7B 40Pred untrained"
AURISTREAM_COLOR = model_color("AuriStream7BDeep_40Pred_BigAudioDataset_500k")

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results_figures"
    / "functional_organization_across_layers"
    / "2026-09-20"
    / "all_models_by_task"
)
DEFAULT_NATURALSTORIES_RUNS = Path(
    "/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/"
    "multilingual-models-brain-auristream/additional_analyses/"
    "NaturalStories/reproduction/runs"
)
DEFAULT_AURISTREAM_BASELINE = (
    DEFAULT_OUTPUT_DIR
    / "figure5_exploration_neural_alignment_7b40pred_baseline200_summary.csv"
)
DEFAULT_AURISTREAM_NATURAL = (
    DEFAULT_OUTPUT_DIR
    / "figure5_exploration_neural_alignment_7b40pred_naturalstories_summary.csv"
)


@dataclass(frozen=True)
class TextModelSpec:
    label: str
    baseline_directory: str
    baseline_source: str
    n_states: int
    naturalstories_run: str
    naturalstories_model_key: str
    color: str
    marker: str


# A single pink-to-purple family distinguishes text models from the established
# AuriStream teal while marker shape keeps each model identifiable in grayscale.
TEXT_MODELS = (
    TextModelSpec(
        label="GPT-2 XL",
        baseline_directory="gpt2-xl",
        baseline_source="gpt2-xl",
        n_states=49,
        naturalstories_run="gpt2_xl_context20p48s_ramptrim100s_alllayers_3shift",
        naturalstories_model_key="gpt2_xl",
        color="#D49ABB",
        marker="o",
    ),
    TextModelSpec(
        label="GPT-J 6B",
        baseline_directory="EleutherAI_gpt-j-6b",
        baseline_source="EleutherAI_gpt-j-6b",
        n_states=29,
        naturalstories_run=(
            "gpt_j_6b_context20p48s_ramptrim100s_alllayers_3shift"
        ),
        naturalstories_model_key="gpt_j_6b",
        color="#AD6FA8",
        marker="D",
    ),
    TextModelSpec(
        label="Qwen3 8B",
        baseline_directory="Qwen_Qwen3-8B",
        baseline_source="Qwen_Qwen3-8B",
        n_states=37,
        naturalstories_run=(
            "qwen3_8b_context20p48s_ramptrim100s_alllayers_3shift"
        ),
        naturalstories_model_key="qwen3_8b",
        color="#79539A",
        marker="^",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auristream-repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--naturalstories-runs",
        type=Path,
        default=DEFAULT_NATURALSTORIES_RUNS,
    )
    parser.add_argument(
        "--auristream-baseline-summary",
        type=Path,
        default=DEFAULT_AURISTREAM_BASELINE,
    )
    parser.add_argument(
        "--auristream-naturalstories-summary",
        type=Path,
        default=DEFAULT_AURISTREAM_NATURAL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stem",
        default="figure6_exploration_neural_alignment_text_models",
    )
    parser.add_argument("--ncsnr-threshold", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--reuse-text-baseline-tables",
        action="store_true",
        help="Reuse saved text-model Baseline 200 tables in the output directory.",
    )
    return parser.parse_args()


def _baseline_paths(repo: Path, spec: TextModelSpec) -> list[Path]:
    root = (
        repo
        / "results_neural"
        / "inv_neural_pred_paral"
        / spec.baseline_directory
    )
    paths = sorted(root.glob("*_last-tok_zX0_zY1_neural_meta.csv.gz"))
    expected = len(EXPECTED_UIDS) * spec.n_states
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} canonical last-token files for {spec.label}; "
            f"found {len(paths)} in {root}"
        )
    return paths


def _read_text_baseline_file(
    arguments: tuple[str, str, str, int, float],
) -> dict[str, object]:
    label, path_text, expected_source, n_states, ncsnr_threshold = arguments
    path = Path(path_text)
    source = pd.read_csv(
        path,
        usecols=[
            "uid",
            "source_model",
            "source_layer",
            "ncsnr",
            ROI_COLUMN,
            "r_cv",
            "r_cv_norm",
        ],
    )
    uid_values = source["uid"].unique()
    layer_values = source["source_layer"].unique()
    model_values = source["source_model"].unique()
    if len(uid_values) != 1 or len(layer_values) != 1 or len(model_values) != 1:
        raise RuntimeError(f"Expected one participant, layer, and model in {path}")
    if str(model_values[0]) != expected_source:
        raise RuntimeError(
            f"Expected source_model={expected_source!r} in {path}; "
            f"found {model_values[0]!r}"
        )
    uid = str(uid_values[0])
    layer = int(layer_values[0])
    if uid not in EXPECTED_UIDS or layer not in range(n_states):
        raise RuntimeError(f"Unexpected participant/layer in {path}: {uid}, {layer}")
    selected = source[
        source["ncsnr"].gt(ncsnr_threshold)
        & source[ROI_COLUMN].isin(LANGUAGE_FROIS)
    ]
    if selected.empty:
        raise RuntimeError(f"No selected language-network voxels in {path}")
    values = selected[["r_cv", "r_cv_norm"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite selected score in {path}")
    return {
        "dataset": "Baseline 200",
        "condition": label,
        "uid": uid,
        "layer": layer,
        "relative_depth": layer / (n_states - 1),
        "raw_r": float(selected["r_cv"].mean()),
        "normalized_r": float(selected["r_cv_norm"].mean()),
        "n_voxels": int(len(selected)),
        "representation": "last-tok",
        "source_path": str(path.resolve()),
    }


def load_text_baseline(
    repo: Path,
    ncsnr_threshold: float,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = []
    for spec in TEXT_MODELS:
        jobs.extend(
            (
                spec.label,
                str(path),
                spec.baseline_source,
                spec.n_states,
                ncsnr_threshold,
            )
            for path in _baseline_paths(repo, spec)
        )
    if workers <= 1:
        rows = [_read_text_baseline_file(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_read_text_baseline_file, jobs, chunksize=1))
    participants = pd.DataFrame(rows).sort_values(
        ["condition", "uid", "layer"]
    ).reset_index(drop=True)
    for spec in TEXT_MODELS:
        subset = participants[participants["condition"].eq(spec.label)]
        if subset.duplicated(["uid", "layer"]).any():
            raise RuntimeError(f"Duplicate participant/layer rows for {spec.label}")
        for uid in EXPECTED_UIDS:
            layers = np.sort(subset.loc[subset["uid"].eq(uid), "layer"].unique())
            if not np.array_equal(layers, np.arange(spec.n_states)):
                raise RuntimeError(f"Incomplete layer grid for {spec.label}, {uid}")
    summary = (
        participants.groupby(
            ["dataset", "condition", "layer", "relative_depth"],
            as_index=False,
        )
        .agg(
            mean=("normalized_r", "mean"),
            sem=("normalized_r", "sem"),
            raw_mean=("raw_r", "mean"),
            raw_sem=("raw_r", "sem"),
            count=("uid", "nunique"),
            mean_n_voxels=("n_voxels", "mean"),
        )
        .sort_values(["condition", "layer"])
        .reset_index(drop=True)
    )
    if not summary["count"].eq(len(EXPECTED_UIDS)).all():
        raise RuntimeError("Text-model Baseline 200 summary is missing participants")
    summary["metric"] = "mean participant-level r_cv_norm in language fROIs"
    return summary, participants


def load_auristream_summary(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    required = {"dataset", "condition", "layer", "relative_depth", "mean", "sem", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Missing {sorted(missing)} in {path}")
    frame = frame[frame["dataset"].eq(dataset)].copy()
    conditions = set(frame["condition"])
    expected = {AURISTREAM_LABEL, UNTRAINED_LABEL}
    if conditions != expected:
        raise RuntimeError(f"Expected {expected} in {path}; found {conditions}")
    if not np.isfinite(frame[["mean", "sem"]].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite AuriStream summary values in {path}")
    return frame


def load_text_naturalstories(runs_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    rows = []
    sources = []
    for spec in TEXT_MODELS:
        run_dir = runs_root / spec.naturalstories_run
        manifest_path = run_dir / "run_manifest.json"
        summary_path = run_dir / "results" / "layer_means.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_key") != spec.naturalstories_model_key:
            raise RuntimeError(f"Unexpected model in {manifest_path}")
        if not np.isclose(manifest.get("context_seconds", np.nan), 20.48):
            raise RuntimeError(f"Expected context_seconds=20.48 in {manifest_path}")
        if manifest.get("test_trim_start_seconds") != 100:
            raise RuntimeError(f"Expected a 100-second scoring trim: {manifest_path}")
        if not manifest.get("correlation_scored_after_test_trim_only", False):
            raise RuntimeError(f"Scoring-only trim is not confirmed: {manifest_path}")
        if manifest.get("shift") != 3:
            raise RuntimeError(f"Expected shift=3: {manifest_path}")
        frame = pd.read_csv(summary_path).sort_values("layer").reset_index(drop=True)
        if len(frame) != spec.n_states:
            raise RuntimeError(
                f"Expected {spec.n_states} layers in {summary_path}; found {len(frame)}"
            )
        if not frame["count"].eq(9).all():
            raise RuntimeError(f"Expected nine held-out stories: {summary_path}")
        frame = frame[["layer", "mean", "sem", "count"]].copy()
        frame.insert(0, "condition", spec.label)
        frame.insert(0, "dataset", "Natural Stories")
        frame["relative_depth"] = frame["layer"] / (spec.n_states - 1)
        frame["metric"] = "mean held-out-story Pearson r"
        rows.append(frame)
        sources.extend([manifest_path.resolve(), summary_path.resolve()])
    return pd.concat(rows, ignore_index=True), sources


def _curve_style(condition: str) -> dict[str, object]:
    if condition == AURISTREAM_LABEL:
        return {
            "color": AURISTREAM_COLOR,
            "marker": "s",
            "markersize": 7.2,
            "line_alpha": 0.40,
            "marker_alpha": 0.78,
            "error_alpha": 0.24,
            "zorder": 5,
        }
    if condition == UNTRAINED_LABEL:
        return {
            "color": LAYER_UNTRAINED_7B_COLOR,
            "marker": "s",
            "markersize": 7.2,
            "line_alpha": LAYER_UNTRAINED_LINE_ALPHA,
            "marker_alpha": LAYER_UNTRAINED_MARKER_ALPHA,
            "error_alpha": LAYER_UNTRAINED_ERROR_ALPHA,
            "zorder": 1,
        }
    spec = next(spec for spec in TEXT_MODELS if spec.label == condition)
    return {
        "color": spec.color,
        "marker": spec.marker,
        "markersize": 6.2,
        "line_alpha": 0.44,
        "marker_alpha": 0.78,
        "error_alpha": 0.18,
        "zorder": 3,
    }


def draw_curve(axis: plt.Axes, frame: pd.DataFrame, condition: str) -> None:
    curve = frame[frame["condition"].eq(condition)].sort_values("layer")
    if curve.empty:
        raise RuntimeError(f"No plotting rows for {condition}")
    style = _curve_style(condition)
    x = curve["relative_depth"].to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    sem = curve["sem"].to_numpy(dtype=float)
    axis.errorbar(
        x,
        y,
        yerr=sem,
        fmt="none",
        ecolor=to_rgba(style["color"], style["error_alpha"]),
        elinewidth=0.78,
        capsize=0,
        zorder=style["zorder"],
    )
    axis.plot(
        x,
        y,
        color=to_rgba(style["color"], style["line_alpha"]),
        linewidth=1.55,
        marker=style["marker"],
        markersize=style["markersize"],
        markerfacecolor=to_rgba(style["color"], style["marker_alpha"]),
        markeredgecolor=to_rgba(style["color"], style["marker_alpha"]),
        zorder=style["zorder"] + 1,
    )


def set_y_limits(axis: plt.Axes, frame: pd.DataFrame) -> None:
    low = float((frame["mean"] - frame["sem"]).min())
    high = float((frame["mean"] + frame["sem"]).max())
    span = max(0.01, high - low)
    axis.set_ylim(low - 0.06 * span, high + 0.08 * span)


def legend_handles() -> list[Line2D]:
    handles = []
    for condition in (
        AURISTREAM_LABEL,
        *(spec.label for spec in TEXT_MODELS),
        UNTRAINED_LABEL,
    ):
        style = _curve_style(condition)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=1.55,
                marker=style["marker"],
                markersize=style["markersize"],
                label=condition,
            )
        )
    return handles


def make_figure(baseline: pd.DataFrame, natural: pd.DataFrame) -> plt.Figure:
    init_rcparams(
        font_family="DejaVu Sans",
        label_size=17,
        title_size=19,
        tick_size=15.5,
        legend_size=12.6,
        base_size=16,
        figure_title_size=21,
        hide_top_right_spines=True,
        axes_linewidth=1.2,
        lines_linewidth=1.55,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), sharex=True)
    order = (
        UNTRAINED_LABEL,
        *(spec.label for spec in TEXT_MODELS),
        AURISTREAM_LABEL,
    )
    panels = (
        (
            axes[0],
            baseline,
            "Baseline 200",
            "Noise-ceiling-normalized predictivity ($r$)",
        ),
        (
            axes[1],
            natural,
            "Natural Stories",
            "Held-out-story Pearson $r$",
        ),
    )
    for panel_index, (axis, frame, title, ylabel) in enumerate(panels):
        for condition in order:
            draw_curve(axis, frame, condition)
        axis.set_xlim(*RELATIVE_DEPTH_LIMITS)
        axis.set_xticks(RELATIVE_DEPTH_TICKS)
        axis.set_xlabel("Relative layer depth", fontsize=17)
        axis.set_ylabel(ylabel, fontsize=17)
        axis.set_title(title, fontsize=19, weight="bold", pad=10)
        axis.tick_params(direction="out", length=4.0, width=1.0, labelsize=15.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_layer_grid(axis, vertical=True)
        set_y_limits(axis, frame)
        axis.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_index),
            transform=axis.transAxes,
            fontsize=18,
            weight="bold",
            va="bottom",
            ha="left",
        )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.078),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.0,
        handletextpad=0.5,
        fontsize=12.6,
    )
    fig.text(
        0.5,
        0.014,
        "Vertical lines show SEM across 8 participants or 9 held-out stories · Natural Stories text context = 20.48 s.",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.27, wspace=0.27)
    return fig


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    repo = args.auristream_repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    text_summary_path = output_dir / f"{args.stem}_baseline200_text_summary.csv"
    text_participants_path = output_dir / f"{args.stem}_baseline200_text_participants.csv"
    if (
        args.reuse_text_baseline_tables
        and text_summary_path.exists()
        and text_participants_path.exists()
    ):
        text_baseline = pd.read_csv(text_summary_path)
        text_participants = pd.read_csv(text_participants_path)
    else:
        text_baseline, text_participants = load_text_baseline(
            repo,
            args.ncsnr_threshold,
            args.workers,
        )
        text_baseline.to_csv(text_summary_path, index=False)
        text_participants.to_csv(text_participants_path, index=False)

    auri_baseline = load_auristream_summary(
        args.auristream_baseline_summary.expanduser().resolve(),
        "Baseline 200",
    )
    baseline = pd.concat([auri_baseline, text_baseline], ignore_index=True)
    baseline_path = output_dir / f"{args.stem}_baseline200_plot_data.csv"
    baseline.to_csv(baseline_path, index=False)

    auri_natural = load_auristream_summary(
        args.auristream_naturalstories_summary.expanduser().resolve(),
        "Natural Stories",
    )
    text_natural, natural_sources = load_text_naturalstories(
        args.naturalstories_runs.expanduser().resolve()
    )
    natural = pd.concat([auri_natural, text_natural], ignore_index=True)
    natural_path = output_dir / f"{args.stem}_naturalstories_plot_data.csv"
    natural.to_csv(natural_path, index=False)

    combined = pd.concat(
        [
            baseline[
                ["dataset", "condition", "layer", "relative_depth", "mean", "sem", "count", "metric"]
            ],
            natural[
                ["dataset", "condition", "layer", "relative_depth", "mean", "sem", "count", "metric"]
            ],
        ],
        ignore_index=True,
    )
    combined_path = output_dir / f"{args.stem}_plot_data.csv"
    combined.to_csv(combined_path, index=False)

    fig = make_figure(baseline, natural)
    saved = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{args.stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        saved.append(str(path))
    plt.close(fig)

    peaks = []
    for (dataset, condition), frame in combined.groupby(["dataset", "condition"], sort=False):
        peak = frame.loc[frame["mean"].idxmax()]
        peaks.append(
            {
                "dataset": dataset,
                "condition": condition,
                "peak_layer": int(peak["layer"]),
                "peak_relative_depth": float(peak["relative_depth"]),
                "peak_mean": float(peak["mean"]),
                "peak_sem": float(peak["sem"]),
                "n": int(peak["count"]),
            }
        )
    peak_path = output_dir / f"{args.stem}_descriptive_peaks.csv"
    pd.DataFrame(peaks).to_csv(peak_path, index=False)

    manifest = {
        "figure": "AuriStream-MTP and text-model neural alignment across layers",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo),
        "files": saved,
        "plot_data": str(combined_path),
        "descriptive_peaks": str(peak_path),
        "baseline200": {
            "metric": "mean participant-level r_cv_norm across language fROIs",
            "n": 8,
            "text_representation": "last-tok",
            "ncsnr_threshold": args.ncsnr_threshold,
            "language_frois": list(LANGUAGE_FROIS),
            "auri_source": str(args.auristream_baseline_summary),
            "text_summary": str(text_summary_path),
            "text_participants": str(text_participants_path),
        },
        "natural_stories": {
            "metric": "mean held-out-story Pearson r",
            "n": 9,
            "text_context_seconds": 20.48,
            "scoring_trim_seconds": 100,
            "shift": 3,
            "auri_source": str(args.auristream_naturalstories_summary),
            "text_sources": [str(path) for path in natural_sources],
        },
        "text_colors": {spec.label: spec.color for spec in TEXT_MODELS},
        "auristream_color": AURISTREAM_COLOR,
        "untrained_color": LAYER_UNTRAINED_7B_COLOR,
        "note": "Peak table is descriptive; no peak-selected inferential comparison is implied.",
    }
    manifest_path = output_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"saved": saved, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
