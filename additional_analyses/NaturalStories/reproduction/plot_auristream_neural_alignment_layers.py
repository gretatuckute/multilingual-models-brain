#!/usr/bin/env python3
"""Plot AuriStream-MTP 7B 40Pred neural alignment across model depth.

The left panel aggregates the Baseline 200 language-network encoding results
directly from the per-participant/per-layer CSV files.  The right panel reads
the completed Natural Stories 100-second ramp-trim summaries.  Both panels
compare the trained checkpoint with its corrected untrained checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_AURISTREAM_REPO = Path(
    "/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/AuriStream-alpha"
)
if (SCRIPT_PATH.parents[1] / "auristream").is_dir():
    AURISTREAM_REPO = SCRIPT_PATH.parents[1]
    DEFAULT_OUTPUT_DIR = (
        AURISTREAM_REPO
        / "results_figures"
        / "functional_organization_across_layers"
        / "2026-09-20"
        / "all_models_by_task"
    )
else:
    AURISTREAM_REPO = Path(
        os.environ.get("AURISTREAM_REPO", DEFAULT_AURISTREAM_REPO)
    ).expanduser().resolve()
    DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "figures" / "neural_alignment_7b40pred"
if str(AURISTREAM_REPO) not in sys.path:
    sys.path.insert(0, str(AURISTREAM_REPO))

from auristream.plotting import (  # noqa: E402
    LAYER_UNTRAINED_7B_COLOR,
    LAYER_UNTRAINED_ERROR_ALPHA,
    LAYER_UNTRAINED_LINE_ALPHA,
    LAYER_UNTRAINED_MARKER_ALPHA,
    add_layer_grid,
    init_rcparams,
    model_color,
)


MODEL_KEY = "AuriStream7BDeep_40Pred_BigAudioDataset_500k"
MODEL_SOURCE = f"TuKoResearch_{MODEL_KEY}"
UNTRAINED_SOURCE = f"{MODEL_SOURCE}-randinit"
N_STATES = 97
EXPECTED_LAYERS = np.arange(N_STATES)
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

DEFAULT_NATURALSTORIES_RUNS = Path(
    "/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/"
    "multilingual-models-brain-auristream/additional_analyses/"
    "NaturalStories/reproduction/runs"
)
NATURALSTORIES_RUNS = {
    "AuriStream-MTP 7B 40Pred": (
        "AuriStream7BDeep_40Pred_BigAudioDataset_500k_"
        "ramptrim100s_alllayers_3shift"
    ),
    "7B 40Pred untrained": (
        "AuriStream7BDeep_40Pred_BigAudioDataset_500k-randinit_"
        "ramptrim100s_alllayers_3shift"
    ),
}

TRAINED_LABEL = "AuriStream-MTP 7B 40Pred"
UNTRAINED_LABEL = "7B 40Pred untrained"
TRAINED_COLOR = model_color(MODEL_KEY)
UNTRAINED_COLOR = LAYER_UNTRAINED_7B_COLOR
RELATIVE_DEPTH_TICKS = np.linspace(0.0, 1.0, 6)
RELATIVE_DEPTH_LIMITS = (-0.015, 1.015)

LINEWIDTH = 1.6
MARKERSIZE = 7.5
TRAINED_LINE_ALPHA = 0.40
TRAINED_MARKER_ALPHA = 0.78
UNTRAINED_LINE_ALPHA = LAYER_UNTRAINED_LINE_ALPHA
UNTRAINED_MARKER_ALPHA = LAYER_UNTRAINED_MARKER_ALPHA
ERROR_LINEWIDTH = 0.82
TRAINED_ERROR_ALPHA = 0.26
UNTRAINED_ERROR_ALPHA = LAYER_UNTRAINED_ERROR_ALPHA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auristream-repo", type=Path, default=AURISTREAM_REPO)
    parser.add_argument(
        "--naturalstories-runs",
        type=Path,
        default=DEFAULT_NATURALSTORIES_RUNS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--stem",
        default="figure5_exploration_neural_alignment_7b40pred",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Processes used to read the compressed Baseline 200 result grid.",
    )
    parser.add_argument("--ncsnr-threshold", type=float, default=0.4)
    parser.add_argument(
        "--reuse-baseline-tables",
        action="store_true",
        help="Reuse the two saved Baseline 200 CSV tables if they exist.",
    )
    return parser.parse_args()


def _baseline_result_paths(repo: Path, result_dir: str) -> list[Path]:
    root = repo / "results_neural" / "inv_neural_pred_paral" / result_dir
    paths = sorted(
        path for path in root.glob("*.csv.gz") if "permutesm" not in path.name
    )
    expected = len(EXPECTED_UIDS) * N_STATES
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} non-permuted Baseline 200 files in {root}; "
            f"found {len(paths)}"
        )
    return paths


def _read_baseline_file(arguments: tuple[str, str, float, str]) -> dict[str, object]:
    path_text, expected_source, ncsnr_threshold, condition = arguments
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
    if uid not in EXPECTED_UIDS or layer not in EXPECTED_LAYERS:
        raise RuntimeError(f"Unexpected participant/layer in {path}: {uid}, {layer}")
    selected = source[
        source["ncsnr"].gt(ncsnr_threshold)
        & source[ROI_COLUMN].isin(LANGUAGE_FROIS)
    ]
    if selected.empty:
        raise RuntimeError(f"No language-network voxels survived selection in {path}")
    values = selected[["r_cv", "r_cv_norm"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite selected encoding score in {path}")
    return {
        "dataset": "Baseline 200",
        "condition": condition,
        "uid": uid,
        "layer": layer,
        "relative_depth": layer / (N_STATES - 1),
        "raw_r": float(selected["r_cv"].mean()),
        "normalized_r": float(selected["r_cv_norm"].mean()),
        "n_voxels": int(len(selected)),
        "source_path": str(path.resolve()),
    }


def _validate_participant_grid(frame: pd.DataFrame, condition: str) -> None:
    subset = frame[frame["condition"].eq(condition)]
    if subset.duplicated(["uid", "layer"]).any():
        raise RuntimeError(f"Duplicate Baseline 200 participant/layer: {condition}")
    observed_uids = tuple(sorted(subset["uid"].unique()))
    if observed_uids != tuple(sorted(EXPECTED_UIDS)):
        raise RuntimeError(
            f"Unexpected Baseline 200 participants for {condition}: {observed_uids}"
        )
    for uid in EXPECTED_UIDS:
        layers = np.sort(subset.loc[subset["uid"].eq(uid), "layer"].unique())
        if not np.array_equal(layers, EXPECTED_LAYERS):
            missing = np.setdiff1d(EXPECTED_LAYERS, layers).tolist()
            raise RuntimeError(f"Missing {condition} layers for {uid}: {missing}")
        voxel_counts = subset.loc[subset["uid"].eq(uid), "n_voxels"]
        if voxel_counts.nunique() != 1:
            raise RuntimeError(
                f"Language-network voxel count changes over layers: {condition}, {uid}"
            )


def load_baseline200(
    repo: Path,
    ncsnr_threshold: float,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs: list[tuple[str, str, float, str]] = []
    specs = (
        (MODEL_KEY, MODEL_SOURCE, TRAINED_LABEL),
        (f"{MODEL_KEY}-randinit", UNTRAINED_SOURCE, UNTRAINED_LABEL),
    )
    for result_dir, source_model, condition in specs:
        jobs.extend(
            (str(path), source_model, ncsnr_threshold, condition)
            for path in _baseline_result_paths(repo, result_dir)
        )
    if workers <= 1:
        rows = [_read_baseline_file(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_read_baseline_file, jobs, chunksize=1))
    participants = pd.DataFrame(rows).sort_values(
        ["condition", "uid", "layer"]
    ).reset_index(drop=True)
    for condition in (TRAINED_LABEL, UNTRAINED_LABEL):
        _validate_participant_grid(participants, condition)
    summary = (
        participants.groupby(
            ["dataset", "condition", "layer", "relative_depth"],
            as_index=False,
            sort=True,
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
        raise RuntimeError("Baseline 200 summary does not contain eight participants")
    summary["metric"] = "mean participant-level r_cv_norm in language fROIs"
    return summary, participants


def load_naturalstories(runs_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    rows = []
    sources = []
    for condition, run_name in NATURALSTORIES_RUNS.items():
        run_dir = runs_root / run_name
        manifest_path = run_dir / "run_manifest.json"
        summary_path = run_dir / "results" / "layer_means.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_source = (
            MODEL_SOURCE.replace("_", "/", 1)
            if condition == TRAINED_LABEL
            else UNTRAINED_SOURCE.replace("_", "/", 1)
        )
        if manifest.get("source_model") != expected_source:
            raise RuntimeError(
                f"Unexpected Natural Stories checkpoint in {manifest_path}: "
                f"{manifest.get('source_model')!r}"
            )
        if manifest.get("layers") != EXPECTED_LAYERS.tolist():
            raise RuntimeError(f"Expected Natural Stories layers 0--96: {manifest_path}")
        if manifest.get("test_trim_start_seconds") != 100:
            raise RuntimeError(f"Expected a 100-second scoring trim: {manifest_path}")
        if not manifest.get("correlation_scored_after_test_trim_only", False):
            raise RuntimeError(f"Scoring-only trim is not confirmed: {manifest_path}")
        if manifest.get("training_bins_trimmed") is not False:
            raise RuntimeError(f"Training bins must remain untrimmed: {manifest_path}")
        if manifest.get("shift") != 3:
            raise RuntimeError(f"Expected shift=3: {manifest_path}")
        frame = pd.read_csv(summary_path).sort_values("layer").reset_index(drop=True)
        required = {"layer", "mean", "sem", "count"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Missing columns {sorted(missing)} in {summary_path}")
        if not np.array_equal(frame["layer"].to_numpy(), EXPECTED_LAYERS):
            raise RuntimeError(f"Incomplete Natural Stories layer grid: {summary_path}")
        if not frame["count"].eq(9).all():
            raise RuntimeError(f"Expected nine held-out stories: {summary_path}")
        if not np.isfinite(frame[["mean", "sem"]].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite Natural Stories score: {summary_path}")
        frame = frame[["layer", "mean", "sem", "count"]].copy()
        frame.insert(0, "condition", condition)
        frame.insert(0, "dataset", "Natural Stories")
        frame["relative_depth"] = frame["layer"] / (N_STATES - 1)
        frame["metric"] = "mean held-out-story Pearson r"
        rows.append(frame)
        sources.extend([manifest_path.resolve(), summary_path.resolve()])
    return pd.concat(rows, ignore_index=True), sources


def _draw_curve(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    color: str,
    line_alpha: float,
    marker_alpha: float,
    error_alpha: float,
    zorder: float,
) -> None:
    curve = frame.sort_values("layer")
    x = curve["relative_depth"].to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    sem = curve["sem"].to_numpy(dtype=float)
    axis.errorbar(
        x,
        y,
        yerr=sem,
        fmt="none",
        ecolor=to_rgba(color, error_alpha),
        elinewidth=ERROR_LINEWIDTH,
        capsize=0,
        zorder=zorder,
    )
    axis.plot(
        x,
        y,
        color=to_rgba(color, line_alpha),
        linewidth=LINEWIDTH,
        linestyle="-",
        marker="s",
        markersize=MARKERSIZE,
        markerfacecolor=to_rgba(color, marker_alpha),
        markeredgecolor=to_rgba(color, marker_alpha),
        zorder=zorder + 1,
    )


def _set_y_limits(axis: plt.Axes, frame: pd.DataFrame) -> None:
    low = float((frame["mean"] - frame["sem"]).min())
    high = float((frame["mean"] + frame["sem"]).max())
    span = max(high - low, 0.01)
    axis.set_ylim(low - 0.07 * span, high + 0.09 * span)


def make_figure(baseline: pd.DataFrame, natural: pd.DataFrame) -> plt.Figure:
    init_rcparams(
        font_family="DejaVu Sans",
        label_size=17,
        title_size=19,
        tick_size=15.5,
        legend_size=13.2,
        base_size=16,
        figure_title_size=21,
        hide_top_right_spines=True,
        axes_linewidth=1.2,
        lines_linewidth=LINEWIDTH,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), sharex=True)
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
        _draw_curve(
            axis,
            frame[frame["condition"].eq(UNTRAINED_LABEL)],
            color=UNTRAINED_COLOR,
            line_alpha=UNTRAINED_LINE_ALPHA,
            marker_alpha=UNTRAINED_MARKER_ALPHA,
            error_alpha=UNTRAINED_ERROR_ALPHA,
            zorder=1,
        )
        _draw_curve(
            axis,
            frame[frame["condition"].eq(TRAINED_LABEL)],
            color=TRAINED_COLOR,
            line_alpha=TRAINED_LINE_ALPHA,
            marker_alpha=TRAINED_MARKER_ALPHA,
            error_alpha=TRAINED_ERROR_ALPHA,
            zorder=3,
        )
        axis.set_xlim(*RELATIVE_DEPTH_LIMITS)
        axis.set_xticks(RELATIVE_DEPTH_TICKS)
        axis.set_xlabel("Relative layer depth", fontsize=17)
        axis.set_ylabel(ylabel, fontsize=17)
        axis.set_title(title, fontsize=19, weight="bold", pad=10)
        axis.tick_params(direction="out", length=4.0, width=1.0, labelsize=15.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_layer_grid(axis, vertical=True)
        _set_y_limits(axis, frame)
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
    handles = [
        Line2D(
            [0],
            [0],
            color=TRAINED_COLOR,
            linewidth=LINEWIDTH,
            marker="s",
            markersize=MARKERSIZE,
            label=TRAINED_LABEL,
        ),
        Line2D(
            [0],
            [0],
            color=UNTRAINED_COLOR,
            linewidth=LINEWIDTH,
            marker="s",
            markersize=MARKERSIZE,
            label=UNTRAINED_LABEL,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.2,
        handletextpad=0.55,
        fontsize=13.2,
    )
    fig.text(
        0.5,
        0.014,
        "Vertical lines show SEM across 8 participants (Baseline 200) or 9 held-out stories (Natural Stories).",
        ha="center",
        va="bottom",
        fontsize=11.4,
        color="#333333",
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.245, wspace=0.27)
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
    runs_root = args.naturalstories_runs.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary_path = output_dir / f"{args.stem}_baseline200_summary.csv"
    baseline_participants_path = output_dir / f"{args.stem}_baseline200_participants.csv"
    if (
        args.reuse_baseline_tables
        and baseline_summary_path.exists()
        and baseline_participants_path.exists()
    ):
        baseline = pd.read_csv(baseline_summary_path)
        participants = pd.read_csv(baseline_participants_path)
        for condition in (TRAINED_LABEL, UNTRAINED_LABEL):
            _validate_participant_grid(participants, condition)
    else:
        baseline, participants = load_baseline200(
            repo,
            args.ncsnr_threshold,
            args.workers,
        )
        baseline.to_csv(baseline_summary_path, index=False)
        participants.to_csv(baseline_participants_path, index=False)
    natural, natural_sources = load_naturalstories(runs_root)
    natural_path = output_dir / f"{args.stem}_naturalstories_summary.csv"
    natural.to_csv(natural_path, index=False)
    combined = pd.concat(
        [
            baseline[
                [
                    "dataset",
                    "condition",
                    "layer",
                    "relative_depth",
                    "mean",
                    "sem",
                    "count",
                    "metric",
                ]
            ],
            natural[
                [
                    "dataset",
                    "condition",
                    "layer",
                    "relative_depth",
                    "mean",
                    "sem",
                    "count",
                    "metric",
                ]
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

    baseline_manifest = (
        repo
        / "ACTV"
        / "TuKoResearch"
        / f"{MODEL_KEY}-randinit"
        / "mean-tok"
        / "baseline200_manifest.json"
    ).resolve()
    manifest = {
        "figure": "AuriStream-MTP 7B 40Pred neural alignment across layers",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo),
        "files": saved,
        "plot_data": str(combined_path),
        "baseline200_summary": str(baseline_summary_path),
        "baseline200_participants": str(baseline_participants_path),
        "naturalstories_summary": str(natural_path),
        "baseline200": {
            "metric": "mean participant-level r_cv_norm across selected language fROIs",
            "center": "mean across participants",
            "error": "SEM across participants",
            "n": len(EXPECTED_UIDS),
            "uids": list(EXPECTED_UIDS),
            "ncsnr_threshold": args.ncsnr_threshold,
            "language_frois": list(LANGUAGE_FROIS),
            "corrected_untrained_checkpoint_manifest": str(baseline_manifest),
            "trained_result_dir": str(
                (repo / "results_neural" / "inv_neural_pred_paral" / MODEL_KEY).resolve()
            ),
            "untrained_result_dir": str(
                (
                    repo
                    / "results_neural"
                    / "inv_neural_pred_paral"
                    / f"{MODEL_KEY}-randinit"
                ).resolve()
            ),
        },
        "natural_stories": {
            "metric": "mean held-out-story Pearson r",
            "center": "mean across held-out stories",
            "error": "SEM across held-out stories",
            "n": 9,
            "scoring_trim_seconds": 100,
            "shift": 3,
            "sources": [str(path) for path in natural_sources],
        },
        "style": {
            "trained_color": TRAINED_COLOR,
            "untrained_color": UNTRAINED_COLOR,
            "marker": "square at every measured layer",
            "connector": "thin solid line",
            "error_bars": "uncapped vertical lines",
            "font": "DejaVu Sans",
        },
    }
    manifest_path = output_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"saved": saved, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
