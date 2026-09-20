#!/usr/bin/env python3
"""Fit Natural Stories from an AuriStream-alpha audio-representation cache.

AuriStream-alpha owns waveform loading, WavCoch tokenization, model inference,
pooling, and activation caching. This script only validates that cache and
passes each already-binned layer to the same cross-story RidgeCV function used
by the validated XGLM and mT5 reproductions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from reproduce import (
    SHIFT,
    SHIFT_LABEL,
    STORIES,
    STORY_NAMES,
    environment_metadata,
    fit_binned_layer,
    load_pickle,
    save_pickle,
    sha256,
)


DEFAULT_MODEL = "TuKoResearch/AuriStream100M_40Pred_BigAudioDataset_500k"
DEFAULT_STIMSET = "naturalstories_2s_responsegrid"


def canonical_model_key(source_model: str) -> str:
    """Mirror Alpha's model-key convention without importing model code here."""
    return source_model.rstrip("/").split("/")[-1]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    code_repos = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", default=DEFAULT_MODEL)
    parser.add_argument("--stimset_name", default=DEFAULT_STIMSET)
    parser.add_argument("--sent_embed", choices=("mean-tok",), default="mean-tok")
    parser.add_argument("--ACTVCACHEDIR", type=Path, default=code_repos / "AuriStream-alpha" / "ACTV")
    parser.add_argument("--activation_cache", type=Path, default=None)
    parser.add_argument("--stimulus_cache", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=here.parent)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument(
        "--held_out_stories",
        nargs="+",
        default=list(STORIES),
        choices=STORIES,
    )
    args = parser.parse_args()

    cache_base = args.ACTVCACHEDIR / args.source_model / args.sent_embed / args.stimset_name
    if args.activation_cache is None:
        args.activation_cache = Path(f"{cache_base}_actv.pkl")
    if args.stimulus_cache is None:
        args.stimulus_cache = Path(f"{cache_base}_stim.pkl")
    if args.output_dir is None:
        args.output_dir = here / "runs" / f"{canonical_model_key(args.source_model)}_{SHIFT_LABEL}"
    return args


def validate_and_index_cache(
    activations: pd.DataFrame,
    stimuli: pd.DataFrame,
    response,
) -> Dict[str, list[str]]:
    if not isinstance(activations.columns, pd.MultiIndex) or activations.columns.nlevels != 2:
        raise ValueError("Expected Alpha activation columns with MultiIndex (layer, unit)")
    if not activations.index.equals(stimuli.index):
        raise ValueError("Activation and stimulus-cache indices differ")
    if not np.isfinite(activations.to_numpy()).all():
        raise ValueError("Activation cache contains non-finite values")

    stimuli = stimuli.copy()
    stimuli["story_id"] = stimuli["story_id"].astype(str)
    ordered_items: Dict[str, list[str]] = {}
    for story in STORIES:
        story_rows = stimuli.loc[stimuli["story_id"] == story].sort_values("tr_index")
        expected_bins = len(response[STORY_NAMES[story]])
        expected_indices = np.arange(expected_bins)
        observed_indices = story_rows["tr_index"].to_numpy(dtype=int)
        if not np.array_equal(observed_indices, expected_indices):
            raise ValueError(
                f"Story {story}: cache TR indices do not equal 0..{expected_bins - 1}"
            )
        ordered_items[story] = story_rows.index.to_list()
    if sum(map(len, ordered_items.values())) != len(stimuli):
        raise ValueError("Stimulus cache contains rows outside the nine canonical stories")
    return ordered_items


def fit_cache(
    activations: pd.DataFrame,
    response,
    ordered_items: Dict[str, list[str]],
    source_model: str,
    selected_layers: Sequence[int],
    held_out_stories: Sequence[str],
) -> dict:
    layerwise: Dict[int, pd.DataFrame] = {}
    rows = []
    for layer in selected_layers:
        print(f"[fit] model={source_model} layer={layer}", flush=True)
        layer_activations = activations.xs(layer, axis=1, level=0)
        binned = {
            story: layer_activations.loc[item_ids].to_numpy(dtype=np.float32, copy=True)
            for story, item_ids in ordered_items.items()
        }
        layer_rows, layerwise[layer] = fit_binned_layer(
            binned=binned,
            response=response,
            layer=layer,
            model_key=canonical_model_key(source_model),
            held_out_stories=held_out_stories,
        )
        rows.extend(layer_rows)

    long_frame = pd.DataFrame(rows)
    means = long_frame.groupby("layer", as_index=False).agg(
        mean=("r", "mean"),
        std=("r", "std"),
        count=("r", "count"),
    )
    means["sem"] = means["std"] / np.sqrt(means["count"])
    return {"layerwise": layerwise, "long_frame": long_frame, "means": means}


def main() -> None:
    args = parse_args()
    started = time.time()
    activations = pd.read_pickle(args.activation_cache)
    stimuli = pd.read_pickle(args.stimulus_cache)
    response_path = args.data_dir / "response" / f"d_shift_{SHIFT}"
    response = load_pickle(response_path)
    ordered_items = validate_and_index_cache(activations, stimuli, response)

    available_layers = sorted(int(layer) for layer in activations.columns.get_level_values(0).unique())
    selected_layers = available_layers if args.layers is None else list(args.layers)
    unavailable = sorted(set(selected_layers) - set(available_layers))
    if unavailable:
        raise ValueError(f"Unavailable layers {unavailable}; cache has {available_layers}")

    fitted = fit_cache(
        activations=activations,
        response=response,
        ordered_items=ordered_items,
        source_model=args.source_model,
        selected_layers=selected_layers,
        held_out_stories=args.held_out_stories,
    )

    result_dir = args.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_key = canonical_model_key(args.source_model)
    stem = f"cross_story_{SHIFT_LABEL}_{model_key}_reproduced"
    save_pickle(fitted["layerwise"], result_dir / f"{stem}.pkl")
    fitted["long_frame"].to_csv(result_dir / f"{stem}.csv", index=False)
    fitted["means"].to_csv(result_dir / "layer_means.csv", index=False)
    benchmark_summary = fitted["means"].rename(
        columns={
            "mean": "reproduced_mean",
            "std": "reproduced_std",
            "count": "reproduced_count",
            "sem": "reproduced_sem",
        }
    )
    benchmark_summary.to_csv(result_dir / "benchmark_layer_summary.csv", index=False)

    peak = fitted["means"].loc[fitted["means"]["mean"].idxmax()]
    report = [
        f"# Natural Stories {model_key} shift-{SHIFT} benchmark",
        "",
        "- New audio-representation benchmark; there is no released upstream reference for this model.",
        "- Activations were loaded from AuriStream-alpha rather than recomputed in this repository.",
        f"- Peak representation level: {int(peak['layer'])}",
        f"- Peak mean held-out-story Pearson r: {peak['mean']:.9f}",
        "",
        "Layer summaries are in `results/benchmark_layer_summary.csv`.",
    ]
    (args.output_dir / "REPRODUCTION_REPORT.md").write_text("\n".join(report) + "\n")

    alpha_manifest = Path(str(args.activation_cache).replace("_actv.pkl", "_manifest.json"))
    metadata = {
        "model_key": model_key,
        "source_model": args.source_model,
        "analysis": "Natural Stories Alpha audio-representation activation-cache fit",
        "shift": SHIFT,
        "stories": list(STORIES),
        "held_out_stories": list(args.held_out_stories),
        "layers": selected_layers,
        "activation_cache": str(args.activation_cache.resolve()),
        "activation_cache_sha256": sha256(args.activation_cache),
        "stimulus_cache": str(args.stimulus_cache.resolve()),
        "stimulus_cache_sha256": sha256(args.stimulus_cache),
        "alpha_manifest": str(alpha_manifest.resolve()) if alpha_manifest.exists() else None,
        "alpha_extraction": json.loads(alpha_manifest.read_text()) if alpha_manifest.exists() else None,
        "response_file": str(response_path.resolve()),
        "response_sha256": sha256(response_path),
        "activation_shape": list(activations.shape),
        "ridge_cv_implementation": "reproduce.fit_binned_layer",
        "environment": environment_metadata(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved AuriStream benchmark under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
