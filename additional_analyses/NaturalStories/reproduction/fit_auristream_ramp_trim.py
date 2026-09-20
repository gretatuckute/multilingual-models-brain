#!/usr/bin/env python3
"""Fit AuriStream Natural Stories with onset-ramp-corrected test scoring.

Training, RidgeCV, and the full held-out prediction are unchanged; Pearson r is
computed only after the requested number of initial held-out-story bins, as in
the final corrected Natural Stories comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from fit_auristream import canonical_model_key, validate_and_index_cache
from reproduce import (
    RIDGE_ALPHAS,
    SHIFT,
    SHIFT_LABEL,
    STORIES,
    STORY_NAMES,
    environment_metadata,
    load_pickle,
    save_pickle,
    sha256,
)


DEFAULT_MODEL = "TuKoResearch/AuriStream7BDeep_40Pred_BigAudioDataset_500k"
DEFAULT_STIMSET = "naturalstories_2s_responsegrid"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    code_repos = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", default=DEFAULT_MODEL)
    parser.add_argument("--stimset_name", default=DEFAULT_STIMSET)
    parser.add_argument("--sent_embed", choices=("mean-tok",), default="mean-tok")
    parser.add_argument(
        "--ACTVCACHEDIR",
        type=Path,
        default=code_repos / "AuriStream-alpha" / "ACTV",
    )
    parser.add_argument("--activation_cache", type=Path, default=None)
    parser.add_argument("--stimulus_cache", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=here.parent)
    parser.add_argument(
        "--response_file",
        type=Path,
        default=None,
        help=(
            "Optional shifted neural-response pickle. Defaults to the combined "
            "language-network response/d_shift_3 derivative."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--held_out_stories",
        nargs="+",
        default=list(STORIES),
        choices=STORIES,
    )
    parser.add_argument(
        "--test_trim_start_bins",
        type=int,
        default=50,
        help="Initial 2-second held-out-story bins omitted only when scoring r.",
    )
    args = parser.parse_args()

    cache_base = args.ACTVCACHEDIR / args.source_model / args.sent_embed / args.stimset_name
    if args.activation_cache is None:
        args.activation_cache = Path(f"{cache_base}_actv.pkl")
    if args.stimulus_cache is None:
        args.stimulus_cache = Path(f"{cache_base}_stim.pkl")
    if args.response_file is None:
        args.response_file = args.data_dir / "response" / f"d_shift_{SHIFT}"
    if not args.response_file.is_file():
        parser.error(f"Response file does not exist: {args.response_file}")
    if args.test_trim_start_bins < 0:
        parser.error("--test_trim_start_bins must be non-negative")
    return args


def fit_trimmed(
    activations: pd.DataFrame,
    response,
    ordered_items: Dict[str, list[str]],
    source_model: str,
    selected_layers: Sequence[int],
    held_out_stories: Sequence[str],
    test_trim_start_bins: int,
) -> dict:
    model_key = canonical_model_key(source_model)
    layerwise: Dict[int, pd.DataFrame] = {}
    rows = []
    for layer in selected_layers:
        print(f"[fit] model={source_model} layer={layer}", flush=True)
        layer_activations = activations.xs(layer, axis=1, level=0)
        binned = {
            story: layer_activations.loc[item_ids].to_numpy(dtype=np.float32, copy=True)
            for story, item_ids in ordered_items.items()
        }
        layer_rows = []
        for held_out in held_out_stories:
            training_stories = [story for story in STORIES if story != held_out]
            x_train = np.concatenate([binned[story] for story in training_stories])
            y_train = np.concatenate(
                [response[STORY_NAMES[story]] for story in training_stories]
            )
            x_test = binned[held_out]
            y_test = np.asarray(response[STORY_NAMES[held_out]])
            if test_trim_start_bins >= len(y_test) - 2:
                raise ValueError(
                    f"Story {held_out} has {len(y_test)} bins; cannot omit "
                    f"{test_trim_start_bins} and retain at least three"
                )

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_train = x_scaler.fit_transform(x_train)
            x_test = x_scaler.transform(x_test)
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

            regressor = RidgeCV(alphas=RIDGE_ALPHAS)
            regressor.fit(x_train, y_train)
            prediction = regressor.predict(x_test)
            scored_y = y_test[test_trim_start_bins:]
            scored_prediction = prediction[test_trim_start_bins:]
            correlation = float(pearsonr(scored_y, scored_prediction).statistic)
            row = {
                "layer": int(layer),
                "lang": held_out,
                "story": STORY_NAMES[held_out],
                "r": correlation,
                "alpha": float(regressor.alpha_),
                "n_train": int(x_train.shape[0]),
                "n_test_full": int(x_test.shape[0]),
                "n_test_scored": int(scored_y.shape[0]),
                "test_trim_start_bins": int(test_trim_start_bins),
                "test_trim_start_seconds": int(2 * test_trim_start_bins),
            }
            rows.append(row)
            layer_rows.append([held_out, correlation])
            print(
                f"[fit] model={model_key} layer={layer} held_out={held_out} "
                f"r={correlation:.9f} alpha={regressor.alpha_:g} "
                f"scored_bins={len(scored_y)}/{len(y_test)}",
                flush=True,
            )
        layerwise[int(layer)] = pd.DataFrame(layer_rows, columns=["lang", "r"])

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
    response_path = args.response_file
    response = load_pickle(response_path)
    ordered_items = validate_and_index_cache(activations, stimuli, response)

    available_layers = sorted(
        int(layer) for layer in activations.columns.get_level_values(0).unique()
    )
    selected_layers = list(dict.fromkeys(args.layers))
    unavailable = sorted(set(selected_layers) - set(available_layers))
    if unavailable:
        raise ValueError(f"Unavailable layers {unavailable}; cache has {available_layers}")

    fitted = fit_trimmed(
        activations=activations,
        response=response,
        ordered_items=ordered_items,
        source_model=args.source_model,
        selected_layers=selected_layers,
        held_out_stories=args.held_out_stories,
        test_trim_start_bins=args.test_trim_start_bins,
    )

    result_dir = args.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_key = canonical_model_key(args.source_model)
    stem = f"cross_story_{SHIFT_LABEL}_{model_key}_ramptrim_reproduced"
    save_pickle(fitted["layerwise"], result_dir / f"{stem}.pkl")
    fitted["long_frame"].to_csv(result_dir / f"{stem}.csv", index=False)
    fitted["means"].to_csv(result_dir / "layer_means.csv", index=False)

    alpha_manifest = Path(str(args.activation_cache).replace("_actv.pkl", "_manifest.json"))
    metadata = {
        "model_key": model_key,
        "source_model": args.source_model,
        "analysis": "Natural Stories held-out-story onset-ramp scoring diagnostic",
        "exploratory": True,
        "shift": SHIFT,
        "stories": list(STORIES),
        "held_out_stories": list(args.held_out_stories),
        "layers": selected_layers,
        "test_trim_start_bins": args.test_trim_start_bins,
        "test_trim_start_seconds": 2 * args.test_trim_start_bins,
        "training_bins_trimmed": False,
        "prediction_computed_for_full_test_story": True,
        "correlation_scored_after_test_trim_only": True,
        "activation_cache": str(args.activation_cache.resolve()),
        "activation_cache_sha256": sha256(args.activation_cache),
        "stimulus_cache": str(args.stimulus_cache.resolve()),
        "stimulus_cache_sha256": sha256(args.stimulus_cache),
        "alpha_manifest": str(alpha_manifest.resolve()) if alpha_manifest.exists() else None,
        "response_file": str(response_path.resolve()),
        "response_sha256": sha256(response_path),
        "ridge_cv_implementation": "same scalers, alphas, and RidgeCV as reproduce.fit_binned_layer",
        "artifact_reference": "Antonello et al., arXiv:2305.11863v4, Section 3.5",
        "environment": environment_metadata(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved ramp diagnostic under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
