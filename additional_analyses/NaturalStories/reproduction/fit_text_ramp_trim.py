#!/usr/bin/env python3
"""Fit a text model on Natural Stories with onset-ramp-trimmed test scoring.

The extracted word embeddings, two-second binning, leave-one-story-out folds,
standardization, RidgeCV alphas, and full-story predictions are identical to
``reproduce.py``.  Only the reported held-out-story Pearson correlation is
changed: scoring begins after a configurable number of initial two-second bins.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from reproduce import (
    MODEL_SPECS,
    RIDGE_ALPHAS,
    SHIFT,
    SHIFT_LABEL,
    STORIES,
    STORY_NAMES,
    bin_words_to_two_seconds,
    embedding_path,
    environment_metadata,
    load_pickle,
    save_pickle,
    sha256,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=here.parent,
        help="NaturalStories directory containing transcribed/ and response/.",
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=None,
        help="Run directory containing the already extracted embeddings.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument(
        "--held-out-stories",
        nargs="+",
        default=list(STORIES),
        choices=STORIES,
    )
    parser.add_argument(
        "--test-trim-start-bins",
        type=int,
        default=50,
        help="Initial two-second held-out-story bins omitted only from Pearson r.",
    )
    args = parser.parse_args()
    if args.source_run is None:
        args.source_run = here / "runs" / f"{args.model_key}_{SHIFT_LABEL}"
    if args.test_trim_start_bins < 0:
        parser.error("--test-trim-start-bins must be non-negative")
    return args


def fit_binned_layer_trimmed(
    binned: Mapping[str, np.ndarray],
    response: Mapping[str, np.ndarray],
    layer: int,
    model_key: str,
    held_out_stories: Sequence[str],
    test_trim_start_bins: int,
) -> tuple[List[dict], pd.DataFrame]:
    rows: List[dict] = []
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
    return rows, pd.DataFrame(layer_rows, columns=["lang", "r"])


def main() -> None:
    args = parse_args()
    started = time.time()
    spec = MODEL_SPECS[args.model_key]
    response_path = args.data_dir / "response" / f"d_shift_{SHIFT}"
    response = load_pickle(response_path)
    payloads = {
        story: load_pickle(embedding_path(args.source_run, spec, story))
        for story in STORIES
    }
    context_keys = (
        "causal_context_mode",
        "context_words",
        "context_seconds",
        "target_bin_seconds",
        "context_window_anchor",
        "context_boundary_rule",
    )
    context_metadata = {
        key: payloads[STORIES[0]].get(key) for key in context_keys
    }
    for story, payload in payloads.items():
        observed = {key: payload.get(key) for key in context_keys}
        if observed != context_metadata:
            raise ValueError(
                f"Story {story} has inconsistent context metadata: "
                f"{observed} != {context_metadata}"
            )
    transcripts = {
        story: pd.read_csv(args.data_dir / "transcribed" / f"{story}.csv")
        for story in STORIES
    }

    available_layers = sorted(int(layer) for layer in payloads[STORIES[0]]["layers"])
    selected_layers = available_layers if args.layers is None else list(dict.fromkeys(args.layers))
    unavailable = sorted(set(selected_layers) - set(available_layers))
    if unavailable:
        raise ValueError(
            f"Unavailable layers {unavailable}; embeddings have {available_layers}"
        )
    for story, payload in payloads.items():
        story_layers = sorted(int(layer) for layer in payload["layers"])
        if story_layers != available_layers:
            raise ValueError(f"Story {story} has inconsistent layers: {story_layers}")

    layerwise: Dict[int, pd.DataFrame] = {}
    rows: List[dict] = []
    for layer in selected_layers:
        print(f"[fit] model={spec.key} layer={layer}", flush=True)
        binned = {
            story: bin_words_to_two_seconds(
                payloads[story]["layers"][layer],
                transcripts[story],
                len(response[STORY_NAMES[story]]),
            )
            for story in STORIES
        }
        layer_rows, layerwise[layer] = fit_binned_layer_trimmed(
            binned=binned,
            response=response,
            layer=layer,
            model_key=spec.key,
            held_out_stories=args.held_out_stories,
            test_trim_start_bins=args.test_trim_start_bins,
        )
        rows.extend(layer_rows)

    result_dir = args.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cross_story_{SHIFT_LABEL}_{spec.label}_ramptrim_reproduced"
    save_pickle(layerwise, result_dir / f"{stem}.pkl")
    long_frame = pd.DataFrame(rows)
    long_frame.to_csv(result_dir / f"{stem}.csv", index=False)
    means = long_frame.groupby("layer", as_index=False).agg(
        mean=("r", "mean"),
        std=("r", "std"),
        count=("r", "count"),
    )
    means["sem"] = means["std"] / np.sqrt(means["count"])
    means.to_csv(result_dir / "layer_means.csv", index=False)

    embedding_paths = {
        story: embedding_path(args.source_run, spec, story).resolve()
        for story in STORIES
    }
    metadata = {
        "model_key": spec.key,
        "model": spec.model_id,
        "extraction": spec.extraction,
        **context_metadata,
        "analysis": "Natural Stories held-out-story onset-ramp-corrected scoring",
        "shift": SHIFT,
        "stories": list(STORIES),
        "held_out_stories": list(args.held_out_stories),
        "layers": selected_layers,
        "test_trim_start_bins": args.test_trim_start_bins,
        "test_trim_start_seconds": 2 * args.test_trim_start_bins,
        "training_bins_trimmed": False,
        "prediction_computed_for_full_test_story": True,
        "correlation_scored_after_test_trim_only": True,
        "source_run": str(args.source_run.resolve()),
        "embedding_files": {
            story: str(path) for story, path in embedding_paths.items()
        },
        "embedding_sha256": {
            story: sha256(path) for story, path in embedding_paths.items()
        },
        "response_file": str(response_path.resolve()),
        "response_sha256": sha256(response_path),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "environment": environment_metadata(),
        "elapsed_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved ramp-corrected results under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
