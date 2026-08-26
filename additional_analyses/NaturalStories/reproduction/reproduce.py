#!/usr/bin/env python3
"""Reproduce released Natural Stories layer-wise encoding results.

The runner leaves Andrea de Varda's original analysis files unchanged. It
reimplements their model-specific word extraction, two-second temporal
binning, and leave-one-story-out RidgeCV evaluation for two published models:
XGLM-small and mT5-large. It also supports GPT-2 XL as a new autoregressive
comparison without a released Natural Stories reference. Generated runs live
below ``reproduction/runs/``, which is ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import sklearn
import torch
import transformers
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MT5EncoderModel,
    T5Tokenizer,
    XGLMForCausalLM,
    XGLMTokenizer,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    display_name: str
    model_id: str
    extraction: str
    reference: str
    context_words: int | None = None


MODEL_SPECS = {
    "xglm_small": ModelSpec(
        key="xglm_small",
        label="xglm_small",
        display_name="XGLM-small",
        model_id="facebook/xglm-564M",
        extraction="full_story_causal",
        reference="story_level_pickle",
    ),
    "mt5_large": ModelSpec(
        key="mt5_large",
        label="mt5_large",
        display_name="mT5-large",
        model_id="google/mt5-large",
        extraction="full_story_bidirectional",
        reference="layer_summary_csv",
    ),
    "gpt2_xl": ModelSpec(
        key="gpt2_xl",
        label="gpt2_xl",
        display_name="GPT-2 XL",
        model_id="openai-community/gpt2-xl",
        extraction="trailing_word_context_causal",
        reference="none",
        context_words=100,
    ),
}

SHIFT = 3
SHIFT_LABEL = f"{SHIFT}shift"
STORIES = ("1", "2", "3", "4", "5", "6", "7", "9", "10")
STORY_NAMES = {
    "1": "boar",
    "2": "aqua",
    "3": "matchstickseller",
    "4": "kingofbirds",
    "5": "elvis",
    "6": "mrsticky",
    "7": "highschool",
    "9": "tulips",
    "10": "tree",
}
RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), default="xglm_small")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=here.parent,
        help="NaturalStories directory containing transcribed/, response/, and results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: reproduction/runs/<model-key>_3shift.",
    )
    parser.add_argument("--model-id", default=None, help="Optional checkpoint override.")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stories", nargs="+", default=list(STORIES), choices=STORIES)
    parser.add_argument(
        "--held-out-stories",
        nargs="+",
        default=list(STORIES),
        choices=STORIES,
        help="Held-out folds to fit. Training still uses all other canonical stories.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layers to fit after extraction; default is every extracted level.",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--reference-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.extract_only and args.fit_only:
        parser.error("--extract-only and --fit-only are mutually exclusive")
    if args.output_dir is None:
        args.output_dir = here / "runs" / f"{args.model_key}_{SHIFT_LABEL}"
    return args


def save_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def embedding_path(output_dir: Path, spec: ModelSpec, story: str) -> Path:
    return output_dir / "embeddings" / f"{spec.label}_{story}.pkl"


def transcript_words(data_dir: Path, story: str) -> tuple[pd.DataFrame, List[str]]:
    transcript_path = data_dir / "transcribed" / f"{story}.csv"
    frame = pd.read_csv(transcript_path)
    words = frame["text"].str.cat(sep=" ").split()
    if not words:
        raise ValueError(f"No transcript words found in {transcript_path}")
    if len(words) != len(frame):
        raise ValueError(
            f"Story {story}: transcript produced {len(words)} whitespace-delimited words "
            f"from {len(frame)} timestamp rows"
        )
    return frame, words


def word_token_lengths(words: Sequence[str], tokenizer) -> List[int]:
    lengths: List[int] = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        if not tokens:
            raise ValueError(f"Tokenizer produced no token for word {word!r}")
        # The original code removes the SentencePiece/WordPiece prefix from the
        # first token only for matching. It does not change the token count.
        lengths.append(len(tokens))
    return lengths


def extraction_payload(
    story: str,
    data_dir: Path,
    spec: ModelSpec,
    model,
    word_count: int,
    model_token_count: int | None,
    by_layer: Mapping[int, np.ndarray],
) -> dict:
    transcript_path = data_dir / "transcribed" / f"{story}.csv"
    return {
        "story": story,
        "model_key": spec.key,
        "model": str(getattr(model.config, "_name_or_path", spec.model_id)),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "extraction": spec.extraction,
        "context_words": spec.context_words,
        "transcript_path": str(transcript_path.resolve()),
        "transcript_sha256": sha256(transcript_path),
        "word_count": word_count,
        "model_token_count_with_special": model_token_count,
        "layers": dict(by_layer),
    }


@torch.inference_mode()
def extract_xglm_story(
    story: str,
    data_dir: Path,
    output_dir: Path,
    spec: ModelSpec,
    tokenizer: XGLMTokenizer,
    model: XGLMForCausalLM,
    device: torch.device,
    overwrite: bool,
) -> dict:
    destination = embedding_path(output_dir, spec, story)
    if destination.exists() and not overwrite:
        print(f"[extract] story={story}: using {destination}", flush=True)
        return load_pickle(destination)

    _, words = transcript_words(data_dir, story)
    input_ids = tokenizer.encode(" ".join(words), return_tensors="pt")
    print(
        f"[extract] model={spec.key} story={story} words={len(words)} "
        f"model_tokens={input_ids.shape[1]}",
        flush=True,
    )
    outputs = model(input_ids.to(device), output_hidden_states=True, return_dict=True)

    # XGLM prepends one special token (emb_start=1 in the original code).
    token_embeddings = [state[0, 1:, :].detach().cpu().float().numpy() for state in outputs.hidden_states]
    lengths = word_token_lengths(words, tokenizer)
    expected_tokens = sum(lengths)
    observed_tokens = token_embeddings[0].shape[0]
    if observed_tokens != expected_tokens:
        raise ValueError(
            f"Story {story}: {observed_tokens} non-special embeddings for "
            f"{expected_tokens} transcript subword tokens"
        )

    by_layer: Dict[int, np.ndarray] = {}
    for layer, embeddings in enumerate(token_embeddings):
        word_vectors = []
        cursor = 0
        for length in lengths:
            word_vectors.append(embeddings[cursor : cursor + length].mean(axis=0))
            cursor += length
        by_layer[layer] = np.vstack(word_vectors).astype(np.float32, copy=False)

    payload = extraction_payload(
        story,
        data_dir,
        spec,
        model,
        len(words),
        int(input_ids.shape[1]),
        by_layer,
    )
    save_pickle(payload, destination)
    print(f"[extract] story={story}: saved {destination}", flush=True)
    return payload


@torch.inference_mode()
def extract_mt5_story(
    story: str,
    data_dir: Path,
    output_dir: Path,
    spec: ModelSpec,
    tokenizer: T5Tokenizer,
    model: MT5EncoderModel,
    device: torch.device,
    overwrite: bool,
) -> dict:
    destination = embedding_path(output_dir, spec, story)
    if destination.exists() and not overwrite:
        print(f"[extract] story={story}: using {destination}", flush=True)
        return load_pickle(destination)

    _, words = transcript_words(data_dir, story)
    encoder = model.get_encoder()
    inputs = tokenizer(" ".join(words), return_tensors="pt")
    print(
        f"[extract] model={spec.key} story={story} words={len(words)} "
        f"model_tokens={inputs['input_ids'].shape[1]} full_story_bidirectional=True",
        flush=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = encoder(**inputs, output_hidden_states=True, return_dict=True)

    # The released curve is reproduced only by full-story bidirectional mT5
    # embeddings. The final EOS token is removed, matching emb_end=-1 in the
    # upstream code, and subword vectors are averaged within each transcript word.
    token_embeddings = [
        state[0, :-1, :].detach().cpu().float().numpy()
        for state in outputs.hidden_states
    ]
    lengths = word_token_lengths(words, tokenizer)
    expected_tokens = sum(lengths)
    observed_tokens = token_embeddings[0].shape[0]
    if observed_tokens != expected_tokens:
        raise ValueError(
            f"Story {story}: {observed_tokens} non-special embeddings for "
            f"{expected_tokens} transcript subword tokens"
        )

    by_layer: Dict[int, np.ndarray] = {}
    for level, embeddings in enumerate(token_embeddings):
        word_vectors = []
        cursor = 0
        for length in lengths:
            word_vectors.append(embeddings[cursor : cursor + length].mean(axis=0))
            cursor += length
        by_layer[level] = np.vstack(word_vectors).astype(np.float32, copy=False)
    payload = extraction_payload(
        story,
        data_dir,
        spec,
        model,
        len(words),
        int(inputs["input_ids"].shape[1]),
        by_layer,
    )
    save_pickle(payload, destination)
    print(f"[extract] story={story}: saved {destination}", flush=True)
    return payload


@torch.inference_mode()
def extract_gpt2_story(
    story: str,
    data_dir: Path,
    output_dir: Path,
    spec: ModelSpec,
    tokenizer,
    model,
    device: torch.device,
    overwrite: bool,
) -> dict:
    """Extract each word from a causal trailing-context GPT-2 forward pass."""
    destination = embedding_path(output_dir, spec, story)
    if destination.exists() and not overwrite:
        print(f"[extract] story={story}: using {destination}", flush=True)
        return load_pickle(destination)

    _, words = transcript_words(data_dir, story)
    context_words = int(spec.context_words or 100)
    number_of_levels = int(model.config.n_layer) + 1
    vectors: Dict[int, List[np.ndarray]] = {level: [] for level in range(number_of_levels)}
    print(
        f"[extract] model={spec.key} story={story} words={len(words)} "
        f"trailing_context_words={context_words} causal_attention=True",
        flush=True,
    )

    batch_size = 16
    for batch_start in range(0, len(words), batch_size):
        word_indices = range(batch_start, min(batch_start + batch_size, len(words)))
        chunk_texts = []
        final_word_starts = []
        for word_index in word_indices:
            start = max(0, word_index - context_words + 1)
            chunk_words = words[start : word_index + 1]
            chunk_texts.append(" ".join(chunk_words))
            final_word_start = len(" ".join(chunk_words[:-1]))
            if len(chunk_words) > 1:
                final_word_start += 1
            final_word_starts.append(final_word_start)

        encoded = tokenizer(
            chunk_texts,
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        target_masks = []
        for row_index, (word_index, chunk_text, final_word_start) in enumerate(
            zip(word_indices, chunk_texts, final_word_starts)
        ):
            row_offsets = offsets[row_index]
            target_mask = (row_offsets[:, 1] > final_word_start) & (
                row_offsets[:, 0] < len(chunk_text)
            )
            if not bool(target_mask.any()):
                raise ValueError(
                    f"Story {story}, word {word_index}: no GPT-2 token overlaps target "
                    f"word {words[word_index]!r}"
                )
            target_masks.append(target_mask)
        if encoded["input_ids"].shape[1] > int(model.config.n_positions):
            raise ValueError(
                f"GPT-2 context has {encoded['input_ids'].shape[1]} tokens, exceeding "
                f"the model limit of {model.config.n_positions}"
            )
        inputs = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        for level, state in enumerate(outputs.hidden_states):
            for row_index, target_mask in enumerate(target_masks):
                vector = state[row_index, target_mask.to(device), :].mean(dim=0)
                vectors[level].append(vector.detach().cpu().float().numpy())

        words_complete = min(batch_start + batch_size, len(words))
        print(
            f"[extract] model={spec.key} story={story} "
            f"words_complete={words_complete}/{len(words)}",
            flush=True,
        )

    by_layer = {
        level: np.vstack(level_vectors).astype(np.float32, copy=False)
        for level, level_vectors in vectors.items()
    }
    payload = extraction_payload(
        story,
        data_dir,
        spec,
        model,
        len(words),
        None,
        by_layer,
    )
    save_pickle(payload, destination)
    print(f"[extract] story={story}: saved {destination}", flush=True)
    return payload


def load_model(spec: ModelSpec, model_id: str, revision: str, device: torch.device):
    if spec.key == "xglm_small":
        tokenizer = XGLMTokenizer.from_pretrained(model_id, revision=revision)
        model = XGLMForCausalLM.from_pretrained(model_id, revision=revision)
    elif spec.key == "mt5_large":
        tokenizer = T5Tokenizer.from_pretrained(model_id, revision=revision)
        model = MT5EncoderModel.from_pretrained(model_id, revision=revision)
    elif spec.key == "gpt2_xl":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported model {spec.key}")
    model.eval().to(device)
    return tokenizer, model


def extract_story(
    story: str,
    data_dir: Path,
    output_dir: Path,
    spec: ModelSpec,
    tokenizer,
    model,
    device: torch.device,
    overwrite: bool,
) -> dict:
    if spec.key == "xglm_small":
        return extract_xglm_story(
            story, data_dir, output_dir, spec, tokenizer, model, device, overwrite
        )
    if spec.key == "mt5_large":
        return extract_mt5_story(
            story, data_dir, output_dir, spec, tokenizer, model, device, overwrite
        )
    if spec.key == "gpt2_xl":
        return extract_gpt2_story(
            story, data_dir, output_dir, spec, tokenizer, model, device, overwrite
        )
    raise ValueError(f"Unsupported model {spec.key}")


def impute_nan_columns(array: np.ndarray) -> np.ndarray:
    # Exact operation used by upstream fit_encoding_natstor.py.
    masked_means = ma.array(array, mask=np.isnan(array)).mean(axis=0)
    return np.where(np.isnan(array), masked_means, array)


def bin_words_to_two_seconds(
    embeddings: np.ndarray,
    transcript: pd.DataFrame,
    number_of_bins: int,
) -> np.ndarray:
    if embeddings.shape[0] != len(transcript):
        raise ValueError(
            f"Embedding/transcript mismatch: {embeddings.shape[0]} vectors for {len(transcript)} rows"
        )
    time_grid = np.arange(0, number_of_bins * 2, 2)
    word_ends = transcript["end"].to_numpy(dtype=float)
    word_bins = np.zeros(len(word_ends), dtype=int)
    for index, end_time in enumerate(word_ends):
        candidates = np.where(end_time > time_grid)[0]
        if candidates.size == 0:
            raise ValueError(f"Word ending at {end_time} s precedes the first time bin")
        word_bins[index] = int(candidates[-1])

    binned = []
    for bin_index in range(number_of_bins):
        selected = embeddings[word_bins == bin_index]
        if selected.size == 0:
            binned.append(np.full(embeddings.shape[1], np.nan, dtype=np.float32))
        else:
            binned.append(selected.mean(axis=0))
    return impute_nan_columns(np.asarray(binned))


def result_stem(spec: ModelSpec) -> str:
    return f"cross_story_{SHIFT_LABEL}_{spec.label}_reproduced"


def fit_reproduction(
    data_dir: Path,
    output_dir: Path,
    spec: ModelSpec,
    layers: Sequence[int] | None,
    held_out_stories: Sequence[str],
) -> dict:
    response = load_pickle(data_dir / "response" / f"d_shift_{SHIFT}")
    payloads = {
        story: load_pickle(embedding_path(output_dir, spec, story)) for story in STORIES
    }
    transcripts = {
        story: pd.read_csv(data_dir / "transcribed" / f"{story}.csv") for story in STORIES
    }
    first_payload = payloads[STORIES[0]]
    available_layers = sorted(int(layer) for layer in first_payload["layers"])
    selected_layers = available_layers if layers is None else list(layers)
    unknown = sorted(set(selected_layers) - set(available_layers))
    if unknown:
        raise ValueError(f"Requested unavailable layers: {unknown}; available={available_layers}")

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
        layer_rows = []
        for held_out in held_out_stories:
            training_stories = [story for story in STORIES if story != held_out]
            x_train = np.concatenate([binned[story] for story in training_stories])
            y_train = np.concatenate([response[STORY_NAMES[story]] for story in training_stories])
            x_test = binned[held_out]
            y_test = np.asarray(response[STORY_NAMES[held_out]])

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_train = x_scaler.fit_transform(x_train)
            x_test = x_scaler.transform(x_test)
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

            regressor = RidgeCV(alphas=RIDGE_ALPHAS)
            regressor.fit(x_train, y_train)
            prediction = regressor.predict(x_test)
            correlation = float(pearsonr(y_test, prediction).statistic)
            row = {
                "layer": layer,
                "lang": held_out,
                "story": STORY_NAMES[held_out],
                "r": correlation,
                "alpha": float(regressor.alpha_),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
            }
            rows.append(row)
            layer_rows.append([held_out, correlation])
            print(
                f"[fit] model={spec.key} layer={layer} held_out={held_out} "
                f"r={correlation:.9f} alpha={regressor.alpha_:g}",
                flush=True,
            )
        layerwise[layer] = pd.DataFrame(layer_rows, columns=["lang", "r"])

    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = result_stem(spec)
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
    return {"layerwise": layerwise, "long_frame": long_frame, "means": means}


def xglm_reference_comparison(
    data_dir: Path,
    output_dir: Path,
    reproduced: Mapping[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    reference_path = data_dir / "results" / f"cross_story_{SHIFT_LABEL}_xglm_small"
    reference = load_pickle(reference_path)
    rows = []
    for layer, reproduced_frame in reproduced.items():
        reference_frame = reference[layer].copy()
        reference_frame["lang"] = reference_frame["lang"].astype(str)
        reproduced_frame = reproduced_frame.copy()
        reproduced_frame["lang"] = reproduced_frame["lang"].astype(str)
        merged = reference_frame.merge(
            reproduced_frame, on="lang", suffixes=("_reference", "_reproduced")
        )
        for row in merged.itertuples(index=False):
            difference = float(row.r_reproduced - row.r_reference)
            rows.append(
                {
                    "layer": int(layer),
                    "lang": str(row.lang),
                    "r_reference": float(row.r_reference),
                    "r_reproduced": float(row.r_reproduced),
                    "difference": difference,
                    "absolute_difference": abs(difference),
                }
            )
    detailed = pd.DataFrame(rows)
    reference_summary = detailed.groupby("layer", as_index=False).agg(
        reference_mean=("r_reference", "mean"),
        reference_std=("r_reference", "std"),
        reference_count=("r_reference", "count"),
    )
    reproduced_summary = detailed.groupby("layer", as_index=False).agg(
        reproduced_mean=("r_reproduced", "mean"),
        reproduced_std=("r_reproduced", "std"),
        reproduced_count=("r_reproduced", "count"),
    )
    layer_summary = reference_summary.merge(reproduced_summary, on="layer")
    return detailed, layer_summary, reference_path


def mt5_reference_comparison(
    data_dir: Path,
    means: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    reference_path = data_dir / "results" / "mt5_large.csv"
    raw_reference = pd.read_csv(reference_path)
    consistency = raw_reference.groupby("layer").agg(
        mean_values=("r_mean", "nunique"), std_values=("r_std", "nunique")
    )
    if (consistency.to_numpy() != 1).any():
        raise ValueError("Released mT5 layer summaries are inconsistent across duplicated rows")
    reference = (
        raw_reference[["layer", "r_mean", "r_std"]]
        .drop_duplicates("layer")
        .rename(columns={"r_mean": "reference_mean", "r_std": "reference_std"})
        .sort_values("layer")
    )
    reference["reference_count"] = len(STORIES)
    reproduced = means.rename(
        columns={
            "mean": "reproduced_mean",
            "std": "reproduced_std",
            "count": "reproduced_count",
        }
    )
    layer_summary = reference.merge(
        reproduced[
            ["layer", "reproduced_mean", "reproduced_std", "reproduced_count"]
        ],
        on="layer",
        how="inner",
    )
    return layer_summary.copy(), layer_summary, reference_path


def compare_with_reference(
    data_dir: Path,
    output_dir: Path,
    spec: ModelSpec,
    fitted: dict,
    tolerance: float,
) -> dict:
    if spec.reference == "story_level_pickle":
        detailed, layer_summary, reference_path = xglm_reference_comparison(
            data_dir, output_dir, fitted["layerwise"]
        )
        scalar_reference = detailed["r_reference"]
        scalar_reproduced = detailed["r_reproduced"]
        unit = "layer_by_story"
    elif spec.reference == "layer_summary_csv":
        detailed, layer_summary, reference_path = mt5_reference_comparison(
            data_dir, fitted["means"]
        )
        scalar_reference = layer_summary["reference_mean"]
        scalar_reproduced = layer_summary["reproduced_mean"]
        unit = "layer_summary"
    elif spec.reference == "none":
        reproduced = fitted["means"].rename(
            columns={
                "mean": "reproduced_mean",
                "std": "reproduced_std",
                "count": "reproduced_count",
            }
        )
        layer_summary = reproduced[
            ["layer", "reproduced_mean", "reproduced_std", "reproduced_count"]
        ].copy()
        result_dir = output_dir / "results"
        layer_summary.to_csv(result_dir / "benchmark_layer_summary.csv", index=False)
        summary = {
            "reference_path": None,
            "reference_sha256": None,
            "comparison_unit": "none_new_benchmark",
            "n_compared": 0,
            "max_absolute_difference": None,
            "mean_absolute_difference": None,
            "correlation_across_values": None,
            "maximum_layer_std_absolute_difference": None,
            "tolerance": None,
            "passes_tolerance": None,
        }
        return {"frame": layer_summary, "layer_summary": layer_summary, "summary": summary}
    else:  # pragma: no cover - model registry controls this
        raise ValueError(f"Unsupported reference type {spec.reference}")

    layer_summary = layer_summary.copy()
    layer_summary["mean_difference"] = (
        layer_summary["reproduced_mean"] - layer_summary["reference_mean"]
    )
    layer_summary["mean_absolute_difference"] = layer_summary["mean_difference"].abs()
    layer_summary["std_difference"] = (
        layer_summary["reproduced_std"] - layer_summary["reference_std"]
    )
    layer_summary["std_absolute_difference"] = layer_summary["std_difference"].abs()

    result_dir = output_dir / "results"
    detailed.to_csv(result_dir / "reference_comparison.csv", index=False)
    layer_summary.to_csv(result_dir / "reference_layer_summary.csv", index=False)
    max_scalar_difference = float(np.max(np.abs(scalar_reproduced - scalar_reference)))
    summary = {
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256(reference_path),
        "comparison_unit": unit,
        "n_compared": int(len(scalar_reference)),
        "max_absolute_difference": max_scalar_difference,
        "mean_absolute_difference": float(np.mean(np.abs(scalar_reproduced - scalar_reference))),
        "correlation_across_values": (
            float(pearsonr(scalar_reference, scalar_reproduced).statistic)
            if len(scalar_reference) >= 2
            else None
        ),
        "maximum_layer_std_absolute_difference": float(
            layer_summary["std_absolute_difference"].max()
        ),
        "tolerance": float(tolerance),
    }
    summary["passes_tolerance"] = summary["max_absolute_difference"] <= tolerance
    return {"frame": detailed, "layer_summary": layer_summary, "summary": summary}


def write_report(
    output_dir: Path,
    spec: ModelSpec,
    comparison: dict,
    means: pd.DataFrame,
) -> None:
    summary = comparison["summary"]
    peak_row = means.loc[means["mean"].idxmax()]
    if summary["comparison_unit"] == "none_new_benchmark":
        report = [
            f"# Natural Stories {spec.display_name} shift-{SHIFT} benchmark",
            "",
            "- This is a new benchmark run; the upstream repository does not include a released Natural Stories reference for this model.",
            f"- Reproduced peak layer: {int(peak_row['layer'])}",
            f"- Reproduced peak mean held-out-story r: {peak_row['mean']:.9f}",
            "",
            "Layer summaries are in `results/layer_means.csv`.",
        ]
        (output_dir / "REPRODUCTION_REPORT.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        return
    value_correlation = summary["correlation_across_values"]
    correlation_text = (
        f"{value_correlation:.12f}"
        if value_correlation is not None
        else "not computed (fewer than two values)"
    )
    report = [
        f"# Natural Stories {spec.display_name} shift-{SHIFT} reproduction",
        "",
        f"- Comparison unit: {summary['comparison_unit']}",
        f"- Compared values: {summary['n_compared']}",
        f"- Maximum absolute difference: {summary['max_absolute_difference']:.9g}",
        f"- Mean absolute difference: {summary['mean_absolute_difference']:.9g}",
        f"- Correlation across released and reproduced values: {correlation_text}",
        f"- Maximum layer-SD absolute difference: "
        f"{summary['maximum_layer_std_absolute_difference']:.9g}",
        f"- Requested tolerance: {summary['tolerance']:.9g}",
        f"- Passes tolerance: {summary['passes_tolerance']}",
        f"- Reproduced peak layer: {int(peak_row['layer'])}",
        f"- Reproduced peak mean held-out-story r: {peak_row['mean']:.9f}",
        "",
        "Layer summaries are in `results/reference_layer_summary.csv`.",
        "Detailed available comparisons are in `results/reference_comparison.csv`.",
    ]
    (output_dir / "REPRODUCTION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    spec = MODEL_SPECS[args.model_key]
    model_id = args.model_id or spec.model_id
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model_key={spec.key}", flush=True)
    print(f"model_id={model_id}", flush=True)
    print(f"data_dir={data_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    model_commit_hash = None
    if not args.fit_only:
        device = torch.device(args.device)
        tokenizer, model = load_model(spec, model_id, args.model_revision, device)
        model_commit_hash = getattr(model.config, "_commit_hash", None)
        for story in args.stories:
            extract_story(
                story,
                data_dir,
                output_dir,
                spec,
                tokenizer,
                model,
                device,
                args.overwrite_embeddings,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        first_payload = load_pickle(embedding_path(output_dir, spec, STORIES[0]))
        model_commit_hash = first_payload.get("model_commit_hash")

    run_metadata = {
        "model_key": spec.key,
        "model": model_id,
        "model_revision": args.model_revision,
        "model_commit_hash": model_commit_hash,
        "extraction": spec.extraction,
        "context_words": spec.context_words,
        "shift": SHIFT,
        "stories": list(args.stories),
        "canonical_encoding_stories": list(STORIES),
        "held_out_stories": list(args.held_out_stories),
        "layers": args.layers,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "environment": environment_metadata(),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "elapsed_seconds": None,
    }

    if not args.extract_only:
        fitted = fit_reproduction(
            data_dir,
            output_dir,
            spec,
            args.layers,
            args.held_out_stories,
        )
        comparison = compare_with_reference(
            data_dir,
            output_dir,
            spec,
            fitted,
            args.reference_tolerance,
        )
        run_metadata["comparison"] = comparison["summary"]
        write_report(output_dir, spec, comparison, fitted["means"])
        print(json.dumps(comparison["summary"], indent=2), flush=True)

    run_metadata["elapsed_seconds"] = time.time() - started
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"elapsed_seconds={run_metadata['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
