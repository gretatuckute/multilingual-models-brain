# Natural Stories reference reproduction

This directory reproduces released Natural Stories encoding results before any
AuriStream analysis is added, and supports explicitly labeled new comparison
models. Andrea de Varda's original Natural Stories scripts and result files
remain unchanged; links below point directly to the original repository's
[`revision` branch](https://github.com/Andrea-de-Varda/multilingual-models-brain/tree/revision).

## Validation status

| Model | Status | Validation target |
| --- | --- | --- |
| XGLM-small | Exact reproduction | 225 released layer-by-story correlations |
| mT5-large | Exact reproduction | 25 released layer means and standard deviations |
| GPT-2 XL | New exploratory benchmark | Internal extraction checks only; no released Natural Stories result exists |
| AuriStream 100M–40Pred | New audio benchmark | Alpha activation cache; no released Natural Stories result exists |
| AuriStream 100M–1Pred | New audio benchmark | Alpha activation cache; no released Natural Stories result exists |
| AuriStream 100M–40Pred random init | Untrained audio control | Alpha activation cache; no released Natural Stories result exists |
| AuriStream 7B–40Pred | New audio benchmark | Alpha activation cache; no released Natural Stories result exists |
| AuriStream 7B–1Pred | Prediction-horizon control | Alpha activation cache; no released Natural Stories result exists |
| AuriStream 7B–40Pred random init | Untrained audio control | Alpha activation cache; no released Natural Stories result exists |
| AuriStream 7B–40Pred, 5 s context | Temporal-context control | Alpha activation cache; no released Natural Stories result exists |
| WavCoch bottleneck | Tokenizer-only control | Quantized WavCoch bottleneck states; AuriStream is not run |

## Organization

The implementation is split between two repositories so that model inference
stays with AuriStream and the Natural Stories benchmark stays with the
reproduction of the original analysis:

| Repository | Responsibility | Key code |
| --- | --- | --- |
| [AuriStream-alpha](https://github.com/gretatuckute/AuriStream-alpha) | Load AuriStream/WavCoch, process story audio, extract hidden states, and mean-pool them onto the released two-second response grid | `evals_neural/extract_naturalstories_activations.py`, `evals_neural/run_naturalstories_activations.sh` |
| This Natural Stories fork | Align cached representations with the released responses, run leave-one-story-out encoding models, extract matched text-model representations, and store benchmark summaries | `reproduce.py`, `fit_auristream.py`, `fit_auristream_ramp_trim.py`, `fit_text_ramp_trim.py` |
| AuriStream-alpha | Make the final combined audio/text comparison figure | `scripts/plot_naturalstories_model_context_summary.py` |

More specifically:

- `reproduce.py` extracts XGLM, mT5, GPT-2 XL, GPT-J, and Qwen
  representations; averages subtokens within words and words within the
  released two-second bins; fits the
  [original leave-one-story-out RidgeCV model](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/fit_encoding_natstor.py);
  and, where an upstream result exists, compares against it.
- `fit_auristream.py` reads a standard activation cache produced by
  AuriStream-alpha and passes each already-binned layer to the same validated
  fitting function. It intentionally contains no independent AuriStream or
  WavCoch loader.
- `fit_auristream_ramp_trim.py` and `fit_text_ramp_trim.py` implement the final
  onset-ramp correction. The model is trained and predictions are produced on
  the complete folds; only the first 50 two-second bins (100 seconds) are
  omitted when correlating predictions with the held-out story.
- `run_causal_text_context_extraction.sh` and
  `run_causal_text_context_ramp_fits.sh` run the 100-word and matched
  20.48-second causal text conditions. `run_naturalstories_roi_ramp_fits.sh`
  runs the same corrected analysis for the five released language fROIs.
- The maintained null controls live in AuriStream-alpha: circularly shifted
  neural responses preserve response autocorrelation while breaking alignment,
  and the wrong-story WavCoch-token control replaces the complete input token
  stream using a fixed duration-matched donor-story derangement. Their fit and
  extraction implementations are under `evals_neural/exploratory/`, with
  provenance-checking launchers under `evals_neural/reruns/`.
- `plot_reproduction.py`, `plot_text_ramp_trim.py`,
  `plot_naturalstories_roi_ramp.py`, and
  `plot_naturalstories_best_layer_bar.py` make benchmark-specific summaries.
- `runs/` contains generated embeddings, results, figures, logs, manifests,
  and isolated runtime dependencies. The entire directory is ignored by Git;
  no generated activations, model outputs, figures, or result tables belong in
  commits.

Each completed model has the following local structure:

```text
runs/<model>_3shift/
├── embeddings/
├── results/
├── figures/
├── logs/
├── REPRODUCTION_REPORT.md
└── run_manifest.json
```

## Preserved analysis steps

Every run uses the operations from the
[original Natural Stories analysis](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/fit_encoding_natstor.py):

- word representations are averaged over subword tokens;
- words are averaged into the released two-second response bins using their
  end timestamps;
- the shift-3 language-network response is used;
- features and responses are standardized using each training fold only;
- RidgeCV uses the original alpha grid;
- one story is held out at a time; and
- performance is held-out-story Pearson correlation.

The model-specific extraction is recorded separately because the released mT5
result and the visible extraction call are not consistent:

- `xglm_small`: `facebook/xglm-564M`; the entire story is presented causally in
  one sequence and the prepended special token is removed.
- `mt5_large`: `google/mt5-large`; the exact released curve is obtained when
  the whole story is passed through the bidirectional encoder, the final EOS
  token is removed, and subwords are averaged within words.
- `gpt2_xl`: `openai-community/gpt2-xl`; this is a new decoder-only causal
  comparison rather than a released-result reproduction. Each target word is
  extracted from a trailing context of at most 100 words.
- `gpt_j_6b`: `EleutherAI/gpt-j-6b`; a new decoder-only causal comparison.
- `qwen3_8b`: `Qwen/Qwen3-8B`; a new decoder-only causal comparison.

The three causal text models can use either the paper-style 100-word context or
a trailing 20.48-second context matched to AuriStream. In both conditions,
subtokens overlapping the target word are averaged, and the resulting word
vectors are averaged within the same released two-second response bins.

The
[upstream mT5-large call](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/get_natstor_embeddings.py#L260-L264)
uses `embeddings_chunked`, whose default is a trailing 100-word window, and the
paper describes the same restriction. Re-running that version did not
reproduce the released contextual-layer curve. In contrast, full-story
bidirectional mT5 embeddings reproduce all released layer means to within
`1.39e-07`. This strongly indicates that the released
[`mt5_large.csv`](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/results/mt5_large.csv)
was produced from a pre-existing full-story embedding cache. The 100-word run
is retained locally as `runs/mt5_large_100word_diagnostic/`; it is not treated
as the reference reproduction.

## Running

Generated output defaults to the corresponding ignored run directory:

```bash
python reproduce.py --model-key xglm_small
python reproduce.py --model-key mt5_large
python reproduce.py --model-key gpt2_xl
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream100M_40Pred_BigAudioDataset_500k
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream100M_1Pred_BigAudioDataset_500k
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream100M_40Pred_BigAudioDataset_500k-randinit
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream7BDeep_40Pred_BigAudioDataset_500k
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream7BDeep_1Pred_BigAudioDataset_500k
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream7BDeep_40Pred_BigAudioDataset_500k-randinit
python fit_auristream.py \
  --source_model TuKoResearch/WavCochCausalV8192-vocoder
python fit_auristream.py \
  --source_model TuKoResearch/AuriStream7BDeep_40Pred_BigAudioDataset_500k \
  --stimset_name naturalstories_2s_responsegrid_context5s \
  --output_dir runs/AuriStream7BDeep_40Pred_BigAudioDataset_500k_context5s_3shift
python plot_reproduction.py
python plot_reproduction.py \
  --models AuriStream100M_40Pred_BigAudioDataset_500k-randinit \
           AuriStream100M_1Pred_BigAudioDataset_500k \
           AuriStream100M_40Pred_BigAudioDataset_500k \
           AuriStream7BDeep_40Pred_BigAudioDataset_500k
```

XGLM and mT5 tokenization require `sentencepiece` (tested with version
`0.2.2`). The cluster runs used an isolated copy under the ignored
`runs/_runtime/python_deps/` directory; that generated dependency directory is
not part of the repository.

Extraction and fitting can be separated with `--extract-only` and `--fit-only`.
A restricted extraction smoke can be run with, for example,
`--stories 1 --extract-only`; the resulting story cache is reused by the full
run.

## AuriStream activation alignment

The audio forward pass is intentionally implemented in AuriStream-alpha, where
it imports the existing shared model/tokenizer loader, processor shim, pooling,
and activation-flattening functions. It uses the standard causal
`TuKoResearch/WavCochCausalV8192-vocoder` tokenizer for all AuriStream models.
Each story is tokenized at 200 Hz, and each two-second response bin is represented
by the mean of its corresponding 400 hidden states. The forward pass contains
only a trailing causal context ending at that bin, capped at the 4096-token
(20.48-second) AuriStream training context; it never includes future tokens.

The released response grid is slightly longer than each WAV file. Alpha
therefore appends digital silence at the waveform tail, without shifting or
cropping the recording, so all released response bins have acoustic features.
The exact added samples and seconds are stored per story in Alpha's matching
`_stim.pkl` cache and JSON manifest. This tail alignment is separate from
WavCoch's own internal causal left padding. The benchmark uses the released
`d_shift_3` response directly and does not apply an additional manual shift to
AuriStream activations.

The WavCoch-only control uses Alpha's standard direct-WavCoch model path and
mean-pools the 400 quantized FSQ bottleneck states in each response bin; it
does not instantiate or run AuriStream. The 5-second-context control keeps the
same 2-second target bin but limits AuriStream to at most 3 seconds of earlier
tokens, rather than the standard 18.48 seconds of earlier tokens.

## XGLM-small result

The full XGLM-small reproduction compared all 25 representation levels and all
nine held-out stories: 225 released-versus-reproduced values.

- Maximum absolute difference: `5.01541915e-07`
- Mean absolute difference: `1.53137231e-07`
- Correlation across released and reproduced values: `0.9999999999987167`
- Prespecified numerical tolerance: `1e-05` (passed)
- Released and reproduced peak: layer 15
- Reproduced peak mean held-out-story Pearson r: `0.4421498787`

The checkpoint resolved to Hugging Face commit
`f3059f01b98ccc877c673149e0178c0e957660f9`. The released comparison pickle is
the original repository's
[`cross_story_3shift_xglm_small`](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/results/cross_story_3shift_xglm_small)
pickle (SHA-256
`a24b88c1e8a694c28ee8b85974230054e1725912e3c94f185645181ab9ab379b`).

## mT5-large reference

The original repository's released
[`mt5_large.csv`](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/results/mt5_large.csv)
contains the mean and standard deviation across held-out stories for each of
25 representation levels, but not the nine individual story scores. The mT5
reproduction therefore compares the 25 layer means and layer standard
deviations with that file; the new run also retains its own individual
held-out-story scores.

The full-story reproduction passed the numerical tolerance:

- Maximum absolute layer-mean difference: `1.38500615e-07`
- Mean absolute layer-mean difference: `3.86718338e-08`
- Correlation across released and reproduced layer means:
  `0.9999999999997836`
- Maximum absolute layer-SD difference: `8.20345467e-08`

## GPT-2 XL causal comparison

The GPT-2 XL run is a new benchmark using the reproduced Natural Stories
pipeline; there is no upstream GPT-2 XL result for an exact numerical
comparison. All nine stories produced 49 finite representation levels. The
mean held-out-story curve peaks at layer 22 of 48 (normalized layer position
`0.458`) with Pearson `r = 0.409605`. Performance declines in later layers and
is `r = 0.279175` at layer 48.

The checkpoint resolved to Hugging Face commit
`15ea56dee5df4983c59b2538573817e1667135e2`. The extracted target word has at
most 99 preceding words in context and no subsequent words; causal attention
also prevents access to later tokens inside the input window.

As an independent extraction check, six target words spanning the beginning,
100-word context boundary, middle, and end of story 1 were rerun one at a time
without batching or padding. Across all 49 levels (294 vector comparisons),
the maximum batched-versus-unbatched absolute element difference was
`4.8828125e-04` and the mean absolute element difference was `3.78605e-06`.
These small differences are expected from GPU batch-dependent floating-point
operations; the check found no word-selection or causal-padding mismatch.
