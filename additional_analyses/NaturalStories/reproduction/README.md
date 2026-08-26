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

## Organization

- `reproduce.py` extracts model representations, bins them onto the released
  two-second fMRI grid, fits the
  [original leave-one-story-out RidgeCV model](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/additional_analyses/NaturalStories/fit_encoding_natstor.py),
  and compares the output with the released reference.
- `plot_reproduction.py` makes model-level diagnostic plots and a normalized-
  layer plot modeled on the presentation in
  [the original repository's layer-wise figure](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/revision/plots/layerwise.svg).
- `runs/` contains generated embeddings, results, figures, logs, manifests, and
  isolated runtime dependencies. The entire directory is ignored by Git.

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
python plot_reproduction.py
python plot_reproduction.py --models xglm_small mt5_large gpt2_xl
```

XGLM and mT5 tokenization require `sentencepiece` (tested with version
`0.2.2`). The cluster runs used an isolated copy under the ignored
`runs/_runtime/python_deps/` directory; that generated dependency directory is
not part of the repository.

Extraction and fitting can be separated with `--extract-only` and `--fit-only`.
A restricted extraction smoke can be run with, for example,
`--stories 1 --extract-only`; the resulting story cache is reused by the full
run.

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
