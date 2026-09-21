# AuriStream-MTP 7B 40Pred neural alignment

`figure5_exploration_neural_alignment_7b40pred.{pdf,png}` compares the
AuriStream-MTP checkpoint with its corrected untrained control across all 97
representation levels.

- Baseline 200 reports mean noise-ceiling-normalized predictivity (`r_cv_norm`)
  in language fROIs after applying `NCSNR > 0.4`. Vertical lines are SEM over
  eight participants.
- Natural Stories reports mean held-out-story Pearson correlation for the
  100-second scoring-trim, three-shift analysis. Vertical lines are SEM over
  nine stories.

The exact plotted values are in
`figure5_exploration_neural_alignment_7b40pred_plot_data.csv`. Regenerate with
`../../plot_auristream_neural_alignment_layers.py`; the script reads the
Baseline 200 fits from the AuriStream-alpha repository and the completed
Natural Stories run summaries from this repository.

The two text-model comparison figures add GPT-2 XL, GPT-J 6B, and Qwen3 8B:

- `figure6_neural_alignment_baseline200-last-token_naturalstories-standard`
  uses last-token text representations for Baseline 200.
- `figure6_neural_alignment_baseline200-mean-token_naturalstories-standard`
  uses mean-token text representations for Baseline 200.

Only the Baseline 200 text pooling differs. In both figures, Natural Stories
uses the same 20.48-second text context matched to AuriStream's maximum input
duration. AuriStream itself uses its mean-token representation in Baseline
200. Exact plotted values and descriptive peaks are stored beside each PDF and
PNG. Regenerate either version with
`../../plot_auristream_text_model_neural_alignment.py` and the corresponding
`--baseline-text-pooling` argument.
