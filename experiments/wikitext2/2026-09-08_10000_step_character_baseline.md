# WikiText-2 Character Extended Baseline

- **Date:** 2026-09-08
- **Status:** Completed
- **Run type:** Extended single-run baseline
- **Scientific claim:** Baseline validation only; not a publication-quality comparison

## Objective

Measure whether a longer training budget improves the small character-level MiniGPT baseline while preserving the original 1,000-step artifacts.

## Dataset

- **Dataset:** Salesforce/wikitext
- **Configuration:** wikitext-2-raw-v1
- **Preparation revision requested:** main
- **Training tokens:** 10,869,219
- **Validation tokens:** 1,139,685
- **Test tokens:** 1,282,730
- **Character vocabulary size:** 1,017
- **Tokenizer:** Character-level tokenizer fit on the training split only
- **Context length:** 128

The prepared dataset files and metadata are generated locally and are intentionally not committed to Git.

## Model and training setup

The model and optimization settings match the 1,000-step baseline.

- **Model:** MiniGPT decoder-only Transformer
- **Embedding dimension:** 128
- **Attention heads:** 4
- **Transformer layers:** 4
- **Feed-forward dimension:** 512
- **Configured vocabulary capacity:** 2,048
- **Position encoding:** Learned
- **Dropout:** 0.0
- **Batch size:** 8
- **Learning rate:** 0.0003
- **Weight decay:** 0.1
- **Warmup steps:** 100
- **Gradient clipping norm:** 1.0
- **Evaluation interval:** 100 steps
- **Evaluation batches:** 20
- **Mixed precision:** Disabled
- **Gradient accumulation:** 1
- **Maximum steps:** 10,000
- **Device:** CUDA
- **Random seed:** 42

## User-verified output

~~~text
device: cuda
vocabulary size: 1017
training tokens: 10869219
validation tokens: 1139685
test tokens: 1282730
completed steps: 10000
final training loss: 1.528465
validation loss at step 10000: 1.592640
test loss: 1.681452
elapsed seconds: 96.65
steps per second: 103.47
peak GPU memory (MiB): 129.80
metrics file: runs/wikitext2-character-10k/metrics.jsonl
~~~

## Quantitative comparison

| Metric | 1,000 steps | 10,000 steps | Absolute change |
|---|---:|---:|---:|
| Training loss | 2.491694 | 1.528465 | -0.963229 |
| Validation loss | 2.482680 | 1.592640 | -0.890040 |
| Test loss | 2.469743 | 1.681452 | -0.788291 |
| Elapsed seconds | 10.54 | 96.65 | +86.11 |
| Steps per second | 94.91 | 103.47 | +8.56 |

The two runs use the same configuration and seed, but their validation and test losses are estimated from sampled windows after different numbers of updates. The comparison is therefore directional rather than a fully paired statistical estimate.

## Qualitative generation check

The final checkpoint was loaded with the sampling CLI and evaluated with the same fixed prompt and greedy decoding used for the 1,000-step baseline:

- **Checkpoint:** 'checkpoints/wikitext2-character-10k/step-010000.pt'
- **Prompt:** 'The '
- **Decoding:** Greedy decoding (temperature=0)
- **Requested continuation:** 100 tokens

User-verified output:

~~~text
The first of the season of the season of the contration of the control of the control of the constructio
~~~

Compared with the 1,000-step output, the model now produces a partially structured continuation with recognizable word sequences. It still repeats phrases and truncates a word, so this is evidence of qualitative improvement rather than evidence of fluent language generation.

## Interpretation

The extended run improved all three reported losses relative to the 1,000-step run. The validation and test losses remain higher than the training loss, with the test loss higher than the validation loss in this run. Because evaluation uses sampled windows and there is only one seed, this gap is not sufficient to diagnose overfitting or a dataset effect.

The qualitative output is consistent with the quantitative improvement: the model has learned more local word and phrase structure, but the current character-level model and training budget remain limited.

## Artifacts

- **Metrics:** 'runs/wikitext2-character-10k/metrics.jsonl'
- **Checkpoint directory:** 'checkpoints/wikitext2-character-10k/'
- **Configuration:** 'configs/wikitext2_char_10k.yaml'
- **Sampling script:** 'scripts/sample_mini_gpt.py'
- **Git tracking:** Runtime artifacts remain local and are ignored by Git.

## Limitations

- This is one seed and one model configuration.
- The model is character-level, so its loss is not directly comparable with subword-token language-model results.
- The qualitative check used one prompt and greedy decoding.
- Validation and test estimates use sampled windows rather than a full deterministic sweep.
- No parameter-count or FLOP accounting has been added to this record.
- No paper baseline or ablation has been run.
- The metrics JSONL and checkpoint files are local artifacts, not committed to Git.

## Next steps

1. Preserve this baseline as the reference implementation.
2. Add explicit parameter-count and evaluation-summary reporting.
3. Select and document a small, reproducible paper experiment.
4. Implement the first paper reproduction as a controlled sweep with fixed seeds and budgets.
