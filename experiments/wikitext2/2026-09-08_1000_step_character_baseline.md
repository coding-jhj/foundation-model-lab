# WikiText-2 Character Baseline

- **Date:** 2026-09-08
- **Status:** Completed
- **Run type:** Formal single-run baseline
- **Scientific claim:** Baseline validation only; not a publication-quality comparison

## Objective

Establish a reproducible character-level MiniGPT baseline on the WikiText-2 raw dataset and record training, validation, test, runtime, throughput, and GPU-memory measurements.

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

## Model

- **Model:** MiniGPT decoder-only Transformer
- **Embedding dimension:** 128
- **Attention heads:** 4
- **Transformer layers:** 4
- **Feed-forward dimension:** 512
- **Configured vocabulary capacity:** 2,048
- **Position encoding:** Learned
- **Dropout:** 0.0

## Training setup

- **Maximum steps:** 1,000
- **Batch size:** 8
- **Learning rate:** 0.0003
- **Weight decay:** 0.1
- **Warmup steps:** 100
- **Gradient clipping norm:** 1.0
- **Evaluation interval:** 100 steps
- **Evaluation batches:** 20
- **Mixed precision:** Disabled
- **Gradient accumulation:** 1
- **Device:** CUDA
- **Random seed:** Defined by the configuration and training entry point

## User-verified output

~~~text
device: cuda
vocabulary size: 1017
training tokens: 10869219
validation tokens: 1139685
test tokens: 1282730
completed steps: 1000
final training loss: 2.491694
validation loss at step 1000: 2.482680
test loss: 2.469743
elapsed seconds: 10.54
steps per second: 94.91
peak GPU memory (MiB): 129.80
metrics file: runs/wikitext2-character-baseline/metrics.jsonl
~~~

## Interpretation

The run completed successfully and produced the expected operational and evaluation outputs.

The final validation loss was 2.482680 and the held-out test loss was 2.469743. The training and held-out losses are close in this single run, so there is no immediate evidence of severe overfitting at step 1,000. This observation is descriptive only and should not be generalized without repeated seeds, longer training, and controlled comparisons.

The baseline also confirms that the current implementation can train the real-data corpus on the available GPU with low measured peak memory. The reported memory value describes the instrumented training process and is not a claim about the maximum capacity of the GPU.

## Artifacts

- **Metrics:** `runs/wikitext2-character-baseline/metrics.jsonl`
- **Checkpoint directory:** `checkpoints/wikitext2-character-baseline/`
- **Configuration:** `configs/wikitext2_char.yaml`
- **Dataset preparation script:** `scripts/prepare_wikitext2.py`
- **Git tracking:** Runtime artifacts remain local and are ignored by Git.

## Limitations

- This is one seed and one training budget.
- The model is character-level, so its loss is not directly comparable with subword-token language-model results.
- No generated samples have been evaluated yet.
- No parameter-count or FLOP accounting has been added to this record.
- No paper baseline or ablation has been run.
- The metrics JSONL and checkpoint files are local artifacts, not committed to Git.

## Next steps

1. Inspect the metrics history and verify checkpoint loading.
2. Generate fixed prompts with deterministic decoding and record qualitative samples.
3. Add parameter-count and evaluation summaries to the experiment report.
4. Select a first paper reproduction with a clear, low-compute ablation that fits the current codebase.
