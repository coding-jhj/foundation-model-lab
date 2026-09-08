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

## Quantitative interpretation

The run completed successfully and produced the expected operational and evaluation outputs.

The final validation loss was 2.482680 and the held-out test loss was 2.469743. The training and held-out losses are close in this single run, so there is no immediate evidence of severe overfitting at step 1,000. This observation is descriptive only and should not be generalized without repeated seeds, longer training, and controlled comparisons.

The baseline also confirms that the current implementation can train the real-data corpus on the available GPU with low measured peak memory. The reported memory value describes the instrumented training process and is not a claim about the maximum capacity of the GPU.

## Qualitative generation check

The final checkpoint was loaded successfully and evaluated with a fixed prompt:

- **Checkpoint:** 'checkpoints/wikitext2-character-baseline/step-001000.pt'
- **Prompt:** 'The '
- **Decoding:** Greedy decoding (temperature=0)
- **Requested continuation:** 100 tokens

User-verified output:

~~~text
The the the the the the the the the the the the the the the the the the t the the the the the the the th
~~~

The output is highly repetitive and is not yet qualitatively useful as natural text. The most likely explanation is that 1,000 updates are insufficient for this small character-level model to learn stable long-range structure; this is a working hypothesis, not a proven diagnosis. The successful checkpoint load and vocabulary restriction show that the inference path itself executed correctly.

## Artifacts

- **Metrics:** 'runs/wikitext2-character-baseline/metrics.jsonl'
- **Checkpoint directory:** 'checkpoints/wikitext2-character-baseline/'
- **Configuration:** 'configs/wikitext2_char.yaml'
- **Dataset preparation script:** 'scripts/prepare_wikitext2.py'
- **Sampling script:** 'scripts/sample_mini_gpt.py'
- **Git tracking:** Runtime artifacts remain local and are ignored by Git.

## Limitations

- This is one seed and one training budget.
- The model is character-level, so its loss is not directly comparable with subword-token language-model results.
- The qualitative check used one prompt and greedy decoding.
- No parameter-count or FLOP accounting has been added to this record.
- No paper baseline or ablation has been run.
- The metrics JSONL and checkpoint files are local artifacts, not committed to Git.

## Next steps

1. Run a fresh 10,000-step extended baseline with the same configuration and seed.
2. Repeat the fixed-prompt generation check at step 10,000.
3. Compare the 1,000-step and 10,000-step loss curves and qualitative outputs.
4. Add parameter-count and evaluation summaries to the experiment report.
5. Select a first paper reproduction with a clear, low-compute ablation that fits the current codebase.
