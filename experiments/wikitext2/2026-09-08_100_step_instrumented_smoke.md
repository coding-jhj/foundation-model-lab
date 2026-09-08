# WikiText-2 Character Baseline Instrumented Smoke

- **Date:** 2026-09-08
- **Status:** Completed
- **Run type:** Instrumented baseline smoke test
- **Scientific claim:** None

## Objective

Verify that the instrumented WikiText-2 character baseline records training metrics, elapsed time, throughput, and peak GPU memory during a short run.

## Dataset

- **Dataset:** Salesforce/wikitext
- **Configuration:** wikitext-2-raw-v1
- **Revision requested during preparation:** main
- **Training characters:** 10,869,219
- **Validation characters:** 1,139,685
- **Character vocabulary size:** 1,017

## Model and training setup

- **Model:** MiniGPT
- **Context length:** 128
- **Embedding dimension:** 128
- **Attention heads:** 4
- **Transformer layers:** 4
- **Feed-forward dimension:** 512
- **Configured vocabulary capacity:** 2,048
- **Batch size:** 8
- **Maximum steps:** 100
- **Learning rate:** 0.0003
- **Weight decay:** 0.1
- **Warmup steps:** 100
- **Gradient clipping norm:** 1.0
- **Device:** CUDA

## User-verified output

~~~text
device: cuda
vocabulary size: 1017
training tokens: 10869219
validation tokens: 1139685
completed steps: 100
final training loss: 3.428631
validation loss at step 100: 3.503061
elapsed seconds: 1.67
steps per second: 59.75
peak GPU memory (MiB): 129.80
metrics file: runs/wikitext2-character-baseline/metrics.jsonl
~~~

## Interpretation

The instrumented run completed successfully and produced the expected operational measurements:

- CUDA execution was used.
- Training and validation losses were computed.
- The metrics JSONL path was created by the run.
- Throughput and peak GPU memory were reported.

The loss values show a clear optimization signal compared with the earlier 10-step data smoke, but these short runs are not treated as a final model-quality result. The earlier and current runs were separate executions, and neither provides a controlled comparison across multiple seeds or training budgets.

## Artifacts

- **Metrics:** runs/wikitext2-character-baseline/metrics.jsonl
- **Checkpoint directory:** checkpoints/wikitext2-character-baseline/
- **Git tracking:** Run artifacts are intentionally ignored and remain local.

## Limitations

- Only 100 optimization steps were completed.
- No test-split evaluation was performed.
- No repeated seeds were performed.
- The character-level objective is not directly comparable with subword-token baselines.
- The full metrics history was generated locally but is not committed to Git.

## Next run

Run the formal 1,000-step baseline with the same configuration. Preserve the local metrics JSONL and checkpoint artifacts, then summarize the final and best validation losses, elapsed time, throughput, peak memory, and test-split loss.
