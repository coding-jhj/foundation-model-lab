# WikiText-2 Character Baseline Data Smoke

- **Date:** 2026-09-08
- **Status:** Completed
- **Run type:** Data and hardware smoke test
- **Scientific claim:** None

## Objective

Verify that the prepared WikiText-2 raw splits can be loaded by the character tokenizer and trained by MiniGPT on the configured CUDA device.

## Dataset

- **Dataset:** Salesforce/wikitext
- **Configuration:** wikitext-2-raw-v1
- **Revision requested:** main
- **Preparation command:** scripts/prepare_wikitext2.py
- **License metadata listed on the dataset card:** cc-by-sa-3.0 and gfdl
- **Training examples:** 36,718
- **Validation examples:** 3,760
- **Test examples:** 4,358
- **Training characters:** 10,869,219
- **Validation characters:** 1,139,685
- **Test characters:** 1,282,730
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
- **Maximum steps:** 10
- **Learning rate:** 0.0003
- **Weight decay:** 0.1
- **Warmup steps:** 100
- **Gradient clipping norm:** 1.0
- **Device:** CUDA

## User-verified commands

~~~bash
python scripts/prepare_wikitext2.py --output-dir data/raw/wikitext-2-raw-v1 --revision main
python scripts/train_mini_gpt.py --config configs/wikitext2_char.yaml --max-steps 10
~~~

## User-verified training output

~~~text
device: cuda
vocabulary size: 1017
training tokens: 10869219
validation tokens: 1139685
completed steps: 10
final training loss: 7.753326
validation loss at step 10: 7.735848
~~~

## Interpretation

The data smoke test passed:

- All three WikiText-2 splits were downloaded and serialized.
- The training-only character vocabulary was constructed successfully.
- The configured vocabulary capacity of 2,048 was sufficient for the observed 1,017-character vocabulary.
- MiniGPT completed 10 CUDA optimization steps.

The validation loss being slightly lower than the training loss at step 10 is not interpreted as a generalization result. The two values are estimated from different random batches in a very short run.

## Limitations

- This was a 10-step smoke test, not the formal baseline.
- No throughput or peak-memory metrics were recorded by the earlier CLI version.
- No test-set evaluation was performed.
- No repeated seeds were run.
- Character-level loss is not directly comparable to subword-token language-model results.

## Next run

Pull the metrics-instrumentation update and run the formal 1,000-step baseline. Preserve the generated JSONL metrics file and checkpoint metadata with the experiment record.
