# MiniGPT Smoke Pipeline Validation

- **Date:** 2026-09-08
- **Status:** Completed
- **Run type:** Engineering smoke test
- **Scientific claim:** None

## Research question

Can the repository load an English corpus, construct a character vocabulary, train MiniGPT on CUDA, and report a validation loss from the YAML-driven pipeline?

## Hypothesis

The end-to-end pipeline should complete without an implementation error when using the tracked toy corpus and the smoke configuration.

## Data

- **Training file:** data/examples/tiny_train.txt
- **Validation file:** data/examples/tiny_validation.txt
- **Tokenizer:** CharacterTokenizer
- **Training vocabulary size:** 37
- **Training tokens:** 801
- **Validation tokens:** 192
- **Data status:** Repository-authored toy text for pipeline verification
- **Benchmark status:** Not a benchmark dataset
- **External license:** Not applicable to the repository-authored example text

## Model

- **Model:** MiniGPT
- **Configured vocabulary size:** 96
- **Context length:** 32
- **Embedding dimension:** 32
- **Attention heads:** 4
- **Transformer layers:** 2
- **Feed-forward dimension:** 128
- **Dropout:** 0.0
- **Position encoding:** Learned

## Training configuration

- **Seed:** 42
- **Device:** CUDA
- **Batch size:** 4
- **Maximum steps:** 20
- **Learning rate:** 0.001
- **Weight decay:** 0.01
- **Warmup steps:** 5
- **Gradient clipping norm:** 1.0
- **Evaluation interval:** Every 5 steps
- **Evaluation batches:** 2
- **Checkpoint interval:** Every 10 steps
- **Mixed precision:** Disabled

## User-verified execution

Commands:

~~~bash
python -m pytest
python scripts/train_mini_gpt.py --config configs/smoke.yaml
~~~

Test result:

~~~text
35 passed in 2.95s
~~~

Training output:

~~~text
device: cuda
vocabulary size: 37
training tokens: 801
validation tokens: 192
completed steps: 20
final training loss: 4.043974
validation loss at step 20: 4.133662
~~~

## Interpretation

The implementation passed the intended engineering gate:

- The complete test suite passed.
- The data pipeline loaded both splits.
- MiniGPT trained for 20 optimization steps on CUDA.
- The run produced a final training loss and validation loss.

The validation loss was higher than the final training loss in this run. This is compatible with ordinary train/validation behavior, but this single short run is not sufficient to support a generalization claim.

## Limitations

- The corpus is intentionally tiny and repository-authored.
- The run used only 20 optimization steps.
- No benchmark comparison was performed.
- No repeated-seed estimate was performed.
- Loss history, throughput, peak memory, and checkpoint contents were not independently summarized in this log.
- The generated checkpoint remains a local ignored artifact under outputs/ and is not committed to Git.

## Next experiment

Use a clearly licensed, versioned language-model dataset and define a fixed baseline protocol before changing the model or training procedure. Record dataset identity, preprocessing, token count, throughput, peak GPU memory, full loss history, checkpoint identity, and repeated-seed results.
