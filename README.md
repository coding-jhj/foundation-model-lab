# Foundation Model Lab

A reproducible research and engineering lab for NLP and large language model foundation models.

## Scope

This repository focuses on:

- Decoder-only language model implementation
- Pre-training data pipelines
- Reproducible experiments
- Paper reproduction
- Training and inference efficiency
- Independent research for Research Engineer and LLM research roles

Vision and multimodal research are intentionally out of scope for this lab.

## Portfolio roadmap

1. Build a small language model from first principles.
2. Build a reproducible pre-training pipeline.
3. Reproduce selected scaling, data, and efficiency papers.
4. Select one flagship independent research question.
5. Run pilot, main, and analysis experiments.
6. Publish code, reports, and a paper-style write-up.

## Repository layout

~~~text
foundation-model-lab/
├── data/
│   └── examples/
├── papers/
│   ├── reading_notes/
│   └── reproduction/
├── src/
│   ├── model/
│   ├── data/
│   ├── training/
│   └── evaluation/
├── configs/
├── experiments/
├── reports/
├── scripts/
├── tests/
└── docs/
~~~

## Current milestone

**Milestone 1 — MiniGPT and reproducible pre-training**

The repository now contains:

- Stable mathematical primitives and a character tokenizer
- Causal self-attention and a compact decoder-only Transformer
- Training steps, validation loss estimation, and checkpoint persistence
- Greedy and temperature-based autoregressive generation
- Seed and device utilities
- YAML configuration loading
- An end-to-end training command
- English toy data for a non-scientific smoke test

The toy corpus only verifies that the pipeline runs. It is not evidence for a language-model research claim.

## Local setup

Create and activate a virtual environment:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Install a hardware-appropriate PyTorch build separately when GPU training is required. The exact installation command depends on the operating system, Python version, and CUDA support.

Run the tests:

~~~bash
python -m pytest
~~~

## Run the end-to-end smoke experiment

The smoke configuration uses the tracked English toy corpus and a small model:

~~~bash
python scripts/train_mini_gpt.py --config configs/smoke.yaml
~~~

For a short custom run, override the number of optimization steps:

~~~bash
python scripts/train_mini_gpt.py --config configs/smoke.yaml --max-steps 5
~~~

Checkpoints are written under outputs/mini-gpt-smoke/, which is intentionally ignored by Git.

## Run a real local experiment

Place UTF-8 text files at paths such as:

~~~text
data/raw/train.txt
data/raw/validation.txt
~~~

The tokenizer vocabulary is built from the training text only. Characters that occur only in validation data are encoded as <unk>. Copy configs/baseline.yaml, update the data paths and experiment budget, and run:

~~~bash
python scripts/train_mini_gpt.py --config configs/baseline.yaml
~~~

The current command supports the character tokenizer, learned positional embeddings, single-step updates, and full-precision training. Unsupported configuration options fail explicitly instead of being silently ignored.

## Research protocol

Every experiment must record:

- Research question
- Hypothesis
- Dataset and license
- Model architecture
- Intervention
- Controlled variables
- Random seed
- Hardware and memory
- Training budget
- Metrics and results
- Failure analysis
- Next experiment

The goal is to make every result understandable and reproducible by someone who did not run it.

## Principles

- Start small enough to run experiments repeatedly.
- Change one important variable at a time.
- Keep baselines and interventions comparable.
- Report negative and inconclusive results.
- Do not treat an AI-generated suggestion as experimental evidence.
- Verify every research claim with executed experiments.

## Primary references

- Vaswani et al., *Attention Is All You Need*
- Stanford CS336, *Language Modeling from Scratch*
- Kaplan et al., *Scaling Laws for Neural Language Models*
- Hoffmann et al., *Training Compute-Optimal Large Language Models*
