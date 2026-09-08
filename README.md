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

**Milestone 2 — First real language-model dataset**

The repository now supports:

- Stable mathematical primitives and a character tokenizer
- Causal self-attention and a compact decoder-only Transformer
- Training steps, validation loss estimation, and checkpoint persistence
- Greedy and temperature-based autoregressive generation
- Seed and device utilities
- YAML configuration loading
- End-to-end training commands
- Reproducible preparation of WikiText-2 raw splits
- Checkpoint sampling with an active-vocabulary limit

The tracked toy corpus only verifies that the pipeline runs. It is not evidence for a language-model research claim.

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

## Prepare the first real dataset

The first dataset baseline uses Salesforce WikiText, configuration wikitext-2-raw-v1. The preparation script downloads the train, validation, and test splits through the Hugging Face Datasets library, writes UTF-8 text files under data/raw/, and records metadata.json.

Install the optional data dependency:

~~~bash
python -m pip install -e ".[data]"
~~~

Prepare the dataset:

~~~bash
python scripts/prepare_wikitext2.py --output-dir data/raw/wikitext-2-raw-v1 --revision main
~~~

The dataset card currently lists cc-by-sa-3.0 and gfdl. Review the dataset card and its terms before redistributing derived artifacts.

## Run the WikiText-2 character baseline

~~~bash
python scripts/train_mini_gpt.py --config configs/wikitext2_char.yaml
~~~

For the first hardware smoke run on WikiText-2:

~~~bash
python scripts/train_mini_gpt.py --config configs/wikitext2_char.yaml --max-steps 10
~~~

The dataset source, configuration, license metadata, and intended experimental use are documented in docs/datasets/wikitext2.md.

For an isolated extended baseline that preserves the 1,000-step artifacts:

~~~bash
python scripts/train_mini_gpt.py --config configs/wikitext2_char_10k.yaml
~~~

## Sample from a trained checkpoint

Use the final checkpoint from the formal baseline and greedy decoding for a deterministic qualitative check:

~~~bash
python scripts/sample_mini_gpt.py --config configs/wikitext2_char.yaml --checkpoint checkpoints/wikitext2-character-baseline/step-001000.pt --prompt "The " --temperature 0 --max-new-tokens 100
~~~

The sampling command fits the tokenizer on the training split, restores the checkpoint, and prevents unused model vocabulary slots from appearing in decoded text. Change the prompt or use a nonzero temperature for additional qualitative samples.

## Run a custom local experiment

The tokenizer vocabulary is built from the training text only. Characters that occur only in validation data are encoded as <unk>. Copy a configuration, update the data paths and experiment budget, and run:

~~~bash
python scripts/train_mini_gpt.py --config configs/baseline.yaml
~~~

The current command supports the character tokenizer, learned positional embeddings, single-step updates, linear warmup, and full-precision training. Unsupported configuration options fail explicitly instead of being silently ignored.

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
- Merity et al., *Pointer Sentinel Mixture Models*
- Kaplan et al., *Scaling Laws for Neural Language Models*
- Hoffmann et al., *Training Compute-Optimal Large Language Models*
