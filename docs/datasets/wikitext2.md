# WikiText-2 Raw Dataset

## Dataset identity

- **Repository:** Salesforce/wikitext
- **Configuration:** wikitext-2-raw-v1
- **Splits used:** train, validation, test
- **Source:** https://huggingface.co/datasets/Salesforce/wikitext
- **Preparation script:** scripts/prepare_wikitext2.py

The preparation script downloads the dataset with the Hugging Face Datasets library and serializes each split as UTF-8 text. The original dataset is not committed to this repository. Generated files are placed under data/raw/, which is ignored by Git.

## License metadata

The current dataset card lists:

- cc-by-sa-3.0
- gfdl

Review the dataset card and the applicable license terms before redistributing the raw dataset, serialized splits, model checkpoints, or derived artifacts.

## Intended use in this lab

This dataset is the first real-data baseline after the repository-authored toy corpus. The initial model uses a character tokenizer so that the implementation remains transparent and consistent with the from-scratch learning path.

This baseline is not yet a paper reproduction. It is an engineering and measurement baseline for:

- Data preparation
- Character-level language modeling
- Training and validation loss
- Checkpoint persistence
- GPU memory and throughput measurement

## Reproducibility requirements

Record the following for every run:

- Dataset revision passed to prepare_wikitext2.py
- Number of serialized characters and token IDs
- Character vocabulary
- Configuration file
- Random seed
- GPU name and peak memory
- Training steps and elapsed time
- Training and validation loss history
- Checkpoint path and hash
- Failure analysis

## Related publication

The dataset card cites Merity et al., *Pointer Sentinel Mixture Models*:

https://arxiv.org/abs/1609.07843
