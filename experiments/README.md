# Experiments

This directory contains run-specific records and derived artifacts.

## Naming convention

Use:

~~~text
YYYY-MM-DD_short-descriptive-name/
~~~

Example:

~~~text
2026-09-08_mini-gpt-baseline/
├── config.yaml
├── notes.md
├── metrics.csv
└── artifacts/
~~~

## Required files

Every completed experiment should include:

- The exact configuration used
- A link to the code commit
- Run metadata
- Metrics
- Qualitative samples when relevant
- Failure analysis
- A next-action decision

Large datasets, checkpoints, and generated outputs should remain local or use a dedicated artifact store. Do not commit them to this repository unless the license, size, and reproducibility value justify it.
