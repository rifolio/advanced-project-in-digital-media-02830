# Jev with V5 instructions

This directory contains the Gemini GEPA V5 instructions adapted for Jev:

- [`program.json`](./program.json) — Jev-compatible prompt artifact.

The artifact uses the V5 prompt with a Jev probability threshold of `0.6`.
The threshold is passed separately when evaluating because it controls the
probability-to-label mapping:

```text
yes if probability >= 0.6
no otherwise
```

## Benchmark

Evaluated on 300 MRBench test examples with Jev:

| Metric | Value |
|---|---:|
| Accuracy | 0.963 |
| Precision | 0.852 |
| Recall | 0.939 |
| F1 | 0.893 |
| Balanced accuracy | 0.953 |
| True positives | 46 |
| False positives | 8 |
| False negatives | 3 |
| True negatives | 243 |

## Evaluation

From the repository root, with `TYPESAFE_API_KEY` configured in `.env`:

```powershell
$env:CACHE = "true"

uv run evaluate.py `
  --model jev `
  --program results\jev-v5\program.json `
  --split test `
  --limit 300 `
  --threads 1 `
  --threshold 0.6 `
  --errors-out runs\jev-v5-errors.jsonl
```

The original Gemini V5 artifact is in `results/v5/program.json`; this file
contains the same instruction text in the Jev-compatible `predictor` format.
