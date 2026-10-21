# V5 judge program

This directory contains the selected V5 DSPy judge program:

- [`program.json`](./program.json) — optimized judge prompt and configuration.

## Benchmark

Evaluated on 300 MRBench test examples with the `gemini` judge
(`gemini/gemini-flash-lite-latest`):

| Metric | Value |
|---|---:|
| Accuracy | 0.967 |
| Precision | 0.898 |
| Recall | 0.898 |
| F1 | 0.898 |
| Balanced accuracy | 0.939 |
| True positives | 44 |
| False positives | 5 |
| False negatives | 5 |
| True negatives | 246 |

The result above was obtained with DSPy caching enabled. Because the judge
uses a remote generative model, results from uncached evaluations can vary.
Keep the model name, dataset split, example limit, and cache setting fixed
when reproducing the benchmark.

## Evaluation

From the repository root:

```powershell
$env:CACHE = "true"
uv run evaluate.py `
  --program results\v5\program.json `
  --model gemini `
  --split test `
  --limit 300 `
  --threads 1
```

## MathDial smoke test

MathDial does not provide the `yes`/`no` reveal labels used by MRBench.
Therefore, it is not evaluated with the MRBench accuracy command above.
Instead, the first 30 MathDial test conversations were manually inspected,
and 24 unambiguous teacher interventions were selected:

- [`mathdial-v6-review-results.jsonl`](../mathdial-v6-review-results.jsonl)
  — reviewed labels and V6 predictions.
- [`data/MathDial/test.jsonl`](../../data/MathDial/test.jsonl)
  — original conversations referenced by `row` and `teacher_turn`.

The review policy was:

- `yes` — the teacher explicitly gives the answer or completes the remaining
  reasoning.
- `no` — the teacher asks a question, gives a hint, or provides partial
  guidance while leaving meaningful work for the student.

Example reviewed intervention:

```json
{
  "id": "mathdial_row2_qid6000044_t9",
  "label": "yes",
  "prediction": "yes",
  "response": "(telling)No the answer is $500 as $200 + $150 + $150 is $500"
}
```

To inspect the saved smoke-test results:

```powershell
$rows = Get-Content results\mathdial-v6-review-results.jsonl |
  Where-Object { $_.Trim() } |
  ConvertFrom-Json

Write-Output ("reviewed: " + @($rows).Count)
$rows | Group-Object label, prediction | Select-Object Name, Count
```

The V6 smoke test produced 24 reviewed records: 7 clear reveals and 17
non-reveals. V6 had 5 true positives, 0 false positives, 2 false negatives,
and 17 true negatives. This is a qualitative smoke test, not a representative
MathDial benchmark.
