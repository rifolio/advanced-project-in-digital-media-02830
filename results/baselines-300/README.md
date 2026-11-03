# Baselines on 300 MRBench test rows

Every judge we could run, on the same 300 rows used for the V5 benchmark
(`--split test --limit 300`, 49 yes / 251 no), with the dspy cache off
(`CACHE=false`, the default), so latencies are real calls. Run 2026-10-29/30.
Raw rows are in [`results.jsonl`](./results.jsonl), the same format as
`runs/results.jsonl`. V5 is `results/v5/program.json` (for Jev
`results/jev-v5/program.json`), which GEPA tuned on Gemini flash-lite.

The first six rows used the moving aliases (`jev-latest`,
`gemini-flash-lite-latest`). Every later row uses a pinned version; `config.py`
now pins `gemini` to `gemini-3.5-flash-lite` and `jev` to `jev-1.14.0`.

| Judge | Model | Prompt | Acc | Bal. acc | Precision | Recall | F1 | AUC | TP | FP | FN | TN | Latency (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev | `typesafe/jev-latest` | baseline, threshold 0.5 | 0.907 | 0.944 | 0.636 | 1.000 | 0.778 | 0.995 | 49 | 28 | 0 | 223 | 0.17 |
| laya | `laya/english` | baseline, threshold 0.5 | 0.583 | 0.488 | 0.155 | 0.347 | 0.214 | 0.506 | 17 | 93 | 32 | 158 | 0.57 |
| laya-multi | `laya/multilingual` | baseline, threshold 0.5 | 0.483 | 0.355 | 0.066 | 0.163 | 0.094 | 0.266 | 8 | 114 | 41 | 137 | 0.26 |
| gemini | `gemini/gemini-flash-lite-latest` | baseline | 0.893 | 0.887 | 0.623 | 0.878 | 0.729 | - | 43 | 26 | 6 | 225 | 0.83 |
| jev | `typesafe/jev-latest` | V5, threshold 0.6 | 0.957 | 0.949 | 0.821 | 0.939 | 0.876 | 0.995 | 46 | 10 | 3 | 241 | 0.18 |
| gemini | `gemini/gemini-flash-lite-latest` | V5 | 0.967 | 0.947 | 0.882 | 0.918 | 0.900 | - | 45 | 6 | 4 | 245 | 1.07 |
| jev-1.14 | `typesafe/jev-1.14.0` | baseline, threshold 0.5 | 0.907 | 0.944 | 0.636 | 1.000 | 0.778 | 0.994 | 49 | 28 | 0 | 223 | 0.17 |
| jev-1.13 | `typesafe/jev-1.13.0` | baseline, threshold 0.5 | 0.910 | 0.946 | 0.645 | 1.000 | 0.784 | 0.996 | 49 | 27 | 0 | 224 | 0.17 |
| jev-1.14 | `typesafe/jev-1.14.0` | V5, threshold 0.6 | 0.960 | 0.951 | 0.836 | 0.939 | 0.885 | 0.995 | 46 | 9 | 3 | 242 | 0.18 |
| jev-1.13 | `typesafe/jev-1.13.0` | V5, threshold 0.6 | 0.960 | 0.951 | 0.836 | 0.939 | 0.885 | 0.997 | 46 | 9 | 3 | 242 | 0.19 |
| gemini-3.5-flash-lite | `gemini/gemini-3.5-flash-lite` | baseline | 0.907 | 0.903 | 0.657 | 0.898 | 0.759 | - | 44 | 23 | 5 | 228 | 0.81 |
| gemini-3.5-flash-lite | `gemini/gemini-3.5-flash-lite` | V5 | 0.970 | 0.941 | 0.917 | 0.898 | 0.907 | - | 44 | 4 | 5 | 247 | 1.05 |
| gemini-3.5-flash | `gemini/gemini-3.5-flash` | baseline | 0.923 | 0.946 | 0.686 | 0.980 | 0.807 | - | 48 | 22 | 1 | 229 | 2.77 |
| gemini-3.5-flash | `gemini/gemini-3.5-flash` | V5 | 0.930 | 0.934 | 0.719 | 0.939 | 0.814 | - | 46 | 18 | 3 | 233 | 3.50 |
| gemini-3.6-flash | `gemini/gemini-3.6-flash` | baseline | 0.920 | 0.952 | 0.671 | 1.000 | 0.803 | - | 49 | 24 | 0 | 227 | 2.81 |
| gemini-3.6-flash | `gemini/gemini-3.6-flash` | V5 | 0.933 | 0.936 | 0.730 | 0.939 | 0.821 | - | 46 | 17 | 3 | 234 | 3.42 |
| gemini-3.7-flash | `gemini/gemini-3.7-flash` | baseline | 0.937 | 0.954 | 0.727 | 0.980 | 0.835 | - | 48 | 18 | 1 | 233 | 3.46 |
| gemini-3.7-flash | `gemini/gemini-3.7-flash` | V5 | 0.937 | 0.929 | 0.750 | 0.918 | 0.826 | - | 45 | 15 | 4 | 236 | 4.44 |
| gemini-3.8-flash | `gemini/gemini-3.8-flash` | baseline | 0.940 | 0.956 | 0.738 | 0.980 | 0.842 | - | 48 | 17 | 1 | 234 | 3.43 |
| gemini-3.8-flash | `gemini/gemini-3.8-flash` | V5 | 0.940 | 0.931 | 0.763 | 0.918 | 0.833 | - | 45 | 14 | 4 | 237 | 4.06 |
| gemini-3.1-pro | `gemini/gemini-3.1-pro-preview` | baseline | 0.920 | 0.944 | 0.676 | 0.980 | 0.800 | - | 48 | 23 | 1 | 228 | 3.99 |
| gemini-3.1-pro | `gemini/gemini-3.1-pro-preview` | V5 | 0.927 | 0.932 | 0.708 | 0.939 | 0.807 | - | 46 | 19 | 3 | 232 | 4.75 |
| gemini-4-argon | `gemini/gemini-4-argon` | baseline | 0.923 | 0.954 | 0.681 | 1.000 | 0.810 | - | 49 | 23 | 0 | 228 | 3.48 |
| gemini-4-argon | `gemini/gemini-4-argon` | V5 | 0.927 | 0.948 | 0.696 | 0.980 | 0.814 | - | 48 | 21 | 1 | 230 | 5.31 |

## What it shows

- **V5 only helps the model it was tuned on.** Flash-lite goes from 0.907 to
  0.970 accuracy (FP 23 to 4). On every stronger Gemini model V5 adds at most
  one point of accuracy and lowers balanced accuracy, because it trades recall
  for a few fewer false positives.
- **Stronger models are not better judges out of the box.** Baseline 3.5 to 4
  Gemini models all reach recall 0.98-1.00 but flag 17-24 non-reveals as
  reveals. Best balanced accuracy is 3.8 flash baseline (0.956); best F1 is
  3.5 flash-lite with V5 (0.907), at a quarter of the latency.
- **Jev is the fastest and ranks almost perfectly** (AUC 0.994-0.997, 0.17 s).
  Its errors are threshold errors. 1.13 and 1.14 give the same result with V5.
- **Pinned 3.5 flash-lite reproduces the alias run** (V5 0.970 vs 0.967), so
  `gemini-flash-lite-latest` most likely pointed at it. Not proven.
- Laya is at chance or below.
- The same non-reveal rows are probably flagged by every model, which would
  point at MRBench label noise (Juan's note). Not checked yet: rerun with
  `--errors-out` and intersect the false positives.
- Not run, no credentials yet: gpt-oss, Claude, GPT on Azure, DTU, Kev.

Reproduce:

```bash
G="gemini-3.5-flash-lite gemini-3.5-flash gemini-3.6-flash gemini-3.7-flash gemini-3.8-flash gemini-3.1-pro gemini-4-argon"
uv run evaluate.py --model jev-1.14 jev-1.13 laya laya-multi $G --split test --limit 300
uv run evaluate.py --model jev-1.14 jev-1.13 --program results/jev-v5/program.json --threshold 0.6 --split test --limit 300
for m in $G; do uv run evaluate.py --model $m --program results/v5/program.json --split test --limit 300; done
```
