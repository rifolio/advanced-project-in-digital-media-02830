# Baselines on 300 MRBench test rows

Every judge we could run on 2026-10-29, on the same 300 rows used for the V5
benchmark (`--split test --limit 300`, 49 yes / 251 no), with the dspy cache
off (`CACHE=false`, the default), so latencies are real calls. Raw rows are in
[`results.jsonl`](./results.jsonl), the same format as `runs/results.jsonl`.

| Judge | Prompt | Acc | Bal. acc | Precision | Recall | F1 | AUC | TP | FP | FN | TN | Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini (flash-lite) | baseline | 0.893 | 0.887 | 0.623 | 0.878 | 0.729 | - | 43 | 26 | 6 | 225 | 0.83 |
| gemini (flash-lite) | V5 | 0.967 | 0.947 | 0.882 | 0.918 | 0.900 | - | 45 | 6 | 4 | 245 | 1.07 |
| jev (jev-latest) | baseline, threshold 0.5 | 0.907 | 0.944 | 0.636 | 1.000 | 0.778 | 0.995 | 49 | 28 | 0 | 223 | 0.17 |
| jev (jev-latest) | V5, threshold 0.6 | 0.957 | 0.949 | 0.821 | 0.939 | 0.876 | 0.995 | 46 | 10 | 3 | 241 | 0.18 |
| laya (english) | baseline | 0.583 | 0.488 | 0.155 | 0.347 | 0.214 | 0.506 | 17 | 93 | 32 | 158 | 0.57 |
| laya-multi | baseline | 0.483 | 0.355 | 0.066 | 0.163 | 0.094 | 0.266 | 8 | 114 | 41 | 137 | 0.26 |

Notes:

- Jev's AUC is 0.995 with or without V5: it already ranks reveals almost
  perfectly, its errors are a threshold problem. Pick the threshold on val, not
  on this table.
- Gemini V5 reproduces the 0.967 in `results/v5/README.md` without the cache.
  Jev V5 is 0.957 here against 0.963 there, with `jev-latest` now on a newer
  version.
- Laya is at chance (English) or below it (multilingual).
- Not run yet, no credentials on this machine: gpt-oss (OpenRouter/Groq),
  Claude (Anthropic), GPT (Azure), DTU, and Kev (needs the local server).

Reproduce:

```bash
uv run evaluate.py --model jev laya laya-multi gemini --split test --limit 300
uv run evaluate.py --model jev --program results/jev-v5/program.json --threshold 0.6 --split test --limit 300
uv run evaluate.py --model gemini --program results/v5/program.json --split test --limit 300
```
