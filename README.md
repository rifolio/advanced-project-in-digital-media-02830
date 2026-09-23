# No Spoilers

DTU course project (02830 Advanced Project in Digital Media). AI usage in learning
contexts often shortcuts the learning process by handing over answers directly.
No Spoilers is a guardrail that detects when an LLM tutor's response reveals a
solution instead of guiding the student, eventually as a browser extension that
can block or rephrase such responses.

**Status: work in progress, the approach may change.** Right now we're building
an LLM-as-a-judge evaluation framework: a DSPy program that classifies a tutor
response as revealing the answer or not, optimised with GEPA, evaluated against
MRBench (human-labelled ground truth). Planned next: MathDial, GSM8K, synthetic
data, other judge models (e.g. gpt-oss-safeguard), possibly fine-tuning.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your GEMINI_API_KEY
```

## Usage

```bash
# download MRBench and build data/mrbench.jsonl (train/val/test splits)
uv run prepare_data.py

# baseline judge on a few test examples
uv run evaluate.py --limit 50

# optimise the judge prompt with GEPA (tiny default budget, ~1 iteration)
uv run optimize.py

# evaluate the optimised program
uv run evaluate.py --program runs/gepa/program.json

# compare several judges on the same examples (table + runs/results.jsonl)
uv run evaluate.py --model gemini gpt-oss jev --limit 300

# swap the judge or reflection model (judge name or any LiteLLM model string)
uv run evaluate.py --model openai/gpt-4o-mini
uv run optimize.py --reflection-model gemini/gemini-flash-latest
```

Models default to `JUDGE_MODEL` / `REFLECTION_MODEL` / `NUM_THREADS` in `.env`
(`NUM_RETRIES` sets 429 retries, `CACHE=true` turns on dspy's local cache)
(see `config.py`), and can be overridden per-run with `--model` /
`--reflection-model` / `--threads`.

Judges are named in `config.JUDGES`: `gemini` works out of the box; `gpt-oss`
(Groq or OpenRouter), `dtu` (self-hosted on the DTU GPU) and `azure` are
placeholders you switch on in `.env`; `jev` is TypeSafe AI's Jev classifier
(needs `TYPESAFE_API_KEY`, benchmark only, not optimisable with GEPA). See
`.env.example` for every variable.

## Project layout

```
config.py        judge registry and model selection (env vars / CLI flags)
prepare_data.py  downloads MRBench, builds data/mrbench.jsonl
judge.py         DSPy judge, Jev judge, data loader, GEPA metric
evaluate.py      compare judges: recall/precision/F1/latency on a split
optimize.py      GEPA prompt optimisation, saves runs/gepa/program.json
data/            mrbench.jsonl (committed), raw/ downloads (git-ignored)
runs/            GEPA output and results.jsonl (git-ignored)
```

## Data

Each row is one tutor response labelled `yes` (reveals the answer) or `no`,
from MRBench's `Revealing_of_the_Answer` annotation. Splits are by
conversation, so a conversation never crosses splits. Train and val are
balanced and interleaved (30 conversations each); test is the rest, with a
natural class skew (roughly 18% yes). Leftover "no" rows from train/val
conversations are kept with `split: "unused"`.

## Next steps

- Run GEPA optimisation with a real budget and compare against baseline
- Add MathDial and GSM8K-derived data
- Try other judge models (gpt-oss-safeguard)
- Look into fine-tuning once the evaluation framework is solid
