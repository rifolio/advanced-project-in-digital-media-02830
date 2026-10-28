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

# refine an existing prompt with extra weight on avoiding false positives
uv run optimize.py --program runs/gepa-v4/program.json --iterations 5 --out runs/gepa-v5

# evaluate the optimised program
uv run evaluate.py --program runs/gepa/program.json

# compare several judges on the same examples (table + runs/results.jsonl)
uv run evaluate.py --model gemini gpt-oss jev --limit 300

# optimise Jev instructions with Gemini reflection (uses TypeSafe API calls)
uv run optimize_jev.py --val-limit 20 --iterations 1 --out runs/jev-gepa

# turn GSM8K into balanced training data (data/gsm8k.jsonl + data/kev/ for Kev);
# an LLM writes a reveal and a guide reply per problem, a judge checks the labels
uv run prepare_training_data.py --limit 200

# hand-label a blind sample of it and see how often you agree with the pipeline
uv run label_sample.py --n 40

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

Small open Jev-like models, for testing before fine-tuning. Same yes/no
question as Jev; probability judges also report `auc` (0.5 = chance), since
a fixed 0.5 cut can hide a model that ranks well, or backwards.

```bash
# Laya (in-process encoder, ~0.1-0.6s per example on a Mac)
uv sync --extra local
uv run evaluate.py --model laya laya-multi --split val

# Kev (Qwen-based, runs as a local server from its own repo, same API as Jev)
git clone https://github.com/jaredpalmer/kev ../kev && cd ../kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
uv run evaluate.py --model kev --split val      # back in this repo
```

Kev-4B needs more than 16 GB RAM to be usable; point `KEV_URL` at a Modal
endpoint instead (see `.env.example`).

## Project layout

```
config.py        judge registry and model selection (env vars / CLI flags)
prepare_data.py  downloads MRBench, builds data/mrbench.jsonl
prepare_training_data.py  rewrites GSM8K into training data, data/gsm8k.jsonl + data/kev/
label_sample.py  blind hand-labelling of training rows, agreement with the pipeline
judge.py         DSPy judge, Jev/Kev and Laya judges, data loader, GEPA metric
evaluate.py      compare judges: recall/precision/F1/latency on a split
optimize.py      GEPA prompt optimisation, saves runs/gepa/program.json
optimize_jev.py  GEPA optimisation of Jev instructions via its API
data/            mrbench.jsonl (committed), raw/ downloads (git-ignored)
runs/            GEPA output and results.jsonl (git-ignored)
results/         selected judge programs and benchmark notes (committed)
```

## Data

Each row is one tutor response labelled `yes` (reveals the answer) or `no`,
from MRBench's `Revealing_of_the_Answer` annotation. Splits are by
conversation, so a conversation never crosses splits. Train and val are
balanced and interleaved (30 conversations each); test is the rest, with a
natural class skew (roughly 18% yes). Leftover "no" rows from train/val
conversations are kept with `split: "unused"`.

The judge treats a response as a reveal when it states the final answer, gives
the key intermediate result that makes the answer trivial, or fully works
through the step the student is stuck on. Hints, guiding questions, pointing
out an error, and asking the student to re-check their work are not reveals.

## Next steps

- Run GEPA optimisation with a real budget and compare against baseline
- Add MathDial and GSM8K-derived data
- Try other judge models (gpt-oss-safeguard)
- Look into fine-tuning once the evaluation framework is solid
