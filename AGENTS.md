# AGENTS.md

No Spoilers: an AI guardrail that detects when an LLM tutor response reveals
a solution instead of guiding the student. Current phase: an LLM-as-a-judge
evaluation framework built with DSPy, optimised with GEPA, evaluated on
MRBench.

## Commands

```bash
uv sync                          # install deps
uv run prepare_data.py           # build data/mrbench.jsonl
uv run evaluate.py --limit 50    # baseline eval
uv run evaluate.py --model gemini jev --limit 50   # compare judges
uv run optimize.py               # GEPA prompt optimisation
uv run evaluate.py --selftest     # scoring math check
uv run build_dataset.py --mathdial 20 --gsm8k 20   # v2 data pilot (resumes from data/raw/)
uv run shortcut_audit.py data/gsm8k.jsonl data/reveal_v2.jsonl   # shortcut audit
uv sync --extra embed             # optional deps for embed_audit.py (numpy, scikit-learn, sentence-transformers, matplotlib)
uv run embed_audit.py --model minilm   # embedding audit, results in results/embed-audit/
```

Always `uv run`, never bare `python`/`python3`. `evaluate.py --selftest` runs
a small assertion check on the scoring math; run it after touching
`evaluate.py`.

Run independent evaluations in parallel, one `evaluate.py` process per model
or prompt, never one after another. `--model a b` runs the models in sequence,
so split it: `uv run evaluate.py --model a & uv run evaluate.py --model b & wait`.
Each appends its own lines to `runs/results.jsonl`; 429s are retried.

## Layout

```
config.py        judge registry (JUDGES) and model selection
prepare_data.py  downloads MRBench, builds data/mrbench.jsonl
prepare_training_data.py  rewrites GSM8K into training data, data/gsm8k.jsonl + data/kev/
label_sample.py  blind hand-labelling of training rows, agreement with the pipeline
build_dataset.py  v2 reveal data from MathDial + GSM8K, several model families, cross-family
                 verification, data/reveal_v2.jsonl (datasheet data/reveal_v2.md)
shortcut_audit.py  number/?/length/bag-of-words shortcut audit for any reveal data file
embed_audit.py   embedding probes, label transfer, clusters and plots (EMBEDDERS in config.py)
embed_audit.ipynb  team-facing report of the audit: charts, held-out generator test, rerun switch
judge.py         DSPy judge, Jev judge, data loader, GEPA metric
evaluate.py      compare judges on a split, appends runs/results.jsonl
optimize.py      GEPA prompt optimisation, saves runs/gepa/program.json
```

## Conventions

- `uv` only. Add dependencies with `uv add <package>`, never pip or editing
  pyproject.toml by hand.
- Never read or print `.env` or its contents. Document new variables in
  `.env.example` instead.
- Model names live only in `config.py`: the `JUDGES` registry plus env vars
  (`JUDGE_MODEL`, `REFLECTION_MODEL`, `GENERATOR_MODEL`, `GPT_OSS_MODEL`, `DTU_MODEL`,
  `AZURE_MODEL`, `NUM_THREADS`, `NUM_RETRIES`, `CACHE`) or CLI flags on top.
  Never hardcode a model string anywhere else. A new provider is a new
  `JUDGES` entry, not new code, unless it is not an LLM (like Jev).
- `data/raw/` and `runs/` are git-ignored. `data/mrbench.jsonl` is generated
  by `prepare_data.py` (seeded, committed); regenerate it, never hand-edit.
- Keep scripts flat and simple, no framework scaffolding beyond what DSPy
  needs.
- Keep the test split untouched during optimisation. Only train/val feed
  GEPA.
- Before claiming evaluate.py works, run `uv run evaluate.py --selftest`.

## Git workflow

- Never commit to `main`. Branch off an up-to-date `main`, work there, open a
  PR into `main`, squash merge it.
- Never add `Co-authored-by` trailers, "Generated with" lines or any AI
  attribution to commits or PRs.
- Never put a remote session link (e.g. `https://claude.ai/code/session_...`)
  in a commit, PR title, PR body or comment.
- Commit messages: short plain English, no `feat:`/`fix:` prefixes.
- To open a PR, follow `.claude/skills/open-pr/SKILL.md` (Claude Code: the
  `open-pr` skill). It uses `gh`, or falls back to `PR.txt` to paste by hand.
