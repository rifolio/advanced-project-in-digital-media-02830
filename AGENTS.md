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
uv run optimize.py               # GEPA prompt optimisation
uv run evaluate.py --selftest     # scoring math check
```

Always `uv run`, never bare `python`/`python3`. `evaluate.py --selftest` runs
a small assertion check on the scoring math; run it after touching
`evaluate.py`.

## Layout

```
config.py        model selection (env vars / CLI flags)
prepare_data.py  downloads MRBench, builds data/mrbench.jsonl
judge.py         judge signature, ChainOfThought program, data loader, GEPA metric
evaluate.py      recall/precision/F1/balanced accuracy on a split
optimize.py      GEPA prompt optimisation, saves runs/gepa/program.json
```

## Conventions

- `uv` only. Add dependencies with `uv add <package>`, never pip or editing
  pyproject.toml by hand.
- Never read or print `.env` or its contents. Document new variables in
  `.env.example` instead.
- Model names live only in `config.py` (env vars `JUDGE_MODEL`,
  `REFLECTION_MODEL`, `NUM_THREADS`, `NUM_RETRIES`, `CACHE`) or as CLI flags on top of it. Never
  hardcode a model string anywhere else.
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
