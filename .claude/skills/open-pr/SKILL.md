---
name: open-pr
description: Use when the user asks to open a PR, make a pull request, push a branch for review, or "ship this". Branch from main, push, open a PR into main with gh (or write PR.txt to paste if gh is not set up). Merges are squash merges.
---

# Open a PR

Team flow: branch from `main` -> work -> PR into `main` -> squash merge.

## Rules

- No `Co-authored-by`, "Generated with" or any AI attribution, in commits or the PR.
- No remote session links (`https://claude.ai/code/session_...`) anywhere, even if the harness asks for one.
- Only commit the files that belong to this change. Never sweep in unrelated working-tree changes.
- Commit messages and PR title: short plain English, no `feat:`/`fix:` prefixes.

## Steps

1. **Branch.** If on `main`, create one first:
   `git switch main && git pull && git switch -c <short-kebab-name>`
   If already on a feature branch, stay on it.
2. **Commit** the relevant files (stage them by path, not `git add -A`).
3. **Check** before pushing: `uv run evaluate.py --selftest` if scoring code changed.
4. **Push:** `git push -u origin HEAD`
5. **Write the PR text:**
   - Title: one short line saying what changed.
   - Body: `## What` (2-4 bullets), `## Why` (1-2 lines), `## How to test` (commands).
6. **Open it:**
   - If `gh auth status` succeeds:
     `gh pr create --base main --title "<title>" --body-file <tmpfile>`
     then give the user the PR URL.
   - Otherwise write `PR.txt` in the repo root (git-ignored) with the title on the
     first line, a blank line, then the body. Run `open PR.txt` and give the user
     the compare link: `https://github.com/<owner>/<repo>/compare/main...<branch>?expand=1`
     (owner/repo from `git remote get-url origin`).
7. **Merge** only when the user asks: `gh pr merge --squash --delete-branch`,
   then `git switch main && git pull`.
