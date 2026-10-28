"""Hand-label a random sample of training rows without seeing their labels, then
compare with the labels the pipeline gave them.

  uv run label_sample.py                    # 40 rows from data/gsm8k.jsonl
  uv run label_sample.py --n 60 --seed 1

Answers are saved to runs/blind_labels.jsonl as you go, so you can stop with q or
Ctrl+C and pick up later; rows you already labelled are skipped.
"""

import argparse
import json
import random
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

LABELS = Path("runs/blind_labels.jsonl")


def show(i, n, row):
    print("\n" + "=" * 80 + f"\n[{i}/{n}]  {row['id'].rsplit('_', 1)[0]}\n")
    for turn in row["conversation"].split(" \n "):
        print(textwrap.fill(turn.strip(), 80, subsequent_indent="    "))
    print("\nTUTOR'S NEXT MESSAGE:\n" + textwrap.fill(row["response"], 80, initial_indent="  ", subsequent_indent="  "))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gsm8k.jsonl")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]
    saved = [json.loads(line) for line in LABELS.open()] if LABELS.exists() else []
    human = {s["id"]: s["human"] for s in saved if s["data"] == args.data}

    print("Does the tutor's message reveal the answer? It does if it states the final answer,\n"
          "gives the key intermediate result that makes the answer trivial, or fully works\n"
          "through the step the student is stuck on. Hints, guiding questions, pointing out\n"
          "where an error is and asking the student to re-check are not reveals.")
    LABELS.parent.mkdir(exist_ok=True)
    for i, row in enumerate(sample, 1):
        if row["id"] in human:
            continue
        show(i, len(sample), row)
        answer = ""
        while answer not in ("y", "n", "s", "q"):
            answer = input("\nReveals the answer? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()[:1]
        if answer == "q":
            break
        if answer == "s":
            continue
        human[row["id"]] = "yes" if answer == "y" else "no"
        with LABELS.open("a") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "data": args.data,
                                "id": row["id"], "human": human[row["id"]]}) + "\n")

    done = [r for r in sample if r["id"] in human]
    if not done:
        return
    pairs = Counter((r["label"], human[r["id"]]) for r in done)
    agree = pairs[("yes", "yes")] + pairs[("no", "no")]
    print(f"\n{len(done)}/{len(sample)} labelled, agreement with the pipeline {agree}/{len(done)} "
          f"({agree / len(done):.0%})")
    print(f"  pipeline yes: you said yes {pairs[('yes', 'yes')]}, no {pairs[('yes', 'no')]}")
    print(f"  pipeline no:  you said yes {pairs[('no', 'yes')]}, no {pairs[('no', 'no')]}")
    for r in done:
        if r["label"] != human[r["id"]]:
            print(f"  disagree {r['id']}: pipeline {r['label']}, you {human[r['id']]}  ({r.get('style', '')})")


if __name__ == "__main__":
    main()
