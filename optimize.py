"""Optimise the judge prompt with GEPA. Defaults are tiny (about one iteration).

  uv run optimize.py
  uv run optimize.py --val-limit 74 --iterations 5 --reflection-model gemini/gemini-3.5-flash
"""

import argparse
from pathlib import Path

import dspy

import config
from judge import load, make_judge, metric

MINIBATCH = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help=f"judge model (default {config.JUDGE_MODEL})")
    p.add_argument("--reflection-model", help=f"default {config.REFLECTION_MODEL}")
    p.add_argument("--train-limit", type=int)
    p.add_argument("--val-limit", type=int, default=20)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    p.add_argument("--out", default="runs/gepa")
    args = p.parse_args()

    config.setup(args.model)
    train = load("train", args.train_limit)
    val = load("val", args.val_limit)

    # ponytail: budget math approximates GEPA's loop (one full val eval at start,
    # then per iteration two minibatches + one val eval). Use --iterations to scale.
    budget = len(val) + args.iterations * (2 * MINIBATCH + len(val))
    print(f"train={len(train)} val={len(val)} max_metric_calls={budget}")

    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=config.reflection_lm(args.reflection_model),
        max_metric_calls=budget,
        reflection_minibatch_size=MINIBATCH,
        use_merge=False,
        num_threads=args.threads,
        track_stats=True,
        log_dir=args.out,
    )
    optimized = gepa.compile(make_judge(), trainset=train, valset=val)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    optimized.save(f"{args.out}/program.json")
    print("\nOptimised instructions:\n")
    print(optimized.predict.signature.instructions)
    print(f"\nSaved to {args.out}/program.json")


if __name__ == "__main__":
    main()
