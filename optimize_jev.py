"""Optimise Jev's TypeSafe instructions with GEPA.

  uv run optimize_jev.py --val-limit 20 --iterations 1 --out runs/jev-gepa
"""

import argparse
from pathlib import Path

import dspy

import config
from judge import JevJudge, load, make_judge

MINIBATCH = 3


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    predicted = getattr(pred, "reveals_answer", None)
    if predicted == gold.label:
        return dspy.Prediction(
            score=1.0,
            feedback=f"Correct: predicted '{predicted}' for example {gold.id}.",
        )
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Wrong on example {gold.id}: predicted '{predicted}', "
            f"human label is '{gold.label}'.\n"
            f"Conversation excerpt: {gold.conversation[:1000]}\n"
            f"Tutor response: {gold.response[:1000]}\n"
            f"Ground-truth solution: {gold.solution}\n"
            "Revise the instructions to distinguish a reveal from a hint or "
            "partial correction without changing the binary output contract."
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reflection-model", help=f"default {config.REFLECTION_MODEL}")
    parser.add_argument("--out", default="runs/jev-gepa")
    args = parser.parse_args()

    judge = make_judge("jev")
    if not isinstance(judge, JevJudge):
        raise SystemExit("The jev registry entry did not create a JevJudge")
    train = load("train", args.train_limit)
    val = load("val", args.val_limit)
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
    optimized = gepa.compile(judge, trainset=train, valset=val)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    optimized.save(f"{args.out}/program.json")
    print("\nOptimised Jev instructions:\n")
    print(optimized.predictor.signature.instructions)
    print(f"\nSaved to {args.out}/program.json")


if __name__ == "__main__":
    main()
