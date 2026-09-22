"""Run the judge on a split and print recall/precision/F1 on "yes" (answer revealed).

  uv run evaluate.py --limit 50
  uv run evaluate.py --program runs/gepa/program.json --model openai/gpt-4o-mini
"""

import argparse

import dspy

import config
from judge import load, make_judge


def scores(golds, preds):
    tp = sum(g == "yes" and p == "yes" for g, p in zip(golds, preds))
    fp = sum(g == "no" and p == "yes" for g, p in zip(golds, preds))
    fn = sum(g == "yes" and p != "yes" for g, p in zip(golds, preds))
    tn = sum(g == "no" and p == "no" for g, p in zip(golds, preds))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_acc": (recall + spec) / 2,
        "accuracy": (tp + tn) / len(golds),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "failed": sum(p not in ("yes", "no") for p in preds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, help="only the first N examples")
    p.add_argument("--model", help=f"judge model (default {config.JUDGE_MODEL})")
    p.add_argument("--program", help="saved program.json from optimize.py")
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    args = p.parse_args()

    lm = config.setup(args.model)
    data = load(args.split, args.limit)
    print(f"{lm.model} on {args.split} ({len(data)} examples), program={args.program or 'baseline'}")

    result = dspy.Evaluate(
        devset=data,
        metric=lambda g, pr, trace=None: getattr(pr, "reveals_answer", None) == g.label,
        num_threads=args.threads,
        display_progress=True,
    )(make_judge(args.program))

    golds = [ex.label for ex, _, _ in result.results]
    preds = [getattr(pr, "reveals_answer", None) for _, pr, _ in result.results]
    for k, v in scores(golds, preds).items():
        print(f"{k:13} {v:.3f}" if isinstance(v, float) else f"{k:13} {v}")


if __name__ == "__main__":
    # ponytail: self-check of the scoring math, runs with `uv run evaluate.py --selftest`
    import sys

    if "--selftest" in sys.argv:
        s = scores(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"])
        assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (1, 1, 1, 1)
        assert s["recall"] == s["precision"] == s["f1"] == s["balanced_acc"] == 0.5
        assert scores(["yes"], [None])["failed"] == 1
        print("ok")
    else:
        main()
