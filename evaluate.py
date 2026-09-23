"""Run one or more judges on a split and compare recall/precision/F1 on "yes"
(answer revealed). Each run is appended to runs/results.jsonl.

  uv run evaluate.py --limit 50
  uv run evaluate.py --model gemini gpt-oss jev --limit 300
  uv run evaluate.py --program runs/gepa/program.json --model openai/gpt-4o-mini
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import dspy

import config
from judge import load, make_judge

RESULTS = "runs/results.jsonl"
COLS = ["recall", "precision", "f1", "balanced_acc", "failed", "latency_s"]


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


class Timed(dspy.Module):
    """Wraps a judge to record per-call latency."""

    def __init__(self, judge):
        super().__init__()
        self.judge, self.times = judge, []

    def forward(self, **inputs):
        t = time.perf_counter()
        out = self.judge(**inputs)
        self.times.append(time.perf_counter() - t)
        return out


def run(name, data, args):
    judge = Timed(make_judge(name, args.program))
    result = dspy.Evaluate(
        devset=data,
        metric=lambda g, pr, trace=None: getattr(pr, "reveals_answer", None) == g.label,
        num_threads=args.threads,
        display_progress=True,
    )(judge)
    golds = [ex.label for ex, _, _ in result.results]
    preds = [getattr(pr, "reveals_answer", None) for _, pr, _ in result.results]
    s = scores(golds, preds)
    s["latency_s"] = statistics.median(judge.times) if judge.times else 0.0
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, help="only the first N examples")
    p.add_argument(
        "--model", nargs="+", default=[config.JUDGE_MODEL],
        help=f"judge names ({', '.join(config.JUDGES)}) or LiteLLM model strings",
    )
    p.add_argument("--program", help="saved program.json from optimize.py")
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    args = p.parse_args()

    data = load(args.split, args.limit)
    rows = []
    for name in args.model:
        model, _ = config.resolve(name)
        print(f"\n{name} ({model}) on {args.split} ({len(data)} examples), program={args.program or 'baseline'}")
        s = run(name, data, args)
        for k, v in s.items():
            print(f"{k:13} {v:.3f}" if isinstance(v, float) else f"{k:13} {v}")
        rows.append((name, s))
        Path(RESULTS).parent.mkdir(exist_ok=True)
        with open(RESULTS, "a") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"), "judge": name, "model": model,
                "program": args.program, "split": args.split, "n": len(data), **s,
            }) + "\n")

    if len(rows) > 1:
        w = max(len(name) for name, _ in rows) + 2
        print("\n" + f"{'judge':{w}}" + "".join(f"{c:>14}" for c in COLS))
        for name, s in rows:
            print(f"{name:{w}}" + "".join(
                f"{s[c]:>14.3f}" if isinstance(s[c], float) else f"{s[c]:>14}" for c in COLS
            ))


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
