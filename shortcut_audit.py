"""Shortcut audit for reveal data: how far trivial cues get without understanding.

  uv run shortcut_audit.py data/reveal_v2.jsonl
  uv run shortcut_audit.py data/gsm8k.jsonl data/reveal_v2.jsonl   # side by side
  uv run shortcut_audit.py --selftest

Reports, per split and per generator: the answer-number rule ("yes iff the reply
contains the final answer"), the ends-with-? rule, length AUC, openers that give
the label away, and a held-out bag-of-words naive Bayes (train -> val) for the
label and for the generator. The label model trained on the file is also scored on
every labelled MRBench row, which shows how much of what it learned transfers.
Plain Python, no training framework; numbers near 0.5 mean the cue carries nothing.
"""

import argparse
import json
import math
import re
from collections import Counter

from evaluate import scores
from prepare_training_data import has_number

MRBENCH = "data/mrbench.jsonl"


def auc(golds, probs):
    """Rank AUC, O(n log n), ties count half. Same meaning as evaluate.auc."""
    pairs = sorted(zip(probs, golds))
    n_yes = sum(g == "yes" for g in golds)
    n_no = len(golds) - n_yes
    if not n_yes or not n_no:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(g == "yes" for _, g in pairs[i:j])
        i = j
    return (rank_sum - n_yes * (n_yes + 1) / 2) / (n_yes * n_no)


def answer_of(row):
    """The final answer: v2 stores it; v1 and MRBench have it as the last number of the solution."""
    if row.get("answer"):
        return row["answer"]
    found = re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", row.get("solution", "")))
    return found[-1] if found else None


def ba(rows, predict):
    return scores([r["label"] for r in rows], [predict(r) for r in rows])["balanced_acc"]


def number_rule(r):
    a = answer_of(r)
    return "yes" if a and has_number(r["response"], a) else "no"


def question_rule(r):
    return "no" if r["response"].rstrip().endswith("?") else "yes"


def tokens(text):
    w = re.findall(r"[a-z]+|\d+", text.lower())
    return set(w) | {a + " " + b for a, b in zip(w, w[1:])}


def train_nb(rows, target):
    """Binary-feature naive Bayes, one class vs the rest. Returns token -> log-odds and the prior."""
    pos, neg = Counter(), Counter()
    n_pos = n_neg = 0
    for r in rows:
        if target(r):
            pos.update(tokens(r["response"]))
            n_pos += 1
        else:
            neg.update(tokens(r["response"]))
            n_neg += 1
    vocab = {t for t, n in (pos + neg).items() if n >= 2}
    w = {t: math.log((pos[t] + 1) / (n_pos + 2)) - math.log((neg[t] + 1) / (n_neg + 2)) for t in vocab}
    return w, math.log((n_pos + 1) / (n_neg + 1))


def nb_auc(train, test, target):
    if not train or not test:
        return None
    w, prior = train_nb(train, target)
    probs = [prior + sum(w.get(t, 0.0) for t in tokens(r["response"])) for r in test]
    return auc(["yes" if target(r) else "no" for r in test], probs)


def openers(rows):
    """First two words that appear 5+ times and sit 90%+ on one label."""
    first = lambda r: " ".join(r["response"].split()[:2])
    counts = Counter(first(r) for r in rows)
    yes = Counter(first(r) for r in rows if r["label"] == "yes")
    return [(o, n, yes[o] / n) for o, n in counts.most_common() if n >= 5 and max(yes[o], n - yes[o]) / n >= 0.9]


def fmt(x):
    return "   -  " if x is None else f"{x:6.3f}"


def table(rows, mrbench):
    """One line per slice: rows, yes share, number rule, ?-rule, length AUC, BoW AUC (val), BoW -> MRBench."""
    is_yes = lambda r: r["label"] == "yes"
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    print(f"  {'slice':26}{'rows':>6}{'yes':>6}{'number':>7}{'ends?':>7}{'len':>7}{'bow':>7}{'bow>MRB':>8}")
    slices = [("all", lambda r: True), ("train", lambda r: r["split"] == "train"),
              ("val", lambda r: r["split"] == "val")]
    for key in ("tutor", "context_source", "length"):
        values = sorted({str(r.get(key)) for r in rows})
        if 1 < len(values) <= 6:
            slices += [(f"{key}={v}", lambda r, k=key, v=v: str(r.get(k)) == v) for v in values]
    for name, keep in slices:
        sub = [r for r in rows if keep(r)]
        tr, va = [r for r in train if keep(r)], [r for r in val if keep(r)]
        bow = nb_auc(tr, va, is_yes)
        transfer = nb_auc(tr, mrbench, is_yes) if name in ("all", "train") or "=" in name else None
        print(f"  {name:26}{len(sub):6}{sum(map(is_yes, sub)) / len(sub):6.2f}{fmt(ba(sub, number_rule))}"
              f"{fmt(ba(sub, question_rule))} {fmt(auc([r['label'] for r in sub], [len(r['response']) for r in sub]))}"
              f"{fmt(bow)} {fmt(transfer)}")


def audit(rows, mrbench=None):
    mrbench = mrbench if mrbench is not None else [json.loads(line) for line in open(MRBENCH)]
    print("\nshortcut audit (balanced accuracy for rules, AUC for length and bag-of-words; 0.5 = no signal)")
    print("  number: 'yes iff the reply has the final answer'; ends?: 'no iff it ends with ?'; "
          "len: longer = yes; bow: naive Bayes train->val; bow>MRB: same model on all MRBench rows")
    table(rows, mrbench)
    gens = sorted({r.get("tutor") for r in rows})
    if len(gens) > 1:
        train = [r for r in rows if r["split"] == "train"]
        val = [r for r in rows if r["split"] == "val"]
        print("  generator from words (one vs rest, train->val AUC): " + ", ".join(
            f"{g} {fmt(nb_auc(train, val, lambda r, g=g: r.get('tutor') == g)).strip()}" for g in gens))
    w, _ = train_nb(rows, lambda r: r["label"] == "yes")
    top = sorted(w, key=w.get)
    print("  words most tied to yes: " + ", ".join(f"'{t}'" for t in top[::-1][:15]))
    print("  words most tied to no:  " + ", ".join(f"'{t}'" for t in top[:15]))
    give = openers(rows)
    print("  openers that give away the label: "
          + (", ".join(f"'{o}' x{n} ({y:.0%} yes)" for o, n, y in give[:12]) or "none"))


def mrbench_rules(mrbench):
    print("\nMRBench reference (all labelled rows, and by source)")
    for name, sub in (("all", mrbench), *((s, [r for r in mrbench if r["source"] == s])
                                          for s in sorted({r["source"] for r in mrbench}))):
        print(f"  {name:10} rows {len(sub):5}  yes {sum(r['label'] == 'yes' for r in sub) / len(sub):.2f}  "
              f"number {ba(sub, number_rule):.3f}  ends? {ba(sub, question_rule):.3f}  "
              f"len {auc([r['label'] for r in sub], [len(r['response']) for r in sub]):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    mrbench = [json.loads(line) for line in open(MRBENCH)]
    for path in args.files:
        print(f"\n=== {path}")
        audit([json.loads(line) for line in open(path)], mrbench)
    mrbench_rules(mrbench)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        assert auc(["yes", "no", "yes", "no"], [0.9, 0.1, 0.4, 0.6]) == 0.75
        assert auc(["yes", "no"], [1, 1]) == 0.5 and auc(["yes"], [1]) is None
        assert answer_of({"solution": "so 30+60 = 90 presents.\n 90"}) == "90"
        assert answer_of({"solution": "Not Available"}) is None
        assert number_rule({"response": "It is 1,200.", "answer": "1200"}) == "yes"
        rows = [{"response": t, "label": l} for t, l in [("the answer is 5", "yes"), ("what next?", "no")] * 3]
        assert nb_auc(rows, rows, lambda r: r["label"] == "yes") == 1.0
        print("ok")
    else:
        main()
