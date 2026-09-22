"""Download MRBench and write data/mrbench.jsonl with balanced train/val splits.

Splits are by conversation, so one conversation never lands in two splits.
train/val are balanced yes/no and interleaved, so the first N rows of a split
stay balanced. Leftover "no" rows from train/val conversations are marked "unused".
"""

import json
import random
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/main/MRBench/MRBench_V2.json"
RAW = Path("data/raw/MRBench_V2.json")
OUT = Path("data/mrbench.jsonl")
CONVS_PER_SPLIT = 30  # conversations for train and for val each; the rest is test
SEED = 0


def clean(text):
    return text.replace(" ", " ").strip()


def rows_for(conv):
    for tutor, r in conv["anno_llm_responses"].items():
        yield {
            "id": f"{conv['conversation_id']}_{tutor}",
            "conversation": clean(conv["conversation_history"]),
            "response": clean(r["response"]),
            "label": "no" if r["annotation"]["Revealing_of_the_Answer"] == "No" else "yes",
            "solution": clean(conv["Ground_Truth_Solution"]),
            "source": conv["Data"],
            "tutor": tutor,
        }


def balanced(rows, rng):
    yes = [r for r in rows if r["label"] == "yes"]
    no = [r for r in rows if r["label"] == "no"]
    rng.shuffle(yes)
    rng.shuffle(no)
    n = min(len(yes), len(no))
    picked = [r for pair in zip(yes[:n], no[:n]) for r in pair]
    return picked, yes[n:] + no[n:]


def main():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    convs = json.loads(RAW.read_text())
    rng = random.Random(SEED)
    rng.shuffle(convs)

    groups = {
        "train": convs[:CONVS_PER_SPLIT],
        "val": convs[CONVS_PER_SPLIT : 2 * CONVS_PER_SPLIT],
        "test": convs[2 * CONVS_PER_SPLIT :],
    }
    out = []
    for split, cs in groups.items():
        rows = [r for c in cs for r in rows_for(c)]
        if split == "test":
            out += [{**r, "split": "test"} for r in rows]
            continue
        picked, rest = balanced(rows, rng)
        out += [{**r, "split": split} for r in picked]
        out += [{**r, "split": "unused"} for r in rest]

    with OUT.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for split in ("train", "val", "test", "unused"):
        rows = [r for r in out if r["split"] == split]
        yes = sum(r["label"] == "yes" for r in rows)
        print(f"{split:6} {len(rows):5} rows, {yes} yes")


if __name__ == "__main__":
    main()
