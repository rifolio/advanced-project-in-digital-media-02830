"""Download GSM8K and turn it into training data in data/gsm8k.jsonl.

  uv run prepare_training_data.py --limit 200     # pilot, read the output by hand
  uv run prepare_training_data.py --limit 3000    # full run, about 6,000 rows

GSM8K has maths word problems and worked solutions but no tutoring dialogue,
so an LLM (the generator model(s) from config.py) rewrites each problem into a
short tutor/student conversation in which the student goes wrong or gets stuck,
plus two possible next tutor messages:

  - a reveal response (label "yes"): gives the final answer, the key
    intermediate result, or works the stuck step fully;
  - a guide response (label "no"): a hint, guiding question, or pointer to the
    error that leaves the work to the student.

Steps:
  1. Download the GSM8K train split to data/raw/ and strip the <<...>>
     calculator notes; the final answer is the part after "####".
  2. Drop every problem whose question appears in MRBench. MathDial, and so
     MRBench, was built from GSM8K, so keeping them would leak test problems.
  3. Generate the conversation and both responses. Each call is appended to a
     stage file in data/raw/, so a rerun skips problems already done.
  4. Cut any trailing tutor lines, so the conversation ends with the student.
     Check with plain code: the conversation ends with the student, the reveal
     contains the answer, the guide and the conversation do not, and both
     responses are of similar length. Each response's ending (question or
     statement) is drawn separately from its label, so punctuation does not
     give the label away; a shortcut audit at the end reports what is left.
  5. Verify the labels with a judge; keep a problem only if both of its
     responses get the intended label.
  6. Write rows in the same format as data/mrbench.jsonl. Every problem gives
     one "yes" and one "no" row, so the data is balanced by construction.

Splits are by problem, so the yes and no rows of one problem never land in two
splits. The rows are also exported as Kev training files (API request shape plus
a label), using the same state keys and instructions as judge.JevJudge, so a
model trained on them sees exactly what it is sent at inference.
"""

import argparse
import json
import random
import re
import statistics
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy

import config
from judge import JevJudge, RevealsAnswer, load_instructions, make_judge

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
RAW = Path("data/raw/gsm8k_train.jsonl")
# Bump when the generation prompt changes, so a rerun regenerates instead of reusing old output.
PROMPT_VERSION = 5
GENERATED = Path(f"data/raw/gsm8k_generated_v{PROMPT_VERSION}.jsonl")  # stage file, one line per problem
VERIFIED = Path(f"data/raw/gsm8k_verified_v{PROMPT_VERSION}.jsonl")  # stage file, one line per row and verifier
OUT = Path("data/gsm8k.jsonl")
KEV_DIR = Path("data/kev")
MRBENCH = Path("data/mrbench.jsonl")
MATHDIAL_TEST = Path("data/MathDial/test.jsonl")
VAL_SHARE = 0.1  # share of problems held out as val (and Kev's held-out file)
MAX_LENGTH_RATIO = 3  # longer response at most this many times the shorter one
SEED = 0

OPENER = ("Tutor: Hi, could you please provide a step-by-step solution for the question below? "
          "The question is: {question} \n ")
STUDENT = [
    "makes an arithmetic slip",
    "misreads what the question asks",
    "uses the wrong operation",
    "gets partway and is stuck",
]
REVEAL = [
    "states the final answer outright",
    "gives the key intermediate result so only a trivial step is left",
    "fully works through the step the student is stuck on",
    "asks a leading question that contains the answer, e.g. 'Is it 30?'",
]
GUIDE = [
    "asks one guiding question",
    "gives a conceptual hint without numbers from the solution",
    "points to where the error is without fixing it",
    "asks the student to re-check one specific step",
]
# a key intermediate result may not contain the final number, so the answer check skips it
NO_ANSWER_CHECK = REVEAL[1]
# Endings are drawn separately from the label, so "ends with ?" does not predict it.
# Styles that are questions by nature always end with one.
QUESTION, STATEMENT = "ends with a question to the student", "ends with a statement or instruction, not a question"
ALWAYS_QUESTION = {REVEAL[3], GUIDE[0]}


class RewriteGSM8K(dspy.Signature):
    """Write a realistic tutoring exchange about a maths word problem. The tutor
    has already asked the student to solve it. The student attempts it and goes
    wrong in the given way. The exchange ends with the student still wrong or
    stuck: nobody, student or tutor, states or confirms the correct final answer,
    and the tutor never fixes the mistake for the student.

    Then write two possible next tutor messages, each 1-3 sentences and of similar
    length and tone: one that reveals the answer in the given reveal style, and one
    that only guides in the given guide style and leaves the remaining work to the
    student. Each message must end the way its ending field says. Vary the wording
    and avoid stock openers such as "Think about", "Not quite", "Take another look"
    or "Remember that"."""

    question: str = dspy.InputField()
    solution: str = dspy.InputField()
    student_mistake: str = dspy.InputField()
    reveal_style: str = dspy.InputField()
    reveal_ending: str = dspy.InputField()
    guide_style: str = dspy.InputField()
    guide_ending: str = dspy.InputField()
    student_turns: str = dspy.OutputField(
        desc="1-2 exchanges as 'Student: ...' and 'Tutor: ...' lines, starting and ending with the student"
    )
    reveal_response: str = dspy.OutputField(desc="tutor message that gives the answer away, 1-3 sentences")
    guide_response: str = dspy.OutputField(desc="tutor message that only guides, no answer, 1-3 sentences")


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def has_number(text, number):
    """Whole-number match, ignoring thousands separators: '1,200' matches 1200, '12' does not match 120."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)", text) is not None


def download():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    problems = []
    for i, line in enumerate(RAW.read_text().splitlines()):
        row = json.loads(line)
        steps, answer = row["answer"].split("####")
        problems.append({
            "problem_id": f"gsm8k_{i:04d}",
            "question": row["question"].strip(),
            "solution": re.sub(r"<<[^>]*>>", "", steps).strip(),
            "answer": answer.strip().replace(",", ""),
        })
    return problems


def without_leaks(problems):
    """Drop problems that appear in any MRBench split or in the MathDial test set.
    Matching on the start of the question also catches copies with small edits."""
    seen = [json.loads(line)["conversation"] for line in MRBENCH.open()]
    seen += [json.loads(line)["question"] for line in MATHDIAL_TEST.open()]
    blob = "\n".join(norm(text) for text in seen)
    return [p for p in problems if norm(p["question"])[:60] not in blob]


def read_stage(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def run_parallel(fn, items, threads, out_path):
    """Run fn over items, appending each result to out_path as it finishes.
    Failures are printed and skipped, so a rerun retries them."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failed = 0

    def task(item):
        nonlocal failed
        try:
            result = fn(item)
        except Exception as e:
            with lock:
                failed += 1
                print(f"  failed: {e}"[:300])
            return
        with lock, out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(task, items))
    return failed


def generate(problems, generators, threads):
    done = {g["problem_id"] for g in read_stage(GENERATED)}
    todo = [(i, p) for i, p in enumerate(problems) if p["problem_id"] not in done]
    print(f"generate: {len(problems) - len(todo)} done, {len(todo)} to go with {', '.join(generators)}")
    predictors = {}
    for name in generators:
        predictors[name] = dspy.Predict(RewriteGSM8K)
        predictors[name].set_lm(config.generator_lm(name))

    def one(item):
        i, p = item
        # styles depend on the problem only, so a larger --limit keeps earlier rows unchanged
        rng = random.Random(f"{SEED}-{p['problem_id']}")
        styles = {"student_mistake": rng.choice(STUDENT), "reveal_style": rng.choice(REVEAL),
                  "guide_style": rng.choice(GUIDE)}
        for side in ("reveal", "guide"):
            ending = rng.choice((QUESTION, STATEMENT))
            styles[f"{side}_ending"] = QUESTION if styles[f"{side}_style"] in ALWAYS_QUESTION else ending
        generator = generators[i % len(generators)]
        out = predictors[generator](question=p["question"], solution=p["solution"], **styles)
        turns = out.student_turns.strip().replace("\n", " \n ")
        return {**p, **styles, "generator": generator, "student_turns": turns,
                "conversation": OPENER.format(question=p["question"]) + turns,
                "reveal": out.reveal_response.strip(), "guide": out.guide_response.strip()}

    return run_parallel(one, todo, threads, GENERATED)


def end_on_student(g):
    """Cut the conversation after the student's last turn. Gemini often writes an
    exchange as student + tutor and stops there, so the next message would be a
    second tutor turn in a row. Turns can span several lines."""
    lines = g["student_turns"].split(" \n ")
    student = [i for i, line in enumerate(lines) if line.lstrip().startswith("Student:")]
    if not student:
        return g
    end = next((i for i in range(student[-1] + 1, len(lines)) if lines[i].lstrip().startswith("Tutor:")), len(lines))
    turns = " \n ".join(lines[:end])
    return {**g, "student_turns": turns, "conversation": OPENER.format(question=g["question"]) + turns}


def check(g):
    """Plain-code checks on one generated problem. Returns a reason to drop it, or None."""
    speakers = [line.lstrip().split(":")[0] for line in g["student_turns"].split(" \n ")
                if line.lstrip().startswith(("Student:", "Tutor:"))]
    if not speakers or speakers[-1] != "Student":
        return "conversation does not end with the student"
    if has_number(g["student_turns"], g["answer"]):
        return "answer in conversation"
    if has_number(g["guide"], g["answer"]):
        return "answer in guide"
    if g["reveal_style"] != NO_ANSWER_CHECK and not has_number(g["reveal"], g["answer"]):
        return "answer missing from reveal"
    short, long = sorted((len(g["reveal"]), len(g["guide"])))
    if not short or long > MAX_LENGTH_RATIO * short:
        return "length mismatch"
    return None


def rows_for(g):
    common = {"conversation": g["conversation"], "solution": g["solution"], "source": "gsm8k",
              "tutor": g["generator"], "problem_id": g["problem_id"], "student_mistake": g["student_mistake"]}
    yield {"id": f"{g['problem_id']}_reveal", "response": g["reveal"], "label": "yes",
           "style": g["reveal_style"], "ending": g["reveal_ending"], **common}
    yield {"id": f"{g['problem_id']}_guide", "response": g["guide"], "label": "no",
           "style": g["guide_style"], "ending": g["guide_ending"], **common}


def shortcut_audit(rows):
    """Surface cues that predict the label without understanding it. On MRBench
    test the 'no if it ends with ?' rule scores 0.60 balanced accuracy."""
    yes = [r["response"] for r in rows if r["label"] == "yes"]
    no = [r["response"] for r in rows if r["label"] == "no"]
    if not yes or not no:
        return
    q_yes = sum(t.rstrip().endswith("?") for t in yes) / len(yes)
    q_no = sum(t.rstrip().endswith("?") for t in no) / len(no)
    print(f"\nshortcut audit: ends with '?' yes {q_yes:.2f}, no {q_no:.2f}; "
          f"rule 'no if ?' balanced acc {(1 - q_yes + q_no) / 2:.2f} (MRBench test 0.60)")
    print(f"  median length yes {statistics.median(map(len, yes)):.0f}, no {statistics.median(map(len, no)):.0f} chars")
    opener = lambda t: " ".join(t.split()[:2])
    counts = Counter(opener(t) for t in yes + no)
    one_sided = [(o, n) for o, n in counts.most_common() if n >= 5
                 and max(sum(opener(t) == o for t in yes), sum(opener(t) == o for t in no)) / n >= 0.9]
    print("  openers that give away the label: " + (", ".join(f"'{o}' x{n}" for o, n in one_sided) or "none"))


def verify(rows, verifier, program, threads):
    key = {"verifier": verifier, "program": program}
    done = {v["id"]: v["verdict"] for v in read_stage(VERIFIED) if {k: v[k] for k in key} == key}
    todo = [r for r in rows if r["id"] not in done]
    print(f"verify: {len(rows) - len(todo)} done, {len(todo)} to go with {verifier} ({program or 'baseline'})")
    judge = make_judge(verifier, program)

    def one(row):
        pred = judge(conversation=row["conversation"], response=row["response"])
        return {"id": row["id"], **key, "verdict": getattr(pred, "reveals_answer", None)}

    failed = run_parallel(one, todo, threads, VERIFIED)
    done = {v["id"]: v["verdict"] for v in read_stage(VERIFIED) if {k: v[k] for k in key} == key}
    return done, failed


def to_kev(row, instructions):
    return {"state": {"conversation": row["conversation"], "tutor_response": row["response"]},
            "questions": {"reveals": {"type": "noul",
                                      "instructions": instructions + JevJudge.SUFFIX,
                                      "label": row["label"] == "yes"}}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200, help="number of GSM8K problems to use")
    p.add_argument("--generators", nargs="+", default=config.GENERATOR_MODELS,
                   help="judge names or model strings, problems are split between them round-robin")
    p.add_argument("--verifier", default=config.JUDGE_MODEL, help="judge that checks the labels")
    p.add_argument("--program", default="results/v5/program.json", help="program for the verifier, 'none' for baseline")
    p.add_argument("--no-verify", action="store_true", help="skip the judge check (plain-code checks only)")
    p.add_argument("--kev-program", help="program whose instructions go into the Kev files "
                                         "(default: the baseline RevealsAnswer instructions)")
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    args = p.parse_args()
    program = None if args.program == "none" else args.program

    problems = download()
    clean = without_leaks(problems)
    print(f"gsm8k: {len(problems)} problems, {len(problems) - len(clean)} dropped as in MRBench/MathDial test")
    random.Random(SEED).shuffle(clean)
    chosen = clean[: args.limit]
    ids = {p["problem_id"] for p in chosen}

    gen_failed = generate(chosen, args.generators, args.threads)
    generated = [end_on_student(g) for g in read_stage(GENERATED) if g["problem_id"] in ids]

    reasons = Counter(check(g) for g in generated)
    passed = [g for g in generated if check(g) is None]
    rows = [r for g in passed for r in rows_for(g)]

    ver_failed = 0
    if not args.no_verify:
        verdicts, ver_failed = verify(rows, args.verifier, program, args.threads)
        agree = {r["id"] for r in rows if verdicts.get(r["id"]) == r["label"]}
        disagree = Counter(r["label"] for r in rows if r["id"] in verdicts and r["id"] not in agree)
        pair = lambda g: {f"{g['problem_id']}_reveal", f"{g['problem_id']}_guide"}
        reasons["not verified (call failed)"] = sum(not pair(g) <= verdicts.keys() for g in passed)
        reasons["judge disagreed"] = sum(pair(g) <= verdicts.keys() and not pair(g) <= agree for g in passed)
        passed = [g for g in passed if pair(g) <= agree]

    # problems keep the shuffled order; the last VAL_SHARE of them is val
    order = {p["problem_id"]: i for i, p in enumerate(chosen)}
    passed.sort(key=lambda g: order[g["problem_id"]])
    n_val = round(len(passed) * VAL_SHARE)
    out = []
    for i, g in enumerate(passed):
        split = "val" if i >= len(passed) - n_val else "train"
        out += [{**r, "split": split, "label_by": "generator+checks" if args.no_verify else
                 f"generator+checks+{args.verifier}"} for r in rows_for(g)]

    with OUT.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    instructions = load_instructions(args.kev_program) if args.kev_program else RevealsAnswer.instructions
    KEV_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with (KEV_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in out:
                if r["split"] == split:
                    f.write(json.dumps(to_kev(r, instructions), ensure_ascii=False) + "\n")

    print(f"\n{len(chosen)} problems chosen, {len(generated)} generated, {len(passed)} kept")
    for reason, n in reasons.items():
        if reason:
            print(f"  dropped {n:5}  {reason}")
    if not args.no_verify:
        print(f"  judge said the opposite on {disagree['yes']} reveal and {disagree['no']} guide rows")
    if gen_failed or ver_failed:
        print(f"  {gen_failed} generation and {ver_failed} verification calls failed; rerun to retry them")
    for split in ("train", "val"):
        rows = [r for r in out if r["split"] == split]
        yes = sum(r["label"] == "yes" for r in rows)
        by_gen = Counter(r["tutor"] for r in rows)
        print(f"{split:5} {len(rows):5} rows, {yes} yes  " + ", ".join(f"{k}: {v}" for k, v in by_gen.items()))
    shortcut_audit(out)
    print(f"wrote {OUT}, {KEV_DIR}/train.jsonl, {KEV_DIR}/val.jsonl")


if __name__ == "__main__":
    main()
