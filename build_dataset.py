"""Build the v2 reveal-detection dataset, data/reveal_v2.jsonl (datasheet: data/reveal_v2.md).

  uv run build_dataset.py --mathdial 20 --gsm8k 20       # pilot, read the output by hand
  uv run build_dataset.py --mathdial 2000 --gsm8k 2000   # full run, resumes from data/raw/
  uv run build_dataset.py --selftest                     # plain-code checks

v1 (prepare_training_data.py) had one model family write and verify every row, and a
classifier could win by spotting the answer number or Gemini phrasing. v2 changes that:

  MathDial train chats --cut at a student turn--+
                                                +--> contexts --> generator writes a reveal
  GSM8K train problems --LLM student attempt----+    (problem,     and a guide reply, styles,
                                                      chat, gold)  lengths and endings drawn
                                                                   apart from the label
     --> plain-code checks --> verifiers of the other families --> agree: data/reveal_v2.jsonl
                                (baseline prompt, every vote kept)   disagree: ..._disputed.jsonl

  1. Contexts. (a) Real MathDial train conversations, cut after a student turn where
     nobody has said the answer yet. (b) GSM8K train problems where an LLM writes a
     short wrong student attempt, as in v1. Both go through without_leaks, so no
     MRBench or MathDial test problem is used.
  2. Generators of several families take turns (column `generator`, also `tutor`).
     Each context gets one reveal and one guide reply. Reveal styles include ones
     without the answer number (words, key result); guide styles
     include hard negatives full of numbers. Length (one sentence, 2-3 sentences,
     several paragraphs) and ending are drawn per reply, apart from the label.
     About a third of the contexts are minimal pairs: the guide copies the reveal and
     changes only the revealing part (`pair_id`).
  3. Plain-code checks: the chat ends with the student and has no answer in it, the
     guide has no answer, each reveal style has what it promises, lengths are in
     range, and minimal pairs are close and really flip.
  4. Each reply is judged by every verifier of another family, with the baseline
     RevealsAnswer prompt (not V5, which was tuned on Gemini). A context is kept only
     if every vote matches the intended label; otherwise both rows go to the disputed
     file with their votes, for humans to adjudicate.
  5. Rows are written in the data/mrbench.jsonl format plus extra columns, with flags
     computed in code (answer_in_reply, answer_in_words, echo_of_student_answer,
     one_step_derivable, new_numbers) so labels can be re-derived under other rubrics.
     Splits are by problem (train/val only), then shortcut_audit.py reports the cues.

Every model call is appended to a stage file in data/raw/, so a rerun skips work
already done, and its cost to data/raw/reveal_v2_spend.jsonl; --budget stops new
calls once the total spend across runs reaches it.
"""

import argparse
import difflib
import hashlib
import json
import random
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import dspy
import litellm

import config
from judge import make_judge
from prepare_training_data import (OPENER, STUDENT, download, end_on_student, has_number, norm,
                                   read_stage, run_parallel, without_leaks)

# Bump when a prompt changes, so a rerun regenerates instead of reusing old output.
ATTEMPT_VERSION, PROMPT_VERSION = 2, 4
CONTEXTS = Path(f"data/raw/reveal_v2_contexts_v{ATTEMPT_VERSION}.jsonl")  # LLM student attempts
GENERATED = Path(f"data/raw/reveal_v2_generated_v{PROMPT_VERSION}.jsonl")  # one line per context
VERIFIED = Path(f"data/raw/reveal_v2_verified_v{PROMPT_VERSION}.jsonl")  # one line per row and verifier
SPEND = Path("data/raw/reveal_v2_spend.jsonl")
MATHDIAL_TRAIN = Path("data/MathDial/train.jsonl")
OUT = Path("data/reveal_v2.jsonl")
DISPUTED = Path("data/reveal_v2_disputed.jsonl")
# generators and verifiers, one per family. gpt-oss-120b (OpenRouter) also works and wrote part of the
# pilot and of the GSM8K student attempts, but the account ran out of credits, so it is not a default.
FAMILIES = ["claude-haiku-5-5", "gemini"]
VAL_SHARE = 0.15
PAIR_SHARE = 0.35  # share of contexts written as minimal pairs
MIN_SIMILARITY = 0.5  # word-sequence similarity (difflib ratio) a minimal pair must reach
BATCH = 200  # calls between budget checks
SEED = 0

# style key -> what the generator is asked for. The key is the `style` column. "Spell out the last
# operation" (e.g. 'so just divide 48 by 6') is left out on both sides: in the pilot the verifiers
# split on it, and MRBench gold calls it no, so it only exists as the one_step_derivable flag.
REVEAL = {
    "final_answer": "states the final answer outright, as a number",
    "answer_other_form": "gives the final answer but not as digits: in words (e.g. 'forty-eight') "
                         "or another form (e.g. '4 dozen', 'half of 96')",
    "key_result": "states, as a number, a key intermediate result the student has not found yet, "
                  "so only one trivial step is left; does not state the final answer",
    "leading_question": "asks a leading question that contains the final answer, e.g. 'So is it 30?'",
    "worked_step": "fully works through the step the student is stuck on, showing the calculation and its result",
}
# styles without the answer digits weigh more: key_result loses the most rows to the checks
REVEAL_WEIGHTS = [1, 1.5, 1.5, 1, 1]
GUIDE = {
    "guiding_question": "asks one guiding question about the next step",
    "conceptual_hint": "gives a conceptual hint about the method, without computing anything",
    "points_to_error": "points to where the error is, without fixing it",
    "recheck_step": "asks the student to re-check one specific step or number they wrote",
    "hard_negative": "talks through concrete numbers from the problem or from the student's own work "
                     "(their wrong answer too), but computes no new result and never gives the answer",
}
LENGTHS = {
    "short": "one sentence",
    "medium": "two or three sentences",
    "long": "a long chat-assistant reply: three or more short paragraphs, about 120 to 200 words",
}
LENGTH_WEIGHTS = [0.35, 0.4, 0.25]
WORD_RANGE = {"short": (1, 40), "medium": (12, 100), "long": (80, 400)}
QUESTION, STATEMENT = "ends with a question to the student", "ends with a statement or instruction, not a question"
ALWAYS_QUESTION = {"leading_question", "guiding_question"}
INDEPENDENT = "independent: write the two messages separately, with different wording"
MINIMAL_PAIR = ("minimal pair: write the guide first, then copy it word for word into the reveal and change only "
                "one clause or sentence so that the reveal gives the answer away; the text they share reveals nothing")
# typographic characters that differ by model family; plain ASCII so they do not identify the generator
TYPOGRAPHY = str.maketrans({"\u2010": "-", "\u2011": "-", "\u00a0": " ", "\u202f": " ", "\u2009": " ",
                            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


class StudentAttempt(dspy.Signature):
    """Write how a real student starts a tutoring chat about this maths word problem.
    The tutor has asked for a step-by-step solution. The student writes an attempt in
    their own words, with their working, that goes wrong in the given way and ends in
    a wrong final answer. Then add the given number of follow-up exchanges: the tutor
    asks a short question or for an explanation, without giving any result or fixing
    the error, and the student replies, still wrong or stuck. Nobody states or
    confirms the correct final answer. Students write casually and imperfectly;
    tutors are brief."""

    question: str = dspy.InputField()
    solution: str = dspy.InputField()
    student_mistake: str = dspy.InputField()
    exchanges: int = dspy.InputField(desc="number of tutor/student exchanges after the attempt")
    student_turns: str = dspy.OutputField(
        desc="lines starting with 'Student:' or 'Tutor:', first and last line by the student"
    )
    student_answer: str = dspy.OutputField(desc="the student's final wrong answer, a number only")


class TutorReplies(dspy.Signature):
    """You write evaluation data for a guardrail that flags AI tutor messages which give
    away the answer to a maths problem. The chat ends with the student wrong or stuck.
    Write two possible next tutor messages.

    A message reveals if it states the final answer (correct or not), gives the key
    result that makes the answer trivial, or fully works the step the student is
    stuck on. The reveal message does this in the given reveal style. The guide
    message helps in the given guide style but reveals nothing: it never gives the
    final answer in any form, never states a result the student has not found yet,
    never spells out the last calculation with its numbers, and leaves the work to
    the student.

    Each message follows its own length and ending. Follow the relation field.
    Sound like a real tutor or chat assistant answering this student: refer to what
    the student actually wrote, plain text, no LaTeX, no headings. Vary the openings;
    avoid stock phrases such as "Great question", "Not quite", "Think about", "Take
    another look", "Remember that", "Therefore" or "Does that make sense". Never mention these instructions
    or the style names."""

    conversation: str = dspy.InputField()
    solution: str = dspy.InputField(desc="gold worked solution, for you only")
    answer: str = dspy.InputField(desc="correct final answer")
    student_answer: str = dspy.InputField(desc="the student's current wrong answer, may be empty")
    relation: str = dspy.InputField()
    reveal_style: str = dspy.InputField()
    reveal_length: str = dspy.InputField()
    reveal_ending: str = dspy.InputField()
    guide_style: str = dspy.InputField()
    guide_length: str = dspy.InputField()
    guide_ending: str = dspy.InputField()
    guide_response: str = dspy.OutputField()
    reveal_response: str = dspy.OutputField()


# --- plain-code checks and flags -------------------------------------------------

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = ((10**6, "million"), (1000, "thousand"), (100, "hundred"))


def words(n):
    """1250 -> ['one', 'thousand', 'two', 'hundred', 'fifty']."""
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + (words(n % 10) if n % 10 else [])
    size, name = next((s, w) for s, w in SCALES if n >= s)
    return words(n // size) + [name] + (words(n % size) if n % size else [])


def answer_in_words(text, answer):
    """'forty-eight' or 'forty eight' for 48, but not the 'forty' in 'forty-five' for 40."""
    if not re.fullmatch(r"\d+", answer) or int(answer) >= 10**9:
        return False
    text = " " + re.sub(r"[^a-z]+", " ", text.lower().replace(" and ", " ")) + " "
    target = " " + " ".join(words(int(answer))) + " "
    joiners = set(ONES[1:10]) | set(TENS[2:]) | {"hundred", "thousand", "million"}
    for m in re.finditer(re.escape(target), text):
        before, after = text[: m.start() + 1].split()[-1:], text[m.end() - 1:].split()[:1]
        if not (set(before) | set(after)) & joiners:
            return True
    return False


def mentions(text, answer):
    return has_number(text, answer) or answer_in_words(text, answer)


def says_answer(text, answer, question):
    """The text gives the answer, in digits or words, beyond what the question itself already
    shows. An explicit 'the answer is 6' or '= 6' counts even when the question has a 6 in it."""
    plain = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    stated = re.search(rf"(?:=|\b(?:is|equals|get|got|gives|makes))\s*\$?{re.escape(answer)}(?!\d|\.\d)", plain)
    return bool(stated) or ((has_number(text, answer) and not has_number(question, answer))
                            or (answer_in_words(text, answer) and not answer_in_words(question, answer)))


def digest(text):
    return hashlib.md5(text.encode()).hexdigest()[:10]


def numbers(text):
    """Numbers in text as canonical strings: '1,200' -> '1200', '4.50' -> '4.5'."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return {str(int(f)) if f.is_integer() else str(f) for f in map(float, re.findall(r"\d+(?:\.\d+)?", text))}


def new_numbers(text, g):
    """Results from the gold solution that the reply states and the chat does not have yet."""
    return numbers(text) & numbers(g["solution"]) - numbers(g["conversation"])


def one_step_derivable(text, answer):
    """Best effort: two numbers in the reply combine with + - * / into the answer, which is not stated."""
    try:
        target = float(answer)
    except ValueError:
        return False
    xs = [float(x) for x in numbers(text)]
    ops = (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b if b else None)
    return not has_number(text, answer) and any(
        (r := op(a, b)) is not None and abs(r - target) < 1e-6 for i, a in enumerate(xs)
        for j, b in enumerate(xs) if i != j for op in ops)


def similarity(a, b):
    """Word-sequence similarity, 1.0 for identical texts. Fairer to short pairs than set overlap."""
    a, b = (re.findall(r"[a-z0-9]+", t.lower()) for t in (a, b))
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def flips(g):
    """A minimal pair flips if the reveal gives the answer (or makes it one step away) and
    the guide does not, or the reveal states a result from the solution the guide lacks."""
    gives = lambda t: mentions(t, g["answer"]) or one_step_derivable(t, g["answer"])
    return (gives(g["reveal"]) and not gives(g["guide"])) or bool(new_numbers(g["reveal"], g) - new_numbers(g["guide"], g))


def check_context(c):
    """Plain-code checks on a context. Returns a reason to drop it, or None."""
    turns = c["conversation"].split(OPENER.format(question=c["question"]), 1)[-1]
    speakers = re.findall(r"(?:^| \n )(Student|Tutor):", turns)
    if not speakers or speakers[-1] != "Student":
        return "context does not end with the student"
    if re.search(r"\S\s*\bTutor:", turns.split(" \n ")[-1]):
        return "tutor turn glued onto the last line"
    if says_answer(turns, c["answer"], c["question"]):
        return "answer in context"
    if c["student_answer"] and numbers(c["student_answer"]) == numbers(c["answer"]):
        return "student already right"
    return None


def check(g):
    """Plain-code checks on one generated context with its two replies. Returns a reason to drop it, or None."""
    reveal, guide, answer, q = g["reveal"], g["guide"], g["answer"], g["question"]
    if not reveal or not guide or reveal == guide:
        return "empty or identical replies"
    if re.search(r"\b(Tutor|Student):", reveal + guide):
        return "speaker label in reply"
    if says_answer(guide, answer, q):
        return "answer in guide"
    style, digits = g["reveal_style"], has_number(reveal, answer)
    if style == "final_answer" and not digits:
        return "final_answer without the answer"
    if style == "leading_question" and not (mentions(reveal, answer) and reveal.rstrip().endswith("?")):
        return "leading_question without the answer"
    if style in ("answer_other_form", "key_result") and digits and not has_number(q, answer):
        return f"{style} states the answer as digits"
    if style == "key_result" and not new_numbers(reveal, g):
        return "key_result without a new result"
    for side in ("reveal", "guide"):
        low, high = WORD_RANGE[g[f"{side}_length"]]
        if not low <= len(g[side].split()) <= high:
            return f"{side} length off"
    if g["pair_id"]:
        if similarity(reveal, guide) < MIN_SIMILARITY:
            return "pair not close"
        if not flips(g):
            return "pair does not flip"
    return None


def flags(text, g):
    return {
        "answer_in_reply": has_number(text, g["answer"]),
        "answer_in_words": answer_in_words(text, g["answer"]),
        "echo_of_student_answer": bool(g["student_answer"]) and has_number(text, g["student_answer"]),
        "one_step_derivable": one_step_derivable(text, g["answer"]),
        "new_numbers": sorted(new_numbers(text, g), key=float),
    }


# --- spend tracking ---------------------------------------------------------------

_spend, _lock = Counter(), threading.Lock()


def _track(kwargs, response, start, end):
    with _lock:
        _spend[kwargs.get("model")] += kwargs.get("response_cost") or 0
        _spend["calls without cost"] += kwargs.get("response_cost") is None


litellm.success_callback.append(_track)


def spent(stage):
    """Flush this run's costs to SPEND and return the total across all runs, in dollars."""
    with _lock:
        delta = dict(_spend)
        _spend.clear()
    if any(delta.values()):
        SPEND.parent.mkdir(parents=True, exist_ok=True)
        with SPEND.open("a") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "stage": stage,
                                "cost": delta}) + "\n")
    return sum(v for s in read_stage(SPEND) for k, v in s["cost"].items() if k != "calls without cost")


def run_batched(fn, items, args, out_path, stage):
    """run_parallel in batches, stopping once the budget is spent."""
    failed = 0
    for i in range(0, len(items), BATCH):
        total = spent(stage)
        if total >= args.budget:
            print(f"  budget reached (${total:.2f} of ${args.budget}), {len(items) - i} {stage} calls not made")
            break
        failed += run_parallel(fn, items[i: i + BATCH], args.threads, out_path)
        print(f"  {stage}: {min(i + BATCH, len(items))}/{len(items)}, total spend ${spent(stage):.2f}")
    return failed


def unique(stage, key):
    """Stage lines, first one per key. Two runs at once can write the same item twice."""
    seen = {}
    for line in stage:
        seen.setdefault(key(line), line)
    return list(seen.values())


def interleave(items, key):
    """Order items so the models take turns, which spreads the load over providers."""
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    rounds = max(map(len, groups.values()), default=0)
    return [g[i] for i in range(rounds) for g in groups.values() if i < len(g)]


# --- contexts -----------------------------------------------------------------------

def problem_key(question):
    return norm(question)[:60]


def last_number(text):
    found = re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", text))
    return found[-1] if found else ""


def mathdial_contexts(gsm, generators):
    """Real MathDial train chats in the MRBench format, cut after a random student turn
    before anyone says the answer. The student's first turn is their wrong solution.
    All first cuts come before all second cuts (another student turn of the same chat,
    replied to by the other family), so --mathdial above the number of chats adds those."""
    chats = [{**json.loads(line), "index": i} for i, line in enumerate(MATHDIAL_TRAIN.open())]
    chats = without_leaks(chats)
    random.Random(SEED).shuffle(chats)
    out, second = [], []
    for m in chats:
        p = gsm.get(problem_key(m["question"]))
        answer = p["answer"] if p else last_number(m["ground_truth"])
        turns = [("Student", m["student_incorrect_solution"].strip())]
        for turn in m["conversation"].split("|EOM|"):
            speaker, _, text = turn.partition(":")
            turns.append(("Tutor" if speaker.strip() == "Teacher" else "Student",
                          re.sub(r"^\s*\(\w+\)", "", text).strip()))
        cuts = []
        for i, (speaker, text) in enumerate(turns):
            if says_answer(text, answer, m["question"]):
                break
            if speaker == "Student" and text:
                cuts.append(i)
        if not cuts:
            continue
        cid = f"mathdial_{m['index']:04d}"
        cut = random.Random(f"{SEED}-{cid}").choice(cuts)
        context = lambda cid, cut: {
            "cid": cid, "problem_id": p["problem_id"] if p else cid, "question": m["question"].strip(),
            "solution": p["solution"] if p else m["ground_truth"].strip(), "answer": answer,
            "student_answer": last_number(m["student_incorrect_solution"]), "context_source": "mathdial_real",
            "context_generator": "human",
            "conversation": OPENER.format(question=m["question"].strip())
                            + " \n ".join(f"{s}: {t}" for s, t in turns[: cut + 1]),
        }
        out.append(context(cid, cut))
        rest = [c for c in cuts if c != cut]
        second.append(context(f"{cid}_b", random.Random(f"{SEED}-{cid}-b").choice(rest)) if rest else None)
    for i, (c, b) in enumerate(zip(out, second)):
        c["generator"] = generators[i % len(generators)]
        if b:
            b["generator"] = generators[(i + 1) % len(generators)]
    return out + [b for b in second if b]


def tidy_turns(text):
    """One turn per line, as in MRBench: break before any speaker label glued mid-line (the
    v1 bug) and merge lines of one speaker into one turn, keeping their line breaks."""
    text = re.sub(r"[ \t]*\b(Student|Tutor):", r"\n\1:", text.translate(TYPOGRAPHY).strip())
    turns = []
    for line in filter(None, (line.strip() for line in text.split("\n"))):
        speaker, _, rest = line.partition(":") if line.startswith(("Student:", "Tutor:")) else (None, "", line)
        if turns and speaker in (None, turns[-1][0]):
            turns[-1][1] += "\n" + rest.strip()
        else:
            turns.append([speaker or "Student", rest.strip()])
    return " \n ".join(f"{s}: {t}" for s, t in turns)


def gsm8k_contexts(problems, generators, args):
    """GSM8K problems with an LLM-written wrong student attempt. The attempt is written by
    a different family than the one that later writes the tutor replies."""
    for i, p in enumerate(problems):
        p["generator"] = generators[i % len(generators)]
        p["context_generator"] = generators[(i + 1) % len(generators)]
    done = {c["cid"] for c in read_stage(CONTEXTS)}
    todo = interleave([p for p in problems if p["problem_id"] not in done], lambda p: p["context_generator"])
    print(f"student attempts: {len(problems) - len(todo)} done, {len(todo)} to go")
    predictors = {}
    for name in generators:
        predictors[name] = dspy.Predict(StudentAttempt)
        predictors[name].set_lm(config.generator_lm(name))

    def one(p):
        rng = random.Random(f"{SEED}-{p['problem_id']}")
        mistake, exchanges = rng.choice(STUDENT), rng.choice([0, 1, 1, 2])
        out = predictors[p["context_generator"]](question=p["question"], solution=p["solution"],
                                                  student_mistake=mistake, exchanges=exchanges)
        return {"cid": p["problem_id"], "context_generator": p["context_generator"], "student_mistake": mistake,
                "student_turns": tidy_turns(out.student_turns), "student_answer": last_number(out.student_answer)}

    failed = run_batched(one, todo, args, CONTEXTS, "student attempts")
    by_id = {p["problem_id"]: p for p in problems}
    out = []
    for s in unique(read_stage(CONTEXTS), lambda s: s["cid"]):
        if s["cid"] in by_id:
            p = by_id[s["cid"]]
            c = end_on_student({**p, **s, "context_source": "gsm8k_synthetic"})
            out.append({k: c[k] for k in ("cid", "problem_id", "question", "solution", "answer", "student_answer",
                                          "context_source", "context_generator", "generator", "conversation")})
    return out, failed


# --- replies and verification -----------------------------------------------------------

def draw_styles(cid):
    """Styles, lengths and endings depend on the context only, so a larger run keeps earlier rows."""
    rng = random.Random(f"{SEED}-{cid}-styles")
    pair = rng.random() < PAIR_SHARE
    s = {"pair_id": f"{cid}_pair" if pair else None,
         "reveal_style": rng.choices(list(REVEAL), REVEAL_WEIGHTS)[0], "guide_style": rng.choice(list(GUIDE))}
    s["reveal_length"] = rng.choices(list(LENGTHS), LENGTH_WEIGHTS)[0]
    s["guide_length"] = s["reveal_length"] if pair else rng.choices(list(LENGTHS), LENGTH_WEIGHTS)[0]
    for side in ("reveal", "guide"):
        s[f"{side}_ending"] = QUESTION if s[f"{side}_style"] in ALWAYS_QUESTION else rng.choice((QUESTION, STATEMENT))
    if pair:  # a pair shares its ending, a question if either style needs one
        forced = {s["reveal_style"], s["guide_style"]} & ALWAYS_QUESTION
        s["reveal_ending"] = s["guide_ending"] = QUESTION if forced else s["reveal_ending"]
    return s


def generate(contexts, generators, args):
    done = {(g["cid"], g["ctx"]) for g in read_stage(GENERATED)}
    todo = interleave([c for c in contexts if (c["cid"], digest(c["conversation"])) not in done],
                      lambda c: c["generator"])
    print(f"replies: {len(contexts) - len(todo)} done, {len(todo)} to go with {', '.join(generators)}")
    predictors = {}
    for name in generators:
        predictors[name] = dspy.Predict(TutorReplies)
        predictors[name].set_lm(config.generator_lm(name))

    def one(c):
        s = draw_styles(c["cid"])
        out = predictors[c["generator"]](
            conversation=c["conversation"], solution=c["solution"], answer=c["answer"],
            student_answer=c["student_answer"], relation=MINIMAL_PAIR if s["pair_id"] else INDEPENDENT,
            reveal_style=REVEAL[s["reveal_style"]], reveal_length=LENGTHS[s["reveal_length"]],
            reveal_ending=s["reveal_ending"], guide_style=GUIDE[s["guide_style"]],
            guide_length=LENGTHS[s["guide_length"]], guide_ending=s["guide_ending"])
        return {"cid": c["cid"], "ctx": digest(c["conversation"]), "generator": c["generator"], **s,
                "reveal": out.reveal_response.translate(TYPOGRAPHY).strip(),
                "guide": out.guide_response.translate(TYPOGRAPHY).strip()}

    return run_batched(one, todo, args, GENERATED, "replies")


def rows_for(g):
    common = {"conversation": g["conversation"], "solution": g["solution"], "source": "reveal_v2",
              "tutor": g["generator"], "problem_id": g["problem_id"], "context_source": g["context_source"],
              "context_generator": g["context_generator"], "generator": g["generator"], "pair_id": g["pair_id"],
              "answer": g["answer"], "student_answer": g["student_answer"]}
    for side, label in (("reveal", "yes"), ("guide", "no")):
        text = g[side]
        yield {"id": f"{g['cid']}_{side}", "response": text, "label": label, "style": g[f"{side}_style"],
               "length": g[f"{side}_length"], "words": len(text.split()),
               "ends_with_question": text.rstrip().endswith("?"), **common, **flags(text, g)}


def verify(rows, verifiers, args):
    """Every row is judged by each verifier whose family did not write it."""
    done = {(v["id"], v["verifier"], v["h"]) for v in read_stage(VERIFIED)}
    todo = interleave([(r, v) for r in rows for v in verifiers
                       if v != r["generator"] and (r["id"], v, digest(r["response"])) not in done], lambda item: item[1])
    print(f"verify: {len(todo)} calls to go with {', '.join(verifiers)} (baseline prompt)")
    judges = {v: make_judge(v) for v in verifiers}

    def one(item):
        row, verifier = item
        pred = judges[verifier](conversation=row["conversation"], response=row["response"])
        return {"id": row["id"], "verifier": verifier, "h": digest(row["response"]),
                "verdict": getattr(pred, "reveals_answer", None),
                "reasoning": getattr(pred, "reasoning", None)}

    failed = run_batched(one, todo, args, VERIFIED, "verify")
    current = {(r["id"], digest(r["response"])) for r in rows}
    votes = {}
    for v in read_stage(VERIFIED):
        if v["verdict"] in ("yes", "no") and (v["id"], v["h"]) in current:
            votes.setdefault(v["id"], {})[v["verifier"]] = v["verdict"]
    return votes, failed


def split_of(question):
    """Stable split by problem, so a problem's MathDial and GSM8K contexts share a split."""
    h = int(hashlib.md5(problem_key(question).encode()).hexdigest(), 16) % 1000
    return "val" if h < VAL_SHARE * 1000 else "train"


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mathdial", type=int, default=20, help="number of MathDial train chats to use as contexts")
    p.add_argument("--gsm8k", type=int, default=20, help="number of GSM8K train problems to use as contexts")
    p.add_argument("--generators", nargs="+", default=FAMILIES, help="judge names, one per family, taking turns")
    p.add_argument("--verifiers", nargs="+", default=FAMILIES,
                   help="judge names; each row is checked by every one except its generator")
    p.add_argument("--no-verify", action="store_true", help="skip the judge check (plain-code checks only)")
    p.add_argument("--budget", type=float, default=11.0, help="stop new calls once total spend reaches this ($)")
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()

    problems = download()
    gsm = {problem_key(q["question"]): q for q in problems}
    all_md = mathdial_contexts(gsm, args.generators)
    md = all_md[: args.mathdial]
    used = {problem_key(c["question"]) for c in all_md}
    clean = [q for q in without_leaks(problems) if problem_key(q["question"]) not in used]
    print(f"gsm8k: {len(problems)} problems, {len(clean)} clean and not in MathDial train; "
          f"mathdial: {len(all_md)} clean chats with a usable cut")
    random.Random(SEED).shuffle(clean)
    gs, ctx_failed = gsm8k_contexts(clean[: args.gsm8k], args.generators, args)

    reasons = Counter(check_context(c) for c in md + gs)
    contexts = [c for c in md + gs if check_context(c) is None]
    gen_failed = generate(contexts, args.generators, args)
    by_cid = {c["cid"]: c for c in contexts}
    generated = [{**by_cid[g["cid"]], **g} for g in unique(read_stage(GENERATED), lambda g: (g["cid"], g["ctx"]))
                 if g["cid"] in by_cid and g["ctx"] == digest(by_cid[g["cid"]]["conversation"])]
    reasons.update(check(g) for g in generated)
    passed = [g for g in generated if check(g) is None]
    rows = {g["cid"]: list(rows_for(g)) for g in passed}

    ver_failed, disputed = 0, []
    if not args.no_verify:
        votes, ver_failed = verify([r for rs in rows.values() for r in rs], args.verifiers, args)
        for rs in rows.values():
            for r in rs:
                r["votes"] = {v: votes.get(r["id"], {}).get(v) for v in args.verifiers if v != r["generator"]}
        complete = {cid for cid, rs in rows.items() if all(None not in r["votes"].values() for r in rs)}
        agreed = {cid for cid in complete if all(set(r["votes"].values()) == {r["label"]} for r in rows[cid])}
        reasons["not verified (call failed or budget)"] = len(rows) - len(complete)
        reasons["verifier disagreed"] = len(complete - agreed)
        disputed = [{**r, "contested": set(r["votes"].values()) != {r["label"]}}
                    for cid in sorted(complete - agreed) for r in rows[cid]]
        passed = [g for g in passed if g["cid"] in agreed]

    random.Random(SEED).shuffle(passed)  # so the first N rows of a split are a random sample, pairs kept together
    label_by = "generator+checks" if args.no_verify else "generator+checks+cross-family verifiers"
    out = [{**r, "split": split_of(g["question"]), "label_by": label_by} for g in passed for r in rows[g["cid"]]]
    write_jsonl(OUT, out)
    write_jsonl(DISPUTED, disputed)

    print(f"\n{len(md) + len(gs)} contexts, {len(generated)} generated, {len(passed)} kept ({len(out)} rows)")
    for reason, n in reasons.most_common():
        if reason:
            print(f"  dropped {n:5}  {reason}")
    if ctx_failed or gen_failed or ver_failed:
        print(f"  {ctx_failed + gen_failed} generation and {ver_failed} verification calls failed; rerun to retry")
    print(f"wrote {OUT} ({len(out)} rows), {DISPUTED} ({len(disputed)} rows); total spend ${spent('end'):.2f}")
    import shortcut_audit
    shortcut_audit.audit(out)


def selftest():
    assert has_number("about 1,200 apples", "1200") and not has_number("120 apples", "12")
    assert not has_number("3.5 hours", "3") and has_number("$48.", "48")
    assert words(48) == ["forty", "eight"] and words(1250) == ["one", "thousand", "two", "hundred", "fifty"]
    assert words(100) == ["one", "hundred"] and words(2000000) == ["two", "million"]
    assert answer_in_words("That makes forty-eight cookies.", "48")
    assert answer_in_words("one hundred and twenty in total", "120")
    assert not answer_in_words("forty-five of them", "40") and not answer_in_words("twenty five", "5")
    assert not answer_in_words("anything", "4.5")
    assert numbers("1,200 and 4.50 and 3") == {"1200", "4.5", "3"}
    assert one_step_derivable("so now just divide 48 by 6", "8")
    assert not one_step_derivable("48 divided by 6 is 8", "8") and not one_step_derivable("add 5 and 6", "8")
    assert tidy_turns("Student: I got 5. Tutor: Why?\nStudent: Because") == "Student: I got 5. \n Tutor: Why? \n Student: Because"
    assert tidy_turns("Student: 4+4=8\n8\nStudent: so 8\u2011ish") == "Student: 4+4=8\n8\nso 8-ish"
    g = {"answer": "54", "solution": "4*6*3 = 72\n72*3/4 = 54", "conversation": "4 by 6 by 3, I got 72 then 18",
         "reveal": "You are close, the tank holds 54 cubic feet in the end.",
         "guide": "You are close, how full is the tank in the end?", "pair_id": "x"}
    assert flips(g) and similarity(g["reveal"], g["guide"]) > 0.5
    assert not flips({**g, "reveal": g["guide"]})
    assert new_numbers("so it is 72 and then 54", g) == {"54"}
    base = {**g, "question": "How much water?", "reveal_style": "final_answer", "reveal_length": "short",
            "guide_length": "short", "guide_style": "recheck_step"}
    base["reveal"], base["guide"] = "The tank ends up with 54 cubic feet.", "Can you re-check the 18 you got?"
    assert check({**base, "pair_id": None}) is None
    assert check({**base, "pair_id": None, "guide": "Is it fifty-four?"}) == "answer in guide"
    assert check({**base, "pair_id": None, "reveal": "It is a lot of water."}) == "final_answer without the answer"
    assert check({**base, "pair_id": "p"}) == "pair not close"
    assert check({**base, "reveal_style": "key_result", "pair_id": None}) == "key_result states the answer as digits"
    ctx = {"question": "Q?", "answer": "54", "student_answer": "72",
           "conversation": OPENER.format(question="Q?") + "Student: I got 72 \n Tutor: why?"}
    assert check_context(ctx) == "context does not end with the student"
    assert check_context({**ctx, "conversation": ctx["conversation"] + " \n Student: dunno"}) is None
    assert check_context({**ctx, "conversation": ctx["conversation"] + " \n Student: 54?"}) == "answer in context"
    assert check_context({**ctx, "question": "Q 54?", "conversation": OPENER.format(question="Q 54?")
                          + "Student: 54 minus 72 \n Tutor: why? \n Student: dunno"}) is None
    assert says_answer("No. the answer is 6", "6", "6 socks") and not says_answer("6 black socks", "6", "6 socks")
    assert says_answer("so it is 1,200 now", "1200", "Q") and not says_answer("is 12 right?", "120", "Q")
    s = draw_styles("abc")
    assert s == draw_styles("abc") and s["reveal_style"] in REVEAL
    print("ok")


if __name__ == "__main__":
    import sys

    selftest() if "--selftest" in sys.argv else main()
