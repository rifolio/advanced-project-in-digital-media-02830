"""The judges (DSPy LLM judge, Jev), data loading, GEPA metric."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Literal

import dspy

import config

DATA = "data/mrbench.jsonl"


class RevealsAnswer(dspy.Signature):
    """You are auditing an AI tutor. Decide whether the tutor's latest response
    reveals the answer to the student's problem.

    A response reveals the answer if it states the final answer, gives the key
    intermediate result that makes the answer trivial, or fully works through
    the step the student is stuck on. Hints, guiding questions, pointing out
    where an error is, and asking the student to re-check their work do NOT
    reveal the answer."""

    conversation: str = dspy.InputField(desc="Dialogue so far between tutor and student")
    response: str = dspy.InputField(desc="The tutor's next message, to be judged")
    reveals_answer: Literal["yes", "no"] = dspy.OutputField()


class JevJudge(dspy.Module):
    """TypeSafe AI's Jev (not an LLM): one yes/no "Noul" question per response,
    same definition as the DSPy judge. Not optimisable with GEPA."""

    URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model):
        super().__init__()
        self.model = model.removeprefix("typesafe/")
        self.key = os.getenv("TYPESAFE_API_KEY")
        if not self.key:
            raise SystemExit("Jev needs TYPESAFE_API_KEY in .env")

    def forward(self, conversation, response):
        body = {
            "model": self.model,
            "state": {"conversation": conversation, "tutor_response": response},
            "questions": {"reveals": {"type": "noul", "instructions": RevealsAnswer.__doc__}},
        }
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        for attempt in range(config.NUM_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    p = json.load(r)["answers"]["reveals"]["noul"]
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 529) or attempt == config.NUM_RETRIES:
                    raise
                time.sleep(2**attempt)
        # ponytail: fixed 0.5 threshold, tune on val if Jev's probabilities are skewed
        return dspy.Prediction(reveals_answer="yes" if p >= 0.5 else "no", probability=p)


def make_judge(name=None, program_path=None):
    """Judge by name (see config.JUDGES) or raw model string."""
    model, _ = config.resolve(name)
    if model.startswith("typesafe/"):
        if program_path:
            raise SystemExit("Jev has no prompt program to load")
        return JevJudge(model)
    judge = dspy.ChainOfThought(RevealsAnswer)
    if program_path:
        judge.load(program_path)
    judge.set_lm(config.judge_lm(name))
    return judge


def load(split, limit=None):
    rows = [json.loads(line) for line in open(DATA)]
    rows = [r for r in rows if r["split"] == split][:limit]
    return [dspy.Example(**r).with_inputs("conversation", "response") for r in rows]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric: score plus feedback. The feedback may use the gold solution,
    which the judge itself never sees."""
    predicted = getattr(pred, "reveals_answer", None)
    if predicted == gold.label:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Wrong: predicted '{predicted}', human label is '{gold.label}'. "
            f"The correct solution to the problem is: {gold.solution}\n"
            "Compare the tutor response against it to see whether the answer was given away."
        ),
    )
