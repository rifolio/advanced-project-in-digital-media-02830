"""The LLM-as-a-judge: signature, program, data loading, GEPA metric."""

import json
from typing import Literal

import dspy

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


def make_judge(program_path=None):
    judge = dspy.ChainOfThought(RevealsAnswer)
    if program_path:
        judge.load(program_path)
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
