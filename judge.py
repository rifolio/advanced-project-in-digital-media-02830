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


class JevInstructions(dspy.Signature):
    """Placeholder; the real seed is injected in JevJudge.__init__."""

    conversation: str = dspy.InputField(desc="Dialogue so far between tutor and student")
    response: str = dspy.InputField(desc="The tutor's next message, to be judged")
    reveals_answer: Literal["yes", "no"] = dspy.OutputField()


class JevJudge(dspy.Module):
    """TypeSafe AI's Jev binary classifier with a GEPA-editable instruction carrier."""

    URL = "https://api.typesafe.ai/v1/systemone"
    SUFFIX = "\n\nReturn only the binary reveal probability required by the Noul question."

    def __init__(self, model, seed_instructions=None, threshold=0.5):
        super().__init__()
        self.model = model.removeprefix("typesafe/")
        self.key = os.getenv("TYPESAFE_API_KEY")
        if not self.key:
            raise SystemExit("Jev needs TYPESAFE_API_KEY in .env")
        self.threshold = threshold
        seed = seed_instructions or RevealsAnswer.instructions
        self.predictor = dspy.Predict(JevInstructions.with_instructions(seed))

    def _call_jev(self, instructions, conversation, response):
        body = {
            "model": self.model,
            "state": {"conversation": conversation, "tutor_response": response},
            "questions": {
                "reveals": {
                    "type": "noul",
                    "instructions": instructions + self.SUFFIX,
                }
            },
        }
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        req.add_header("Authorization", "Bearer " + self.key)
        for attempt in range(config.NUM_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r)["answers"]["reveals"]["noul"]
            except urllib.error.HTTPError as e:
                if e.code not in (429, 529) or attempt == config.NUM_RETRIES:
                    raise
                time.sleep(2**attempt)
        # Threshold can be tuned on validation data without changing Jev's prompt.

    def forward(self, conversation, response):
        instructions = self.predictor.signature.instructions
        p = self._call_jev(instructions, conversation, response)
        pred = dspy.Prediction(
            reveals_answer="yes" if p >= self.threshold else "no",
            probability=p,
        )
        trace = dspy.settings.trace
        if trace is not None:
            trace.append(
                (self.predictor, {"conversation": conversation, "response": response}, pred)
            )
        return pred


def make_judge(name=None, program_path=None, threshold=0.5):
    """Judge by name (see config.JUDGES) or raw model string."""
    model, _ = config.resolve(name)
    if model.startswith("typesafe/"):
        seed = None
        if program_path:
            with open(program_path, encoding="utf-8-sig") as f:
                state = json.load(f)
            module_state = state.get("predictor") or state.get("predict")
            if not module_state:
                raise ValueError(f"No predictor instructions found in {program_path}")
            seed = module_state["signature"]["instructions"]
        return JevJudge(model, seed_instructions=seed, threshold=threshold)
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
