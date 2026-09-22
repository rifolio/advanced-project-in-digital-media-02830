import dspy
from dotenv import load_dotenv

load_dotenv()


def main():
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    qa = dspy.Predict("question -> answer")
    print(qa(question="Say hi in one word.").answer)


if __name__ == "__main__":
    main()
