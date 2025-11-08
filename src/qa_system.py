from transformers import pipeline
from pathlib import Path
import sys

# Load the paragraph from context.txt
context_path = Path(__file__).resolve().parents[1] / "data" / "context.txt"
context = context_path.read_text(encoding="utf-8").strip()

# Load a pretrained model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", framework="pt")



def ask_question(question):
    result = qa_pipeline(question=question, context=context)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    answer = ask_question(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['score']:.2f}")
