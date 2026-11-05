import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.generation.grounding_verifier import GroundingVerifier

QUESTIONS = [
    "Which backend framework does Project Orion use?",
    "Which programming language is used for Project Meridian's transaction service?",
]

retriever = Retriever()
generator = AnswerGenerator()
verifier = GroundingVerifier()

for question in QUESTIONS:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    contexts = retriever.retrieve(query=question, top_k=4)

    result = generator.generate(
        question=question,
        contexts=contexts,
    )

    print("\nANSWER GENERATOR RESULT:")
    print(result)

    if result["grounded"]:
        verification = verifier.verify(
            question=question,
            answer=result["answer"],
            contexts=contexts,
        )

        print("\nGROUNDING VERIFIER RESULT:")
        print(verification)
    else:
        print("\nVERIFIER SKIPPED: AnswerGenerator returned grounded=False")

print("\nDiagnostic completed.")
