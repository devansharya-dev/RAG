import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.generation.grounding_verifier import GroundingVerifier
from app.generation.llm_client import LLMClient

questions = [
    "Which backend framework does Project Orion use?",
    "Which programming language is used for Project Meridian's transaction service?",
]

retriever = Retriever()
generator = AnswerGenerator()

verifier = GroundingVerifier(
    llm_client=LLMClient(model="llama3.2:3b")
)

for question in questions:
    print("\n" + "=" * 65)
    print("QUESTION:", question)

    contexts = retriever.retrieve(question, top_k=4)
    result = generator.generate(question, contexts)

    print("ANSWER:", result["answer"])

    verification = verifier.verify(
        question=question,
        answer=result["answer"],
        contexts=contexts,
    )

    print("VERIFICATION:", verification)

