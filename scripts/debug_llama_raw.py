import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.generation.grounding_verifier import GroundingVerifier
from app.generation.llm_client import LLMClient


class DebugLLM:
    def __init__(self):
        self.client = LLMClient(model="llama3.2:3b")

    def generate(self, messages, temperature=0.0, max_tokens=1024):
        response = self.client.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print("\nRAW LLAMA RESPONSE:\n")
        print(response)
        return response


question = (
    "Which programming language is used for Project Meridian's "
    "transaction service?"
)

retriever = Retriever()
generator = AnswerGenerator()
verifier = GroundingVerifier(llm_client=DebugLLM())

contexts = retriever.retrieve(question, top_k=4)
result = generator.generate(question, contexts)

print("GENERATED ANSWER:", result["answer"])

verification = verifier.verify(
    question=question,
    answer=result["answer"],
    contexts=contexts,
)

print("\nFINAL VERIFICATION:")
print(verification)
