import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.retriever import Retriever
from app.generation.prompt_builder import PromptBuilder
from app.generation.llm_client import LLMClient

QUESTIONS = [
    "Which backend framework does Project Orion use?",
    "Which programming language is used for Project Meridian's transaction service?",
]

retriever = Retriever()
prompt_builder = PromptBuilder()
llm = LLMClient()

for question in QUESTIONS:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    contexts = retriever.retrieve(query=question, top_k=4)

    print("\nRETRIEVED PASSAGES:")

    for index, context in enumerate(contexts, start=1):
        metadata = context.get("metadata") or {}

        print(f"\n[{index}] {metadata.get('filename')}")
        print(context.get("content", ""))

    messages = prompt_builder.build(question, contexts)

    print("\nRAW LLM RESPONSE:")

    response = llm.generate(
        messages,
        temperature=0.2,
        max_tokens=512,
    )

    print(response)

print("\nDebugging completed.")
