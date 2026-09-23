from src.ingestion import load_documents
from src.chunking import chunk_text
from src.retriever import Retriever
from src.generator import generate_answer


class RAGPipeline:
    def __init__(self, chunk_size=50, overlap=10, score_threshold=0.4):
        documents = load_documents()

        self.chunks = [
            chunk
            for document in documents
            for chunk in chunk_text(
                document["text"],
                chunk_size=chunk_size,
                overlap=overlap
            )
        ]

        self.retriever = Retriever(self.chunks)
        self.score_threshold = score_threshold

    def answer(self, query, top_k=3):
        retrieved = self.retriever.retrieve(
            query,
            top_k=top_k
        )

        if not retrieved or retrieved[0]["score"] < self.score_threshold:
            return {
                "query": query,
                "answer": "I don't have enough information in the provided context.",
                "retrieved": retrieved
            }

        context = "\n\n".join(
            result["text"]
            for result in retrieved
        )

        answer = generate_answer(
            query,
            context
        )

        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieved
        }
