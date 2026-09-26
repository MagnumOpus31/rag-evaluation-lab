import time
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
        total_start=time.perf_counter()
        retrieval_start=time.perf_counter()
        retrieved = self.retriever.retrieve(
            query,
            top_k=top_k
        )
        retrieval_latency=time.perf_counter()-retrieval_start

        if not retrieved or retrieved[0]["score"] < self.score_threshold:
            total_latency = time.perf_counter() - total_start
            return {
                "query": query,
                "answer": "I don't have enough information in the provided context.",
                "retrieved": retrieved,
                "latency": {
                "retrieval_seconds": retrieval_latency,
                "generation_seconds": 0.0,
                "total_seconds": total_latency
                }
            }

        context = "\n\n".join(
            result["text"]
            for result in retrieved
        )

        generation_start=time.perf_counter()
        answer = generate_answer(
            query,
            context
        )

        generation_latency=time.perf_counter()-generation_start
        total_latency = time.perf_counter() - total_start

        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieved,
            "latency": {
                "retrieval_seconds": retrieval_latency,
                "generation_seconds": generation_latency,
                "total_seconds": total_latency
    }
}
