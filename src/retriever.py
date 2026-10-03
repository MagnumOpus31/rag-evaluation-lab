import os

from src.lightweight_retriever import LightweightRetriever


USE_LIGHTWEIGHT_RETRIEVER = (
    os.getenv("DEPLOYMENT_MODE") == "lightweight"
)


if not USE_LIGHTWEIGHT_RETRIEVER:
    from src.embeddings import create_embedding_model, create_embeddings
    from src.vector_store import create_vector_store, search_vector_store


class Retriever:
    def __init__(self, chunks, embedding_model=None):
        self.chunks = chunks

        if USE_LIGHTWEIGHT_RETRIEVER:
            self.lightweight_retriever = LightweightRetriever(chunks)
            return

        self.model = (
            embedding_model
            if embedding_model is not None
            else create_embedding_model()
        )

        embeddings = create_embeddings(
            self.model,
            chunks
        )

        self.index = create_vector_store(embeddings)

    def retrieve(self, query, top_k=3):

        if USE_LIGHTWEIGHT_RETRIEVER:
            return self.lightweight_retriever.retrieve(
                query,
                top_k=top_k
            )

        query_embedding = create_embeddings(
            self.model,
            [query]
        )

        scores, indices = search_vector_store(
            self.index,
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores, indices):
            if index == -1:
                continue

            results.append(
                {
                    "text": self.chunks[index],
                    "score": float(score),
                    "index": int(index)
                }
            )

        return results