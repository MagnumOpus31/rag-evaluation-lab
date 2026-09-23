from src.embeddings import create_embedding_model, create_embeddings
from src.vector_store import create_vector_store, search_vector_store


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = create_embedding_model()

        embeddings = create_embeddings(self.model, chunks)
        self.index = create_vector_store(embeddings)

    def retrieve(self, query, top_k=3):
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

            results.append({
                "text": self.chunks[index],
                "score": float(score),
                "index": int(index)
            })

        return results
