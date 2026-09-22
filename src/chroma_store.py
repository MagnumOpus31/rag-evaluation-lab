import chromadb
import numpy as np

CHROMA_PATH = "data/chroma"

def create_chroma_store(chunks, embeddings):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name="rag_documents",
        configuration={"hnsw": {"space": "cosine"}}
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=np.asarray(embeddings).tolist()
    )

    return collection


def search_chroma_store(collection, query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=np.asarray(query_embedding).tolist(),
        n_results=top_k
    )

    return results
