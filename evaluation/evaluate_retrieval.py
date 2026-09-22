import json

from src.ingestion import load_documents
from src.chunking import chunk_text
from src.retriever import Retriever
from evaluation.retrieval_metrics import recall_at_k, mean_reciprocal_rank


def main():
    with open("data/eval/questions.json", encoding="utf-8") as file:
        questions = json.load(file)

    documents = load_documents()

    chunks = []
    chunk_sources = []

    for document in documents:
        document_chunks = chunk_text(
            document["text"],
            chunk_size=50,
            overlap=10
        )

        for chunk in document_chunks:
            chunks.append(chunk)
            chunk_sources.append(document["source"])

    retriever = Retriever(chunks)

    recall_scores = []
    mrr_scores = []

    for item in questions:
        results = retriever.retrieve(
            item["question"],
            top_k=3
        )

        retrieved_sources = [
            chunk_sources[
                next(
                    i for i, chunk in enumerate(chunks)
                    if chunk == result["text"]
                )
            ]
            for result in results
        ]

        recall = recall_at_k(
            retrieved_sources,
            item["source"],
            3
        )

        mrr = mean_reciprocal_rank(
            retrieved_sources,
            item["source"]
        )

        recall_scores.append(recall)
        mrr_scores.append(mrr)

        print(
            f"{item['question']} | "
            f"Recall@3={recall} | "
            f"MRR={mrr:.3f}"
        )

    print()
    print("Average Recall@3:", sum(recall_scores) / len(recall_scores))
    print("Average MRR:", sum(mrr_scores) / len(mrr_scores))


if __name__ == "__main__":
    main()
