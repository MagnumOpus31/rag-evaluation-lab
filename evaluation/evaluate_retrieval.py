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
    score_threshold = 0.4

    recall_scores = []
    mrr_scores = []
    ood_results = []

    for item in questions:
        results = retriever.retrieve(
            item["question"],
            top_k=3
        )

        in_domain = item.get("in_domain", True)

        if not in_domain:
            top_score = results[0]["score"] if results else 0.0
            rejected = top_score < score_threshold
            ood_results.append(rejected)
            print(
                f"{item['question']} | "
                f"Rejected={rejected} | "
                f"TopScore={top_score:.3f}"
            )
            continue

        retrieved_sources = [
            chunk_sources[result["index"]]
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
            f"Recall@3={recall} | MRR={mrr:.3f}"
        )

    print()
    print("In-domain Average Recall@3:", sum(recall_scores) / len(recall_scores))
    print("In-domain Average MRR:", sum(mrr_scores) / len(mrr_scores))
    print("OOD Rejection Rate:", sum(ood_results) / len(ood_results))



if __name__ == "__main__":
    main()
