import json
from pathlib import Path

from src.ingestion import load_documents
from src.chunking import chunk_text
from src.retriever import Retriever
from evaluation.retrieval_metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    evidence_recall_at_k
)


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

    top_k_values = [1, 3, 5]
    experiment_results = []

    for top_k in top_k_values:
        recall_scores = []
        mrr_scores = []
        evidence_recall_scores = []

        print()
        print("=" * 60)
        print(f"Evaluating retrieval with top_k={top_k}")
        print("=" * 60)

        for item in questions:
            if not item.get("in_domain", True):
                continue

            results = retriever.retrieve(
                item["question"],
                top_k=top_k
            )

            retrieved_sources = [
                chunk_sources[result["index"]]
                for result in results
            ]

            recall = recall_at_k(
                retrieved_sources,
                item["source"],
                top_k
            )

            mrr = mean_reciprocal_rank(
                retrieved_sources,
                item["source"]
            )

            evidence_recall = None

            if item.get("evidence"):
                retrieved_chunks = [
                    chunks[result["index"]]
                    for result in results
                ]

                evidence_recall = evidence_recall_at_k(
                    retrieved_chunks,
                    item["evidence"],
                    top_k
                )

            recall_scores.append(recall)
            mrr_scores.append(mrr)

            if evidence_recall is not None:
                evidence_recall_scores.append(evidence_recall)

            output = (
                f"{item['question']} | "
                f"Recall@{top_k}={recall} | "
                f"MRR={mrr:.3f}"
            )

            if evidence_recall is not None:
                output += (
                    f" | Evidence Recall@{top_k}="
                    f"{evidence_recall}"
                )

            print(output)

        print()
        print(
            f"Average Recall@{top_k}: "
            f"{sum(recall_scores) / len(recall_scores):.3f}"
        )

        print(
            f"Average MRR: "
            f"{sum(mrr_scores) / len(mrr_scores):.3f}"
        )

        if evidence_recall_scores:
            print(
                f"Average Evidence Recall@{top_k}: "
                f"{sum(evidence_recall_scores) / len(evidence_recall_scores):.3f}"
            )

        experiment_results.append({
            "top_k": top_k,
            "average_recall": sum(recall_scores) / len(recall_scores),
            "average_mrr": sum(mrr_scores) / len(mrr_scores),
            "average_evidence_recall": (
                sum(evidence_recall_scores) / len(evidence_recall_scores)
                if evidence_recall_scores
                else None
    )
})

    print()
    print("=" * 60)
    print("OOD Evaluation")
    print("=" * 60)

    ood_results = []

    for item in questions:
        if item.get("in_domain", True):
            continue

        results = retriever.retrieve(
            item["question"],
            top_k=5
        )

        top_score = results[0]["score"] if results else 0.0
        rejected = top_score < score_threshold

        ood_results.append(rejected)

        print(
            f"{item['question']} | "
            f"Rejected={rejected} | "
            f"TopScore={top_score:.3f}"
        )

    print()
    print(
        "OOD Rejection Rate:",
        f"{sum(ood_results) / len(ood_results):.3f}"
    )

    output_path = Path("data/eval/retrieval_results.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=4)

    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()