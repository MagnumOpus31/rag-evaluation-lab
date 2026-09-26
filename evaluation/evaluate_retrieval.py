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

    score_thresholds = [0.2, 0.3, 0.4, 0.5]

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

        average_recall = sum(recall_scores) / len(recall_scores)
        average_mrr = sum(mrr_scores) / len(mrr_scores)

        print()
        print(
            f"Average Recall@{top_k}: "
            f"{average_recall:.3f}"
        )

        print(
            f"Average MRR: "
            f"{average_mrr:.3f}"
        )

        average_evidence_recall = None

        if evidence_recall_scores:
            average_evidence_recall = (
                sum(evidence_recall_scores)
                / len(evidence_recall_scores)
            )

            print(
                f"Average Evidence Recall@{top_k}: "
                f"{average_evidence_recall:.3f}"
            )

        experiment_results.append({
            "top_k": top_k,
            "average_recall": average_recall,
            "average_mrr": average_mrr,
            "average_evidence_recall": average_evidence_recall
        })

    print()
    print("=" * 60)
    print("Threshold Evaluation")
    print("=" * 60)

    threshold_results = []

    for score_threshold in score_thresholds:

        # OOD evaluation
        rejected_ood = 0
        total_ood = 0

        print()
        print(
            f"Evaluating threshold={score_threshold}"
        )

        for item in questions:
            if item.get("in_domain", True):
                continue

            results = retriever.retrieve(
                item["question"],
                top_k=5
            )

            top_score = (
                results[0]["score"]
                if results
                else 0.0
            )

            rejected = top_score < score_threshold

            total_ood += 1

            if rejected:
                rejected_ood += 1

            print(
                f"OOD: {item['question']} | "
                f"Rejected={rejected} | "
                f"TopScore={top_score:.3f}"
            )

        ood_rejection_rate = (
            rejected_ood / total_ood
            if total_ood
            else 0.0
        )

        # In-domain evaluation
        accepted_in_domain = 0
        total_in_domain = 0

        for item in questions:
            if not item.get("in_domain", True):
                continue

            results = retriever.retrieve(
                item["question"],
                top_k=1
            )

            top_score = (
                results[0]["score"]
                if results
                else 0.0
            )

            accepted = top_score >= score_threshold

            total_in_domain += 1

            if accepted:
                accepted_in_domain += 1

        in_domain_acceptance_rate = (
            accepted_in_domain / total_in_domain
            if total_in_domain
            else 0.0
        )

        threshold_results.append({
            "score_threshold": score_threshold,
            "ood_rejection_rate": ood_rejection_rate,
            "in_domain_acceptance_rate": in_domain_acceptance_rate
        })

        print(
            f"OOD Rejection Rate: "
            f"{ood_rejection_rate:.3f}"
        )

        print(
            f"In-Domain Acceptance Rate: "
            f"{in_domain_acceptance_rate:.3f}"
        )

    output_path = Path("data/eval/retrieval_results.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "retrieval_experiments": experiment_results,
                "threshold_evaluation": threshold_results
            },
            file,
            indent=4
        )

    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()