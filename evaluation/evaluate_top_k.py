import json

from src.rag_pipeline import RAGPipeline
from src.embeddings import create_embedding_model
from evaluation.answer_metrics import semantic_similarity
from evaluation.grounding_metrics import context_support_score
from evaluation.retrieval_metrics import evidence_recall_at_k


def main():
    with open(
        "data/eval/questions.json",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    evaluation_model = create_embedding_model(
        "all-MiniLM-L6-v2"
    )

    pipeline = RAGPipeline(
        chunk_size=50,
        overlap=10,
        score_threshold=0.3
    )

    top_k_values = [1, 3, 5]

    results = []

    for top_k in top_k_values:
        print()
        print("=" * 60)
        print(f"Evaluating top_k={top_k}")
        print("=" * 60)

        evidence_scores = []
        answer_scores = []
        grounding_scores = []

        retrieval_latencies = []
        generation_latencies = []
        total_latencies = []

        for item in questions:
            if not item.get("in_domain", True):
                continue

            result = pipeline.answer(
                item["question"],
                top_k=top_k
            )

            retrieved_chunks = [
                retrieved["text"]
                for retrieved in result["retrieved"]
            ]

            evidence = evidence_recall_at_k(
                retrieved_chunks,
                item["evidence"],
                top_k
            )

            answer_score = semantic_similarity(
                evaluation_model,
                result["answer"],
                item["answer"]
            )

            context = "\n\n".join(
                retrieved["text"]
                for retrieved in result["retrieved"]
            )

            grounding = context_support_score(
                evaluation_model,
                result["answer"],
                context
            )

            evidence_scores.append(evidence)
            answer_scores.append(answer_score)
            grounding_scores.append(grounding)

            retrieval_latencies.append(
                result["latency"]["retrieval_seconds"]
            )

            generation_latencies.append(
                result["latency"]["generation_seconds"]
            )

            total_latencies.append(
                result["latency"]["total_seconds"]
            )

        result_data = {
            "top_k": top_k,
            "embedding_model": "all-MiniLM-L6-v2",
            "evaluation_model": "all-MiniLM-L6-v2",
            "chunk_size": 50,
            "overlap": 10,
            "score_threshold": 0.3,
            "average_evidence_recall": (
                sum(evidence_scores)
                / len(evidence_scores)
            ),
            "average_answer_similarity": (
                sum(answer_scores)
                / len(answer_scores)
            ),
            "average_grounding_score": (
                sum(grounding_scores)
                / len(grounding_scores)
            ),
            "average_retrieval_latency_seconds": (
                sum(retrieval_latencies)
                / len(retrieval_latencies)
            ),
            "average_generation_latency_seconds": (
                sum(generation_latencies)
                / len(generation_latencies)
            ),
            "average_latency_seconds": (
                sum(total_latencies)
                / len(total_latencies)
            )
        }

        results.append(result_data)

        print(
            f"Evidence Recall@{top_k}: "
            f"{result_data['average_evidence_recall']:.3f}"
        )

        print(
            f"Answer Similarity: "
            f"{result_data['average_answer_similarity']:.3f}"
        )

        print(
            f"Grounding Score: "
            f"{result_data['average_grounding_score']:.3f}"
        )

        print(
            f"Retrieval Latency: "
            f"{result_data['average_retrieval_latency_seconds']:.3f} seconds"
        )

        print(
            f"Generation Latency: "
            f"{result_data['average_generation_latency_seconds']:.3f} seconds"
        )

        print(
            f"Total Latency: "
            f"{result_data['average_latency_seconds']:.3f} seconds"
        )

    with open(
        "data/eval/top_k_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        "Results saved to "
        "data/eval/top_k_results.json"
    )


if __name__ == "__main__":
    main()