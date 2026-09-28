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

    model = create_embedding_model()

    chunk_sizes = [30, 50, 100]
    results = []

    for chunk_size in chunk_sizes:
        print()
        print("=" * 60)
        print(
            f"Evaluating chunk_size={chunk_size}"
        )
        print("=" * 60)

        pipeline = RAGPipeline(
            chunk_size=chunk_size,
            overlap=10,
            score_threshold=0.3
        )

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
                top_k=3
            )

            retrieved_chunks = [
                retrieved["text"]
                for retrieved in result["retrieved"]
            ]

            evidence = evidence_recall_at_k(
                retrieved_chunks,
                item["evidence"],
                3
            )

            answer_score = semantic_similarity(
                model,
                result["answer"],
                item["answer"]
            )

            context = "\n\n".join(
                retrieved["text"]
                for retrieved in result["retrieved"]
            )

            grounding = context_support_score(
                model,
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
            "chunk_size": chunk_size,
            "overlap": 10,
            "top_k": 3,
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
            f"Evidence Recall@3: "
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
        "data/eval/chunking_results.json",
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
        "data/eval/chunking_results.json"
    )


if __name__ == "__main__":
    main()