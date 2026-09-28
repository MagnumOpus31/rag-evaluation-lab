import json

from src.embeddings import create_embedding_model
from src.rag_pipeline import RAGPipeline
from evaluation.answer_metrics import semantic_similarity
from evaluation.grounding_metrics import context_support_score
from evaluation.retrieval_metrics import evidence_recall_at_k


def main():
    with open(
        "data/eval/questions.json",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    # Fixed evaluation model.
    # This remains constant while the retrieval
    # embedding model is changed.
    evaluation_model = create_embedding_model(
        "all-MiniLM-L6-v2"
    )

    embedding_models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ]

    results = []

    for model_name in embedding_models:
        print()
        print("=" * 60)
        print(
            f"Evaluating embedding_model={model_name}"
        )
        print("=" * 60)

        # This model controls retrieval.
        embedding_model = create_embedding_model(
            model_name
        )

        pipeline = RAGPipeline(
            chunk_size=50,
            overlap=10,
            score_threshold=0.3,
            embedding_model=embedding_model
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

            # Use the fixed evaluation model so that
            # answer scoring is independent of the
            # retrieval embedding model.
            answer_score = semantic_similarity(
                evaluation_model,
                result["answer"],
                item["answer"]
            )

            context = "\n\n".join(
                retrieved["text"]
                for retrieved in result["retrieved"]
            )

            # Use the fixed evaluation model for
            # context-support scoring as well.
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
            "embedding_model": model_name,
            "evaluation_model": "all-MiniLM-L6-v2",
            "chunk_size": 50,
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
        "data/eval/embedding_results.json",
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
        "data/eval/embedding_results.json"
    )


if __name__ == "__main__":
    main()