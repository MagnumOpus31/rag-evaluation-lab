
import json
import time
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.embeddings import create_embedding_model
from evaluation.answer_metrics import semantic_similarity
from src.rag_pipeline import RAGPipeline
from src.embeddings import create_embedding_model
from evaluation.answer_metrics import semantic_similarity
from evaluation.grounding_metrics import context_support_score


def main():
    with open("data/eval/questions.json", encoding="utf-8") as file:
        questions = json.load(file)

    pipeline = RAGPipeline()
    model = create_embedding_model()

    top_k_values = [1, 3, 5]
    experiment_results=[]

    for top_k in top_k_values:
        scores = []
        grounding_scores = []
        retrieval_latencies = []
        generation_latencies = []
        latencies = []


        print()
        print("=" * 60)
        print(f"Evaluating with top_k={top_k}")
        print("=" * 60)

        for item in questions:
            if not item.get("in_domain", True):
                continue

            result = pipeline.answer(
                item["question"],
                top_k=top_k
            )

            latency = result["latency"]["total_seconds"]
            latencies.append(latency)
            retrieval_latencies.append(
                result["latency"]["retrieval_seconds"]
            )

            generation_latencies.append(
                result["latency"]["generation_seconds"]
            )

            score = semantic_similarity(
                model,
                result["answer"],
                item["answer"]
            )

            context = "\n\n".join(
                retrieved["text"]
                for retrieved in result["retrieved"]
            )

            grounding_score = context_support_score(
                model,
                result["answer"],
                context
            )

            scores.append(score)
            grounding_scores.append(grounding_score)

            print(f"\nQuestion: {item['question']}")
            print(f"Semantic Similarity: {score:.3f}")
            print(f"Grounding Score: {grounding_score:.3f}")
            print(f"Latency: {latency:.3f} seconds")

            average_score = sum(scores) / len(scores)
            average_grounding = sum(grounding_scores) / len(grounding_scores)
            average_latency = sum(latencies) / len(latencies)
            average_retrieval_latency = (
                sum(retrieval_latencies) / len(retrieval_latencies)
            )

            average_generation_latency = (
                sum(generation_latencies) / len(generation_latencies)
)

        experiment_results.append({
            "top_k": top_k,
            "average_answer_similarity": average_score,
            "average_grounding_score": average_grounding,
            "average_retrieval_latency_seconds": average_retrieval_latency,
            "average_generation_latency_seconds": average_generation_latency,
            "average_latency_seconds": average_latency
        })

        print()
        print(
            f"Average Answer Semantic Similarity "
            f"(top_k={top_k}): {average_score:.3f}"
        )

        print(
            f"Average Grounding Score "
            f"(top_k={top_k}): {average_grounding:.3f}"
        )
        print(
            f"Average Retrieval Latency "
            f"(top_k={top_k}): {average_retrieval_latency:.3f} seconds"
        )

        print(
            f"Average Generation Latency "
            f"(top_k={top_k}): {average_generation_latency:.3f} seconds"
        )

        print(
            f"Average Latency "
            f"(top_k={top_k}): {average_latency:.3f} seconds"
        )

    output_path = Path("data/eval/answer_results.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=4)

    print()
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
