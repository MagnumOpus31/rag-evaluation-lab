
import json
import time
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.embeddings import create_embedding_model
from evaluation.answer_metrics import semantic_similarity


def main():
    with open("data/eval/questions.json", encoding="utf-8") as file:
        questions = json.load(file)

    pipeline = RAGPipeline()
    model = create_embedding_model()

    top_k_values = [1, 3, 5]
    experiment_results=[]

    for top_k in top_k_values:
        scores = []
        latencies = []

        print()
        print("=" * 60)
        print(f"Evaluating with top_k={top_k}")
        print("=" * 60)

        for item in questions:
            if not item.get("in_domain", True):
                continue

            start_time = time.perf_counter()

            result = pipeline.answer(
                item["question"],
                top_k=top_k
            )

            latency = time.perf_counter() - start_time
            latencies.append(latency)

            score = semantic_similarity(
                model,
                result["answer"],
                item["answer"]
            )

            scores.append(score)

            print(f"\nQuestion: {item['question']}")
            print(f"Semantic Similarity: {score:.3f}")
            print(f"Latency: {latency:.3f} seconds")

        average_score = sum(scores) / len(scores)
        average_latency = sum(latencies) / len(latencies)

        experiment_results.append({
            "top_k": top_k,
            "average_answer_similarity": average_score,
            "average_latency_seconds": average_latency
        })

        print()
        print(
            f"Average Answer Semantic Similarity "
            f"(top_k={top_k}): {average_score:.3f}"
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
