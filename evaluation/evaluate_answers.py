import json

from src.rag_pipeline import RAGPipeline
from src.embeddings import create_embedding_model
from evaluation.answer_metrics import semantic_similarity


def main():
    with open("data/eval/questions.json", encoding="utf-8") as file:
        questions = json.load(file)

    pipeline = RAGPipeline()
    model = create_embedding_model()

    top_k_values = [1, 3, 5]

    for top_k in top_k_values:
        scores = []

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

            score = semantic_similarity(
                model,
                result["answer"],
                item["answer"]
            )

            scores.append(score)

            print(f"\nQuestion: {item['question']}")
            print(f"Semantic Similarity: {score:.3f}")

        average_score = sum(scores) / len(scores)

        print()
        print(
            f"Average Answer Semantic Similarity "
            f"(top_k={top_k}): {average_score:.3f}"
        )


if __name__ == "__main__":
    main()