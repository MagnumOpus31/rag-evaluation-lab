import json

from src.rag_pipeline import RAGPipeline


def main():
    with open(
        "data/eval/questions.json",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    pipeline = RAGPipeline(
        chunk_size=50,
        overlap=10,
        score_threshold=0.3
    )

    results = []

    for item in questions:
        if item.get("in_domain", True):
            continue

        result = pipeline.answer(
            item["question"],
            top_k=3
        )

        refused = (
            result["answer"]
            == "I don't have enough information in the provided context."
        )

        result_data = {
            "question": item["question"],
            "answer": result["answer"],
            "refused": refused,
            "top_score": (
                result["retrieved"][0]["score"]
                if result["retrieved"]
                else 0.0
            )
        }

        results.append(result_data)

        print()
        print("=" * 60)
        print(f"Question: {item['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Refused: {refused}")
        print(
            f"Top Score: "
            f"{result_data['top_score']:.3f}"
        )

    total_questions = len(results)

    refused_questions = sum(
        result["refused"]
        for result in results
    )

    refusal_rate = (
        refused_questions / total_questions
        if total_questions > 0
        else 0.0
    )

    summary = {
        "total_ood_questions": total_questions,
        "refused_questions": refused_questions,
        "answered_questions": (
            total_questions - refused_questions
        ),
        "ood_refusal_rate": refusal_rate
    }

    output = {
        "summary": summary,
        "results": results
    }

    with open(
        "data/eval/ood_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=4
        )

    print()
    print("=" * 60)
    print("OOD Evaluation Summary")
    print("=" * 60)

    print(
        f"Total OOD Questions: "
        f"{total_questions}"
    )

    print(
        f"Refused Questions: "
        f"{refused_questions}"
    )

    print(
        f"Answered Questions: "
        f"{total_questions - refused_questions}"
    )

    print(
        f"OOD Refusal Rate: "
        f"{refusal_rate:.3f}"
    )

    print()
    print(
        "Results saved to "
        "data/eval/ood_results.json"
    )


if __name__ == "__main__":
    main()