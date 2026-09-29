import json

from src.rag_pipeline import RAGPipeline
from evaluation.answer_correctness import judge_answer


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
        if not item.get("in_domain", True):
            continue

        print()
        print("=" * 60)
        print(
            f"Evaluating: {item['question']}"
        )
        print("=" * 60)

        result = pipeline.answer(
            item["question"],
            top_k=3
        )

        judgment = judge_answer(
            item["question"],
            result["answer"],
            item["answer"]
        )

        result_data = {
            "question": item["question"],
            "expected_answer": item["answer"],
            "generated_answer": result["answer"],
            "correct": judgment["correct"],
            "reason": judgment["reason"]
        }

        results.append(result_data)

        print(
            f"Correct: {judgment['correct']}"
        )

        print(
            f"Reason: {judgment['reason']}"
        )

    correct_count = sum(
        result["correct"]
        for result in results
    )

    total_count = len(results)

    accuracy = (
        correct_count / total_count
        if total_count > 0
        else 0.0
    )

    summary = {
        "total_questions": total_count,
        "correct_answers": correct_count,
        "incorrect_answers": (
            total_count - correct_count
        ),
        "answer_correctness": accuracy
    }

    output = {
        "summary": summary,
        "results": results
    }

    with open(
        "data/eval/correctness_results.json",
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
    print("Answer Correctness Summary")
    print("=" * 60)

    print(
        f"Total Questions: {total_count}"
    )

    print(
        f"Correct Answers: {correct_count}"
    )

    print(
        f"Incorrect Answers: "
        f"{total_count - correct_count}"
    )

    print(
        f"Answer Correctness: "
        f"{accuracy:.3f}"
    )

    print()
    print(
        "Results saved to "
        "data/eval/correctness_results.json"
    )


if __name__ == "__main__":
    main()