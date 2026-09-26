import json
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def main():
    retrieval_data = load_json(
        "data/eval/retrieval_results.json"
    )

    answer_results = load_json(
        "data/eval/answer_results.json"
    )

    retrieval_results = retrieval_data[
        "retrieval_experiments"
    ]

    threshold_results = retrieval_data[
        "threshold_evaluation"
    ]

    config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 50,
        "chunk_overlap": 10,
        "score_threshold": 0.3,
        "generation_model": "qwen2.5:3b"
    }

    report = []

    for retrieval in retrieval_results:
        for answer in answer_results:
            if retrieval["top_k"] == answer["top_k"]:
                report.append({
                    "top_k": retrieval["top_k"],
                    "average_recall": retrieval["average_recall"],
                    "average_mrr": retrieval["average_mrr"],
                    "average_evidence_recall":
                        retrieval["average_evidence_recall"],
                    "average_answer_similarity":
                        answer["average_answer_similarity"],
                    "average_grounding_score":
                        answer["average_grounding_score"],
                    "average_retrieval_latency_seconds":
                        answer[
                            "average_retrieval_latency_seconds"
                        ],
                    "average_generation_latency_seconds":
                        answer[
                            "average_generation_latency_seconds"
                        ],
                    "average_latency_seconds":
                        answer["average_latency_seconds"]
                })

    selected_threshold = None

    for result in threshold_results:
        if result["score_threshold"] == config["score_threshold"]:
            selected_threshold = result
            break

    print()
    print("Experiment Configuration")
    print("-" * 30)

    for key, value in config.items():
        print(f"{key}: {value}")

    print()
    print("=" * 90)
    print("Unified RAG Evaluation Report")
    print("=" * 90)

    print(
        f"{'Top-K':<8}"
        f"{'Recall':<12}"
        f"{'MRR':<12}"
        f"{'Evidence Recall':<18}"
        f"{'Answer Similarity':<20}"
        f"{'Grounding':<12}"
        f"{'Retrieval (s)':<16}"
        f"{'Generation (s)':<16}"
        f"{'Total Latency (s)':<18}"
    )

    print("-" * 90)

    for result in report:
        print(
            f"{result['top_k']:<8}"
            f"{result['average_recall']:<12.3f}"
            f"{result['average_mrr']:<12.3f}"
            f"{result['average_evidence_recall']:<18.3f}"
            f"{result['average_answer_similarity']:<20.3f}"
            f"{result['average_grounding_score']:<12.3f}"
            f"{result['average_retrieval_latency_seconds']:<16.3f}"
            f"{result['average_generation_latency_seconds']:<16.3f}"
            f"{result['average_latency_seconds']:<18.3f}"
        )

    print()

    print("=" * 60)
    print("Threshold Evaluation")
    print("=" * 60)

    for result in threshold_results:
        print(
            f"Threshold {result['score_threshold']:.1f} | "
            f"OOD Rejection: "
            f"{result['ood_rejection_rate']:.3f} | "
            f"In-Domain Acceptance: "
            f"{result['in_domain_acceptance_rate']:.3f}"
        )

    if selected_threshold:
        print()
        print(
            f"Selected Threshold: "
            f"{config['score_threshold']}"
        )

        print(
            f"OOD Rejection Rate: "
            f"{selected_threshold['ood_rejection_rate']:.3f}"
        )

        print(
            f"In-Domain Acceptance Rate: "
            f"{selected_threshold['in_domain_acceptance_rate']:.3f}"
        )

    final_report = {
        "configuration": config,
        "results": report,
        "threshold_evaluation": threshold_results
    }

    output_path = Path(
        "data/eval/unified_report.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            final_report,
            file,
            indent=4
        )

    print()
    print(
        f"Unified report saved to {output_path}"
    )


if __name__ == "__main__":
    main()