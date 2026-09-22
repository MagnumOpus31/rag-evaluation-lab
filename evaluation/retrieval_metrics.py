def recall_at_k(retrieved_sources, expected_source, k):
    top_k_sources = retrieved_sources[:k]

    return int(expected_source in top_k_sources)


def mean_reciprocal_rank(retrieved_sources, expected_source):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1 / rank

    return 0.0
