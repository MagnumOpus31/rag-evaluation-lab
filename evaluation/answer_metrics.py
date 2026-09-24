from src.embeddings import create_embedding_model, create_embeddings


def semantic_similarity(model, generated_answer, expected_answer):
    embeddings = create_embeddings(
        model,
        [generated_answer, expected_answer]
    )

    return float(embeddings[0] @ embeddings[1])
