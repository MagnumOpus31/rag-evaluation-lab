from src.embeddings import create_embeddings


def context_support_score(model, answer, context):
    answer_embedding = create_embeddings(
        model,
        [answer]
    )[0]

    context_embedding = create_embeddings(
        model,
        [context]
    )[0]

    return float(answer_embedding @ context_embedding)