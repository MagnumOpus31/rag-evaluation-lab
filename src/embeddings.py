from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embedding_model(model_name=MODEL_NAME):
    return SentenceTransformer(model_name)


def create_embeddings(model, texts):
    return model.encode(texts, normalize_embeddings=True)
