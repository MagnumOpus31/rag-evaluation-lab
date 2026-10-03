import os

from ollama import Client


LOCAL_MODEL_NAME = "qwen2.5:3b"
CLOUD_MODEL_NAME = "gpt-oss:20b"

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")


if OLLAMA_API_KEY:
    client = Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        }
    )
    MODEL_NAME = CLOUD_MODEL_NAME
else:
    client = Client()
    MODEL_NAME = LOCAL_MODEL_NAME


def generate_answer(query, context):
    prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say:
"I don't have enough information in the provided context."

Answer:"""

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"]
