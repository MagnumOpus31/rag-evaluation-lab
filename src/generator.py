import ollama

MODEL_NAME = "qwen2.5:3b"


def generate_answer(query, context):
    prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say:
"I don't have enough information in the provided context."

Answer:"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
