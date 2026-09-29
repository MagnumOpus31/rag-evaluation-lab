import json

import ollama


MODEL_NAME = "qwen2.5:3b"


def judge_answer(
    question,
    generated_answer,
    expected_answer
):
    prompt = f"""
You are evaluating the correctness of an answer.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Determine whether the generated answer correctly answers
the question based on the expected answer.

Return ONLY valid JSON in exactly this format:

{{
    "correct": true,
    "reason": "brief explanation"
}}

Use true only when the generated answer contains the
essential information needed to answer the question correctly.
Use false if the answer is incorrect, contradictory, or
missing the essential information.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    content = response["message"]["content"]

    try:
        result = json.loads(content)

        if (
            "correct" not in result
            or "reason" not in result
        ):
            raise ValueError(
                "Judge response missing required fields"
            )

        if not isinstance(
            result["correct"],
            bool
        ):
            raise ValueError(
                "Judge 'correct' field must be boolean"
            )

        return result

    except (json.JSONDecodeError, ValueError):
        return {
            "correct": False,
            "reason": (
                "Invalid judge response: "
                + content
            )
        }