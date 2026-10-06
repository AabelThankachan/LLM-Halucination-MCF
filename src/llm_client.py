import ollama


MODEL_NAME = "llama3.2:3b"


def generate_response(prompt):
    """
    Send a prompt to the local Ollama model
    and return its response.
    """

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


if __name__ == "__main__":

    prompt = """
Solve this problem step by step:

Janet's ducks lay 16 eggs per day.
She eats 3 eggs for breakfast and uses 4 eggs
for baking muffins.

She sells the remaining eggs for $2 each.

How much money does she make every day?
"""

    answer = generate_response(prompt)

    print("\n===== MODEL RESPONSE =====\n")
    print(answer)