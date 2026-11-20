import ollama

MODEL_NAME = "llama3.2:3b"


def generate_adversarial_response(question):
    prompt = f"""
You are a challenge agent in a multi-agent mathematical reasoning system.

Your job is NOT to solve the problem correctly.

Instead, construct ONE plausible but incorrect solution that another
reasoning agent might realistically produce.

Follow these rules:

1. Read the problem carefully.
2. Identify one quantity, relationship, percentage, unit, or operation
   that could easily be misunderstood.
3. Build your reasoning around that possible misunderstanding.
4. Perform the calculations consistently with that interpretation.
5. Produce ONE final numerical answer.
6. Do not provide a corrected solution.
7. Do not mention the correct answer.
8. Do not say that your reasoning is wrong.
9. Do not refuse the task.
10. Do not use random numbers. The incorrect answer must come from
    a plausible reasoning mistake.
    

Your response should contain:
- reasoning steps
- one final numerical answer

Problem:
{question}
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
    question = """
Josh decides to try flipping a house.
He buys a house for $80,000 and then puts in $50,000 in repairs.
This increased the value of the house by 150%.
How much profit did he make?
"""

    answer = generate_adversarial_response(question)

    print("\n===== CHALLENGE AGENT RESPONSE =====\n")
    print(answer)