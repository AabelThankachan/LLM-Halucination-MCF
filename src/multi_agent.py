import re
from datasets import load_dataset
from llm_client import generate_response


NUM_QUESTIONS = 5
NUM_AGENTS = 3


AGENT_PROMPTS = [
    """
Solve the following mathematical problem step by step.

Carefully identify all quantities and operations in the problem.
At the end, clearly state your final numerical answer.
""",

    """
Solve the following mathematical problem independently.

Pay special attention to percentages, multiplication, division,
and quantities that occur multiple times. Verify your calculations
before giving the final answer.

At the end, clearly state your final numerical answer.
""",

    """
Solve the following mathematical problem step by step.

After obtaining your answer, perform a second check of the reasoning
and calculations. Correct any mistake you find.

At the end, clearly state your final numerical answer.
"""
]


def extract_final_answer(text):
    """
    Extract the final numerical answer from a model response.
    Handles values such as:
        18
        $18
        70,000
        $70,000
    """

    patterns = [
        r"(?:answer is|answer:|final answer is|final answer:)"
        r"\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",

        r"\$\s*(-?\d[\d,]*(?:\.\d+)?)"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            return matches[-1].replace(",", "")

    numbers = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?",
        text
    )

    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_ground_truth(answer):
    """
    GSM8K answers use the format:

    #### numerical_answer
    """

    match = re.search(
        r"####\s*(-?\d[\d,]*(?:\.\d+)?)",
        answer
    )

    if match:
        return match.group(1).replace(",", "")

    return None


def normalize_number(value):

    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except ValueError:
        return value.strip()


def run_agent(question, agent_number):

    prompt = f"""
{AGENT_PROMPTS[agent_number]}

Question:
{question}
"""

    response = generate_response(prompt)

    answer = extract_final_answer(response)

    return response, normalize_number(answer)


def main():

    print("Loading GSM8K dataset...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main"
    )

    test_data = dataset["test"]

    agent_correct = [0] * NUM_AGENTS
    questions_with_disagreement = 0

    print()
    print("=" * 70)
    print("MULTI-AGENT EXPERIMENT")
    print("=" * 70)
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Normal agents: {NUM_AGENTS}")
    print("=" * 70)

    for question_index in range(NUM_QUESTIONS):

        question = test_data[question_index]["question"]

        ground_truth = normalize_number(
            extract_ground_truth(
                test_data[question_index]["answer"]
            )
        )

        print("\n")
        print("#" * 70)
        print(f"QUESTION {question_index + 1}")
        print("#" * 70)

        print(question)

        responses = []
        predictions = []

        # Run all normal agents
        for agent_number in range(NUM_AGENTS):

            print("\n" + "-" * 70)
            print(f"NORMAL AGENT {agent_number + 1}")
            print("-" * 70)

            response, prediction = run_agent(
                question,
                agent_number
            )

            responses.append(response)
            predictions.append(prediction)

            print(response)
            print("\nPredicted answer:", prediction)

            if prediction == ground_truth:
                agent_correct[agent_number] += 1
                print("Agent result: CORRECT")
            else:
                print("Agent result: INCORRECT")

        # Check whether agents disagree
        unique_predictions = set(predictions)

        if len(unique_predictions) > 1:
            questions_with_disagreement += 1
            disagreement = True
        else:
            disagreement = False

        print("\n" + "=" * 70)
        print("QUESTION SUMMARY")
        print("=" * 70)

        for i, prediction in enumerate(predictions):
            print(
                f"Agent {i + 1}: "
                f"{prediction}"
            )

        print(f"Ground truth: {ground_truth}")

        if disagreement:
            print("Agent agreement: DISAGREEMENT")
        else:
            print("Agent agreement: AGREEMENT")

    # Final statistics
    print("\n")
    print("=" * 70)
    print("MULTI-AGENT RESULTS")
    print("=" * 70)

    for agent_number in range(NUM_AGENTS):

        accuracy = (
            agent_correct[agent_number]
            / NUM_QUESTIONS
        ) * 100

        print(
            f"Agent {agent_number + 1}: "
            f"{agent_correct[agent_number]}/{NUM_QUESTIONS} "
            f"correct "
            f"({accuracy:.2f}%)"
        )

    print(
        f"\nQuestions with agent disagreement: "
        f"{questions_with_disagreement}/{NUM_QUESTIONS}"
    )


if __name__ == "__main__":
    main()