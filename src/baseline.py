import re
from datasets import load_dataset
from llm_client import generate_response


def extract_final_answer(text):
    """
    Extract the final numerical answer from the model response.
    Handles numbers containing commas, such as $70,000.
    """

    patterns = [
        r"(?:answer is|answer:|final answer is|final answer:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"\$\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            return matches[-1].replace(",", "")

    # If no explicit answer is found, use the last number
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)

    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_ground_truth(answer):
    """
    GSM8K answers end with:
    #### numerical_answer

    Example:
    #### 18
    """

    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer)

    if match:
        return match.group(1)

    return None


def normalize_number(value):
    """
    Normalize numerical answers for comparison.
    """

    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except ValueError:
        return value.strip()


def main():

    print("Loading GSM8K dataset...")

    dataset = load_dataset("openai/gsm8k", "main")

    test_data = dataset["test"]

    # Start small.
    NUM_QUESTIONS = 5

    correct = 0

    print(f"\nRunning baseline on {NUM_QUESTIONS} questions...\n")

    for i in range(NUM_QUESTIONS):

        question = test_data[i]["question"]
        ground_truth = extract_ground_truth(test_data[i]["answer"])

        prompt = f"""
Solve the following mathematical problem step by step.

Question:
{question}

At the end, clearly state your final numerical answer.
"""

        print("=" * 70)
        print(f"QUESTION {i + 1}")
        print("=" * 70)

        print(question)

        response = generate_response(prompt)

        predicted_answer = extract_final_answer(response)

        predicted_answer = normalize_number(predicted_answer)
        ground_truth = normalize_number(ground_truth)

        print("\nMODEL RESPONSE:")
        print(response)

        print("\nPREDICTED ANSWER:", predicted_answer)
        print("GROUND TRUTH:", ground_truth)

        if predicted_answer == ground_truth:
            correct += 1
            print("RESULT: CORRECT")
        else:
            print("RESULT: INCORRECT")

    accuracy = (correct / NUM_QUESTIONS) * 100

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(f"Questions evaluated: {NUM_QUESTIONS}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {NUM_QUESTIONS - correct}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()