import re

from datasets import load_dataset

from llm_client import generate_response
from embedding_client import get_embedding, cosine_similarity


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NUM_QUESTIONS = 5
NUM_NORMAL_AGENTS = 3


# ============================================================
# NORMAL AGENT PROMPTS
# ============================================================

NORMAL_PROMPTS = [

    """
Solve the following mathematical problem step by step.

Carefully identify all quantities and relationships in the problem.
Perform the calculations carefully.

At the very end, output exactly one line in this format:

FINAL_ANSWER: <number>

Do not write anything after the FINAL_ANSWER line.
""",

    """
Solve the following mathematical problem independently.

Check the interpretation of the problem and verify every calculation.

At the very end, output exactly one line in this format:

FINAL_ANSWER: <number>

Do not write anything after the FINAL_ANSWER line.
""",

    """
Solve the following mathematical problem step by step.

After completing the solution, perform a second check of your reasoning
and arithmetic.

At the very end, output exactly one line in this format:

FINAL_ANSWER: <number>

Do not write anything after the FINAL_ANSWER line.
"""
]


# ============================================================
# ADVERSARIAL / CHALLENGE AGENT PROMPT
# ============================================================

ADVERSARIAL_PROMPT = """
You are a challenge agent in a multi-agent mathematical reasoning system.

Your task is to construct ONE plausible but incorrect solution.

Do NOT solve the problem correctly.

Follow these rules:

1. Read the problem carefully.
2. Identify one quantity, relationship, percentage, unit, or operation
   that could realistically be misunderstood.
3. Build your reasoning around that misunderstanding.
4. Perform the calculations consistently with that interpretation.
5. Produce ONE plausible incorrect numerical answer.
6. Do not provide a corrected solution.
7. Do not mention the correct answer.
8. Do not say that your reasoning is wrong.
9. Do not refuse the task.
10. Do not use random numbers.
11. The incorrect answer must come from a plausible reasoning mistake.
12. Output only ONE final answer.

At the very end, output exactly one line in this format:

FINAL_ANSWER: <number>

Do not write anything after the FINAL_ANSWER line.

Question:

{question}
"""


# ============================================================
# EXTRACT FINAL ANSWER
# ============================================================

def extract_final_answer(text):
    """
    Extract ONLY the number appearing after:

        FINAL_ANSWER: <number>

    We deliberately do not search arbitrary numbers in the reasoning.
    This prevents numbers such as 6 inches, 40%, 3 days, etc.
    from being incorrectly interpreted as the final answer.
    """

    if text is None:
        return None

    # Look specifically for FINAL_ANSWER
    match = re.search(
        r"FINAL_ANSWER\s*:\s*"
        r"\$?\s*"
        r"(-?\d[\d,]*(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).replace(",", "")

    # If marker is missing, do NOT guess.
    return None


# ============================================================
# EXTRACT GSM8K GROUND TRUTH
# ============================================================

def extract_ground_truth(answer):
    """
    GSM8K answers normally end with:

        #### 18

    Extract only the number after ####.
    """

    if answer is None:
        return None

    match = re.search(
        r"####\s*(-?\d[\d,]*(?:\.\d+)?)",
        answer
    )

    if match:
        return match.group(1).replace(",", "")

    return None


# ============================================================
# NORMALIZE NUMBERS
# ============================================================

def normalize_number(value):
    """
    Normalize numerical strings so that:

        18
        18.0
        18.00

    are treated as the same value.
    """

    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except (ValueError, TypeError):
        return str(value).strip()


# ============================================================
# GENERATE NORMAL AGENT RESPONSE
# ============================================================

def generate_normal_response(question, agent_number):

    prompt = f"""
{NORMAL_PROMPTS[agent_number]}

Question:

{question}
"""

    response = generate_response(prompt)

    answer = normalize_number(
        extract_final_answer(response)
    )

    return response, answer


# ============================================================
# GENERATE ADVERSARIAL RESPONSE
# ============================================================

def generate_adversarial_response(question):

    prompt = ADVERSARIAL_PROMPT.format(
        question=question
    )

    response = generate_response(prompt)

    answer = normalize_number(
        extract_final_answer(response)
    )

    return response, answer


# ============================================================
# CALCULATE EMBEDDING SIMILARITIES
# ============================================================

def calculate_similarities(
    normal_responses,
    adversarial_response
):

    adversarial_embedding = get_embedding(
        adversarial_response
    )

    similarities = []

    for response in normal_responses:

        response_embedding = get_embedding(
            response
        )

        similarity = cosine_similarity(
            response_embedding,
            adversarial_embedding
        )

        similarities.append(similarity)

    return similarities


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("Loading GSM8K dataset...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main"
    )

    test_data = dataset["test"]

    selected_correct = 0

    # Statistics for diagnostics
    extraction_failures = 0
    adversarial_extraction_failures = 0

    agent_correct = [0] * NUM_NORMAL_AGENTS

    print("\n")

    print("=" * 70)
    print("MCF EMBEDDING FILTER EXPERIMENT")
    print("=" * 70)

    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Normal agents: {NUM_NORMAL_AGENTS}")

    print("=" * 70)


    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

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

        print("\nGround truth:", ground_truth)


        # ====================================================
        # NORMAL AGENTS
        # ====================================================

        normal_responses = []
        normal_answers = []


        for agent_number in range(NUM_NORMAL_AGENTS):

            response, answer = generate_normal_response(
                question,
                agent_number
            )

            normal_responses.append(response)
            normal_answers.append(answer)


            print("\n" + "-" * 70)
            print(
                f"NORMAL AGENT {agent_number + 1}"
            )
            print("-" * 70)

            print(response)

            print(
                "\nPredicted answer:",
                answer
            )


            # -----------------------------------------------
            # Individual agent accuracy
            # -----------------------------------------------

            if answer is not None and answer == ground_truth:

                agent_correct[agent_number] += 1


            # -----------------------------------------------
            # Extraction failure
            # -----------------------------------------------

            if answer is None:

                extraction_failures += 1

                print(
                    "WARNING: FINAL_ANSWER marker "
                    "was not detected."
                )


        # ====================================================
        # ADVERSARIAL AGENT
        # ====================================================

        (
            adversarial_response,
            adversarial_answer
        ) = generate_adversarial_response(
            question
        )


        print("\n" + "-" * 70)
        print("ADVERSARIAL AGENT")
        print("-" * 70)

        print(adversarial_response)

        print(
            "\nPredicted answer:",
            adversarial_answer
        )


        if adversarial_answer is None:

            adversarial_extraction_failures += 1

            print(
                "WARNING: Adversarial agent did not "
                "produce a valid FINAL_ANSWER."
            )


        # ====================================================
        # CALCULATE SIMILARITIES
        # ====================================================

        similarities = calculate_similarities(
            normal_responses,
            adversarial_response
        )


        print("\n" + "=" * 70)
        print("SIMILARITY RESULTS")
        print("=" * 70)


        for i, similarity in enumerate(similarities):

            print(
                f"Agent {i + 1} similarity "
                f"with adversarial: "
                f"{similarity:.4f}"
            )


        # ====================================================
        # MCF FILTERING DECISION
        # ====================================================
        #
        # Paper-inspired rule:
        #
        # Select the normal response having the
        # MINIMUM similarity to the adversarial response.
        #
        # ====================================================

        selected_agent = similarities.index(
            min(similarities)
        )

        selected_answer = normal_answers[
            selected_agent
        ]


        print("\n" + "=" * 70)
        print("FILTERING DECISION")
        print("=" * 70)


        print(
            f"Selected normal agent: "
            f"Agent {selected_agent + 1}"
        )

        print(
            f"Selected answer: "
            f"{selected_answer}"
        )

        print(
            f"Ground truth: "
            f"{ground_truth}"
        )


        # ====================================================
        # EVALUATE MCF SELECTION
        # ====================================================

        if (
            selected_answer is not None
            and selected_answer == ground_truth
        ):

            selected_correct += 1

            print(
                "MCF selection result: CORRECT"
            )

        else:

            print(
                "MCF selection result: INCORRECT"
            )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    accuracy = (
        selected_correct /
        NUM_QUESTIONS
    ) * 100


    print("\n")

    print("=" * 70)
    print("FINAL MCF EMBEDDING FILTER RESULTS")
    print("=" * 70)


    print(
        f"Correct selections: "
        f"{selected_correct}/{NUM_QUESTIONS}"
    )

    print(
        f"Selection accuracy: "
        f"{accuracy:.2f}%"
    )


    # ========================================================
    # INDIVIDUAL AGENT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INDIVIDUAL NORMAL AGENT RESULTS")
    print("=" * 70)


    for i in range(NUM_NORMAL_AGENTS):

        agent_accuracy = (
            agent_correct[i] /
            NUM_QUESTIONS
        ) * 100

        print(
            f"Agent {i + 1}: "
            f"{agent_correct[i]}/{NUM_QUESTIONS} "
            f"({agent_accuracy:.2f}%)"
        )


    # ========================================================
    # EXTRACTION DIAGNOSTICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXTRACTION DIAGNOSTICS")
    print("=" * 70)


    print(
        "Normal-agent extraction failures:",
        extraction_failures
    )

    print(
        "Adversarial extraction failures:",
        adversarial_extraction_failures
    )


    print("\n")
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()