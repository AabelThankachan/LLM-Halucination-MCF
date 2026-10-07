import os
import csv
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "results/mcf_results.csv"
OUTPUT_FILE = "results/method_comparison.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_answer(answer):

    if answer is None:
        return None

    answer = str(answer).strip()

    if answer == "":
        return None

    try:
        value = float(answer)

        if value.is_integer():
            return str(int(value))

        return str(value)

    except Exception:
        return answer


def is_correct(predicted, ground_truth):

    predicted = normalize_answer(predicted)
    ground_truth = normalize_answer(ground_truth)

    if predicted is None or ground_truth is None:
        return False

    try:
        return abs(
            float(predicted) - float(ground_truth)
        ) < 1e-6

    except Exception:
        return predicted == ground_truth


# ============================================================
# MAJORITY VOTING
# ============================================================

def majority_vote(answers):

    # Remove missing answers
    valid_answers = [
        normalize_answer(a)
        for a in answers
        if normalize_answer(a) is not None
    ]

    if not valid_answers:
        return None

    counts = Counter(valid_answers)

    # A majority exists when an answer has at least 2 votes
    most_common = counts.most_common()

    if most_common[0][1] >= 2:
        return most_common[0][0]

    # No majority
    return None


# ============================================================
# LOAD CSV
# ============================================================

if not os.path.exists(INPUT_FILE):

    print(
        f"ERROR: Could not find {INPUT_FILE}"
    )

    print(
        "Run mcf_filter.py first."
    )

    exit()


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print()
print("=" * 70)
print("METHOD COMPARISON")
print("=" * 70)

print(
    f"Questions loaded: {len(rows)}"
)

print("=" * 70)


# ============================================================
# RESULT STORAGE
# ============================================================

comparison_rows = []

agent1_correct_count = 0
agent2_correct_count = 0
agent3_correct_count = 0

majority_correct_count = 0
mcf_correct_count = 0

majority_available_count = 0

# Agreement statistics
all_agree_count = 0
two_agree_count = 0
all_different_count = 0

# Disagreement statistics
disagreement_count = 0

# Cases where at least one agent is correct
at_least_one_correct_count = 0

# MCF vs majority
majority_correct_mcf_wrong = 0
mcf_correct_majority_wrong = 0

# MCF success on disagreement
mcf_success_disagreement = 0
majority_success_disagreement = 0

# Agent correctness
all_agents_correct = 0
all_agents_wrong = 0


# ============================================================
# PROCESS EACH QUESTION
# ============================================================

for row in rows:

    question_number = row["question_number"]

    ground_truth = normalize_answer(
        row["ground_truth"]
    )

    agent_answers = [
        normalize_answer(row["agent1_answer"]),
        normalize_answer(row["agent2_answer"]),
        normalize_answer(row["agent3_answer"])
    ]

    mcf_answer = normalize_answer(
        row["mcf_answer"]
    )

    # --------------------------------------------------------
    # Individual agents
    # --------------------------------------------------------

    agent1_correct = is_correct(
        agent_answers[0],
        ground_truth
    )

    agent2_correct = is_correct(
        agent_answers[1],
        ground_truth
    )

    agent3_correct = is_correct(
        agent_answers[2],
        ground_truth
    )

    if agent1_correct:
        agent1_correct_count += 1

    if agent2_correct:
        agent2_correct_count += 1

    if agent3_correct:
        agent3_correct_count += 1

    # --------------------------------------------------------
    # Majority voting
    # --------------------------------------------------------

    majority_answer = majority_vote(
        agent_answers
    )

    majority_available = (
        majority_answer is not None
    )

    if majority_available:
        majority_available_count += 1

    majority_correct = is_correct(
        majority_answer,
        ground_truth
    )

    if majority_correct:
        majority_correct_count += 1

    # --------------------------------------------------------
    # MCF
    # --------------------------------------------------------

    mcf_correct = is_correct(
        mcf_answer,
        ground_truth
    )

    if mcf_correct:
        mcf_correct_count += 1

    # --------------------------------------------------------
    # Agent answer agreement
    # --------------------------------------------------------

    unique_answers = set(
        a for a in agent_answers
        if a is not None
    )

    if len(unique_answers) == 1:

        all_agree_count += 1

    elif len(unique_answers) == 2:

        two_agree_count += 1

    elif len(unique_answers) == 3:

        all_different_count += 1

    # --------------------------------------------------------
    # Disagreement
    # --------------------------------------------------------

    agents_disagree = (
        len(unique_answers) > 1
    )

    if agents_disagree:

        disagreement_count += 1

        if mcf_correct:
            mcf_success_disagreement += 1

        if majority_correct:
            majority_success_disagreement += 1

    # --------------------------------------------------------
    # At least one agent correct
    # --------------------------------------------------------

    any_agent_correct = (
        agent1_correct
        or agent2_correct
        or agent3_correct
    )

    if any_agent_correct:

        at_least_one_correct_count += 1

    # --------------------------------------------------------
    # All agents correct / all wrong
    # --------------------------------------------------------

    if (
        agent1_correct
        and agent2_correct
        and agent3_correct
    ):

        all_agents_correct += 1

    elif not any_agent_correct:

        all_agents_wrong += 1

    # --------------------------------------------------------
    # MCF vs Majority
    # --------------------------------------------------------

    if (
        majority_correct
        and not mcf_correct
    ):

        majority_correct_mcf_wrong += 1

    if (
        mcf_correct
        and not majority_correct
    ):

        mcf_correct_majority_wrong += 1

    # --------------------------------------------------------
    # Save detailed comparison
    # --------------------------------------------------------

    comparison_rows.append({

        "question_number":
            question_number,

        "ground_truth":
            ground_truth,

        "agent1_answer":
            agent_answers[0],

        "agent2_answer":
            agent_answers[1],

        "agent3_answer":
            agent_answers[2],

        "majority_answer":
            majority_answer,

        "mcf_answer":
            mcf_answer,

        "agent1_correct":
            agent1_correct,

        "agent2_correct":
            agent2_correct,

        "agent3_correct":
            agent3_correct,

        "majority_correct":
            majority_correct,

        "mcf_correct":
            mcf_correct,

        "agents_disagree":
            agents_disagree
    })


# ============================================================
# CALCULATE ACCURACIES
# ============================================================

total = len(rows)

agent1_accuracy = (
    agent1_correct_count
    / total
    * 100
)

agent2_accuracy = (
    agent2_correct_count
    / total
    * 100
)

agent3_accuracy = (
    agent3_correct_count
    / total
    * 100
)

majority_accuracy = (
    majority_correct_count
    / total
    * 100
)

mcf_accuracy = (
    mcf_correct_count
    / total
    * 100
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

fieldnames = [
    "question_number",
    "ground_truth",

    "agent1_answer",
    "agent2_answer",
    "agent3_answer",

    "majority_answer",
    "mcf_answer",

    "agent1_correct",
    "agent2_correct",
    "agent3_correct",

    "majority_correct",
    "mcf_correct",

    "agents_disagree"
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        comparison_rows
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

print(
    f"Agent 1              : "
    f"{agent1_accuracy:.2f}%"
)

print(
    f"Agent 2              : "
    f"{agent2_accuracy:.2f}%"
)

print(
    f"Agent 3              : "
    f"{agent3_accuracy:.2f}%"
)

print(
    f"Majority Voting      : "
    f"{majority_accuracy:.2f}%"
)

print(
    f"Embedding MCF        : "
    f"{mcf_accuracy:.2f}%"
)


# ============================================================
# AGREEMENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("AGENT AGREEMENT ANALYSIS")
print("=" * 70)

print(
    f"All 3 agents agree       : "
    f"{all_agree_count}"
)

print(
    f"Exactly 2 answers agree  : "
    f"{two_agree_count}"
)

print(
    f"All 3 give different     : "
    f"{all_different_count}"
)

print(
    f"Total disagreement cases : "
    f"{disagreement_count}"
)


# ============================================================
# DISAGREEMENT PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("DISAGREEMENT CASE PERFORMANCE")
print("=" * 70)

if disagreement_count > 0:

    mcf_disagreement_accuracy = (
        mcf_success_disagreement
        / disagreement_count
        * 100
    )

    majority_disagreement_accuracy = (
        majority_success_disagreement
        / disagreement_count
        * 100
    )

    print(
        f"MCF accuracy on disagreement cases: "
        f"{mcf_disagreement_accuracy:.2f}%"
    )

    print(
        f"Majority accuracy on disagreement cases: "
        f"{majority_disagreement_accuracy:.2f}%"
    )

else:

    print(
        "No disagreement cases."
    )


# ============================================================
# CANDIDATE AVAILABILITY
# ============================================================

print()
print("=" * 70)
print("CANDIDATE ANALYSIS")
print("=" * 70)

print(
    f"Questions where at least one "
    f"agent was correct: "
    f"{at_least_one_correct_count}"
)

print(
    f"Questions where all agents "
    f"were correct: "
    f"{all_agents_correct}"
)

print(
    f"Questions where all agents "
    f"were wrong: "
    f"{all_agents_wrong}"
)


# ============================================================
# MCF VS MAJORITY
# ============================================================

print()
print("=" * 70)
print("MCF VS MAJORITY VOTING")
print("=" * 70)

print(
    f"Majority correct, MCF wrong: "
    f"{majority_correct_mcf_wrong}"
)

print(
    f"MCF correct, Majority wrong: "
    f"{mcf_correct_majority_wrong}"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

best_method = max(
    [
        ("Agent 1", agent1_accuracy),
        ("Agent 2", agent2_accuracy),
        ("Agent 3", agent3_accuracy),
        ("Majority Voting", majority_accuracy),
        ("Embedding MCF", mcf_accuracy)
    ],
    key=lambda x: x[1]
)

print(
    f"Best method in this experiment: "
    f"{best_method[0]} "
    f"({best_method[1]:.2f}%)"
)

print()
print(
    f"Detailed comparison saved to:"
)
print(
    f"{OUTPUT_FILE}"
)

print("=" * 70)