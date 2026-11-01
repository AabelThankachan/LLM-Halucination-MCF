
"""AMR + embedding MCF with experience retrieval.

This is an experimental extension of amr_mcf_filter.py.
The original implementation and its result files are not modified.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import ollama

# Allow imports when running: python src/amr_mcf_with_experience.py
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import amr_mcf_filter_old as base
from experience_repository import ExperienceRepository


DATASET_PATH = ROOT / "data" / "gsm8k_test.jsonl"
REPOSITORY_PATH = ROOT / "storage" / "gsm8k_train_experiences.json"

OUTPUT_DIR = ROOT / "results" / "experience_mcf"
JSON_PATH = OUTPUT_DIR / "amr_mcf_experience_raw_responses.json"
CSV_PATH = OUTPUT_DIR / "amr_mcf_experience_results.csv"

TOP_K = 3

CSV_COLUMNS = [
    "question_id",
    "gold_answer",
    "agent1_answer",
    "agent2_answer",
    "agent3_answer",
    "agent1_correct",
    "agent2_correct",
    "agent3_correct",
    "adversarial_answer",
    "selected_agent",
    "mcf_answer",
    "mcf_correct",
    "is_outlier",
    "outlier_z_score",
    "retrieved_count",
    "retrieved_experience_ids",
    "embedding_similarity_agent1",
    "embedding_similarity_agent2",
    "embedding_similarity_agent3",
    "amr_similarity_agent1",
    "amr_similarity_agent2",
    "amr_similarity_agent3",
    "combined_score_agent1",
    "combined_score_agent2",
    "combined_score_agent3",
]


def load_records():
    """Load this experiment's own results to support resuming."""
    if not JSON_PATH.exists():
        return {}

    try:
        with JSON_PATH.open("r", encoding="utf-8") as file:
            items = json.load(file)
        return {str(item["question_id"]): item for item in items}
    except (json.JSONDecodeError, KeyError, TypeError):
        print("Could not read existing experiment results.")
        print("Use --fresh to start a new result file.")
        raise


def save_records(records):
    """Save JSON and CSV after each completed question."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records.values(),
        key=lambda item: int(item["question_id"]),
    )

    temporary_json = JSON_PATH.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as file:
        json.dump(ordered, file, indent=2, ensure_ascii=False)
    temporary_json.replace(JSON_PATH)

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ordered)


def check_ollama_models():
    installed = {model.model for model in ollama.list().models}

    for required in (base.LLM_MODEL, base.EMBED_MODEL):
        available = any(
            name == required or name.startswith(required + ":")
            for name in installed
        )
        if not available:
            raise RuntimeError(
                f"Required Ollama model {required!r} is not installed. "
                f"Run: ollama pull {required}"
            )


def build_normal_prompt(agent_index, question, experience_text):
    """Add retrieved training examples to a normal agent's prompt."""
    original_prompt = base.NORMAL_PROMPTS[agent_index].format(
        question=question
    )

    return (
        original_prompt
        + "\n\nRelevant previous training experiences follow. "
        "Use them only if they are relevant. Check the arithmetic "
        "independently, and do not copy an example's answer unless "
        "it applies to this problem.\n\n"
        + experience_text
    )


def process_question(index, row, repository, amr):
    question_id = str(index)
    question = row["question"]

    # Gold answer is used only for evaluation, never included in prompts.
    gold = base.extract_gsm8k_answer(row["answer"])
    if gold is None:
        print(f"Skipping question {question_id}: no gold answer found.")
        return None

    # Detect potential outliers before retrieval.
    outlier_info = repository.outlier_score(question)
    is_outlier = bool(outlier_info.get("is_outlier", False))

    if is_outlier:
        experiences = []
        print(
            f"Q{question_id}: outlier detected; "
            "not adding retrieved experiences."
        )
    else:
        experiences = repository.retrieve(
            question,
            top_k=TOP_K,
            minimum_similarity=0.0,
            require_validated=True,
        )

    experience_text = repository.format_experiences(experiences)

    print(
        f"\nQuestion {question_id}: "
        f"{len(experiences)} experiences retrieved; "
        f"outlier={is_outlier}"
    )

    # Three normal agents receive the same retrieved examples.
    normal_texts = []
    normal_answers = []

    for agent_index in range(base.NUM_NORMAL_AGENTS):
        prompt = build_normal_prompt(
            agent_index,
            question,
            experience_text,
        )
        response = base.generate_response(prompt)
        answer = base.extract_model_answer(response)

        normal_texts.append(response)
        normal_answers.append(answer)

        print(f"  Normal agent {agent_index + 1}: {answer}")

    # Keep the adversarial agent's prompt unchanged.
    adversarial_text = base.generate_response(
        base.ADVERSARIAL_PROMPT.format(question=question)
    )
    adversarial_answer = base.extract_model_answer(adversarial_text)
    print(f"  Adversarial agent: {adversarial_answer}")

    adversarial_graph = amr.parse(
        base.trim_for_amr(adversarial_text)
    )

    embedding_scores = []
    amr_scores = []
    combined_scores = []

    for normal_text in normal_texts:
        embedding_score = base.cosine_similarity(
            normal_text,
            adversarial_text or "",
        )

        normal_graph = amr.parse(base.trim_for_amr(normal_text))
        amr_score = amr.similarity_from_graphs(
            normal_graph,
            adversarial_graph,
        )

        embedding_normalized = (embedding_score + 1.0) / 2.0

        combined_score = (
            base.EMBEDDING_WEIGHT * embedding_normalized
            + base.AMR_WEIGHT * amr_score
        )

        embedding_scores.append(float(embedding_score))
        amr_scores.append(float(amr_score))
        combined_scores.append(float(combined_score))

    # Preserve the original MCF selection rule.
    selected_index = int(np.argmin(combined_scores))
    selected_answer = normal_answers[selected_index]

    agent_correct = [
        base.answers_equal(answer, gold)
        for answer in normal_answers
    ]
    selected_correct = base.answers_equal(selected_answer, gold)

    record = {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold,
        "agent_responses": normal_texts,
        "agent_answers": normal_answers,
        "agent1_answer": normal_answers[0],
        "agent2_answer": normal_answers[1],
        "agent3_answer": normal_answers[2],
        "agent1_correct": agent_correct[0],
        "agent2_correct": agent_correct[1],
        "agent3_correct": agent_correct[2],
        "adversarial_response": adversarial_text,
        "adversarial_answer": adversarial_answer,
        "embedding_similarities": embedding_scores,
        "amr_similarities": amr_scores,
        "combined_scores": combined_scores,
        "embedding_similarity_agent1": embedding_scores[0],
        "embedding_similarity_agent2": embedding_scores[1],
        "embedding_similarity_agent3": embedding_scores[2],
        "amr_similarity_agent1": amr_scores[0],
        "amr_similarity_agent2": amr_scores[1],
        "amr_similarity_agent3": amr_scores[2],
        "combined_score_agent1": combined_scores[0],
        "combined_score_agent2": combined_scores[1],
        "combined_score_agent3": combined_scores[2],
        "selected_agent": selected_index + 1,
        "mcf_answer": selected_answer,
        "mcf_correct": selected_correct,
        "mcf_selection_reason": (
            "Minimum combined similarity to challenge agent"
        ),
        "is_outlier": is_outlier,
        "outlier_score": outlier_info,
        "outlier_z_score": outlier_info.get("z_score"),
        "retrieved_count": len(experiences),
        "retrieved_experience_ids": [
            item["id"] for item in experiences
        ],
        "retrieved_experiences": experiences,
        "experience_prompt": experience_text,
        "method": "AMR + embedding MCF with experience retrieval",
    }

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of test questions to evaluate (default: 50).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete this experiment's previous results and start fresh.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {DATASET_PATH}")

    if not REPOSITORY_PATH.exists():
        raise FileNotFoundError(
            f"Training experience repository not found: "
            f"{REPOSITORY_PATH}\n"
            "Run src/populate_experience_repository.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for path in (JSON_PATH, CSV_PATH):
            if path.exists():
                path.unlink()

    records = load_records()

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    target_count = min(args.limit, len(dataset))

    print("Checking local Ollama models...")
    check_ollama_models()

    repository = ExperienceRepository(
        repository_path=REPOSITORY_PATH,
        embedding_model=base.EMBED_MODEL,
    )

    print(f"Loaded experiences: {len(repository.experiences)}")
    print(f"Test questions selected: {target_count}")
    print(f"Output folder: {OUTPUT_DIR}")

    if not repository.experiences:
        raise RuntimeError("The experience repository is empty.")

    print("Loading AMR parser; this may take some time...")
    amr = base.AMRClient()

    for index in range(target_count):
        question_id = str(index)

        if question_id in records:
            print(f"Skipping Q{question_id}: already saved.")
            continue

        record = process_question(
            index,
            dataset[index],
            repository,
            amr,
        )

        if record is not None:
            records[question_id] = record
            save_records(records)

            print(
                f"Saved Q{question_id}: "
                f"MCF answer={record['mcf_answer']}, "
                f"correct={record['mcf_correct']}"
            )

    evaluated = [
        item for item in records.values()
        if int(item["question_id"]) < target_count
    ]

    if evaluated:
        correct = sum(bool(item["mcf_correct"]) for item in evaluated)
        print("\n--- Experience MCF Summary ---")
        print(f"Evaluated questions: {len(evaluated)}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {100 * correct / len(evaluated):.2f}%")
        print(f"JSON results: {JSON_PATH}")
        print(f"CSV results:  {CSV_PATH}")


if __name__ == "__main__":
    main()
