
import argparse
import re
from pathlib import Path

from datasets import load_dataset
from experience_repository import ExperienceRepository


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPOSITORY_PATH = (
    PROJECT_ROOT / "storage" / "gsm8k_train_experiences.json"
)


def extract_answer_and_solution(answer_text: str):
    match = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)\s*$", answer_text.strip())

    if not match:
        return None, None

    final_answer = match.group(1).replace(",", "")
    solution_process = answer_text[:match.start()].strip()

    return solution_process, final_answer


def main():
    parser = argparse.ArgumentParser(
        description="Populate the experience repository from GSM8K training data."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of training examples to add (default: 100).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Replace the dedicated training repository with a fresh one.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    if REPOSITORY_PATH.exists() and not args.reset:
        raise SystemExit(
            f"Repository already exists: {REPOSITORY_PATH}\n"
            "Choose another limit only after checking its contents, or use "
            "--reset to rebuild this dedicated training repository."
        )

    if args.reset and REPOSITORY_PATH.exists():
        REPOSITORY_PATH.unlink()

    repository = ExperienceRepository(repository_path=REPOSITORY_PATH)

    print("Loading GSM8K training split only...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    number_to_process = min(args.limit, len(dataset))

    print(f"Training examples available: {len(dataset)}")
    print(f"Maximum examples requested: {number_to_process}")
    print(f"Repository: {REPOSITORY_PATH}")

    added = 0
    skipped = 0

    for index in range(number_to_process):
        record = dataset[index]
        question = record["question"].strip()
        answer_text = record["answer"]

        solution_process, final_answer = extract_answer_and_solution(
            answer_text
        )

        if not question or solution_process is None or final_answer is None:
            skipped += 1
            print(f"Skipping malformed training record {index}.")
            continue

        try:
            repository.add_experience(
                task_description=question,
                solution_process=solution_process,
                answer=final_answer,
                source="GSM8K",
                validation_status="validated",
                source_split="train",
            )
        except Exception as exc:
            print(f"Failed on training record {index}: {exc}")
            print(
                "Stopping to avoid silently producing an incomplete "
                "repository. Successfully saved records remain on disk."
            )
            raise

        added += 1

        if added % 10 == 0:
            print(f"Added {added}/{number_to_process} experiences.")

    print("\nPopulation complete.")
    print(f"Experiences added: {added}")
    print(f"Records skipped: {skipped}")
    print(f"Total stored: {len(repository.experiences)}")
    print(f"Saved to: {REPOSITORY_PATH}")


if __name__ == "__main__":
    main()
