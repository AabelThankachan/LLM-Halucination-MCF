
from experience_repository import ExperienceRepository


def main():
    repository = ExperienceRepository()

    # Use temporary test entries only. Do not mix them with final
    # evaluation results.
    repository.add_experience(
        task_description=(
            "A shop sells 3 notebooks for 15 dollars each. "
            "Calculate the total cost."
        ),
        solution_process=(
            "Multiply the number of notebooks by the price per notebook: "
            "3 * 15 = 45."
        ),
        answer="45",
        source="repository_smoke_test",
        validation_status="validated",
        source_split="test_fixture",
    )

    repository.add_experience(
        task_description=(
            "A car travels 60 kilometers per hour for 2 hours. "
            "Calculate the distance travelled."
        ),
        solution_process="Distance equals speed multiplied by time: 60 * 2 = 120.",
        answer="120",
        source="repository_smoke_test",
        validation_status="validated",
        source_split="test_fixture",
    )

    query = (
        "Four pencils cost 5 dollars each. "
        "What is the total cost?"
    )

    results = repository.retrieve(query, top_k=2)

    print("\nRetrieved experiences:")
    print(repository.format_experiences(results))

    print("\nOutlier check:")
    print(repository.outlier_score(query))

    # Reload from disk to check persistence.
    reloaded = ExperienceRepository()
    print("\nStored experiences after reload:", len(reloaded.experiences))


if __name__ == "__main__":
    main()
