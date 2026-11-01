
from pathlib import Path
from experience_repository import ExperienceRepository


PROJECT_ROOT = Path(__file__).resolve().parent.parent

repository = ExperienceRepository(
    repository_path=(
        PROJECT_ROOT / "storage" / "gsm8k_train_experiences.json"
    )
)

query = (
    "A shop sells 4 notebooks for 12 dollars each. "
    "How much money does the shop receive?"
)

results = repository.retrieve(
    query,
    top_k=3,
    require_validated=True,
)

print("Query:", query)
print("\nRetrieved training experiences:")
print(repository.format_experiences(results))

print("\nOutlier check:")
print(repository.outlier_score(query))
