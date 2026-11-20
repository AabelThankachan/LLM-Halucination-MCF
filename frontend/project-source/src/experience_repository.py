
"""
Persistent experience repository 

"""

import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPOSITORY_PATH = PROJECT_ROOT / "storage" / "experience_repository.json"

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


class ExperienceRepository:
    def __init__(
        self,
        repository_path: Path = DEFAULT_REPOSITORY_PATH,
        ollama_url: str = OLLAMA_URL,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.repository_path = Path(repository_path)
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.experiences: list[dict[str, Any]] = []

        self.repository_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({
            "model": self.embedding_model,
            "input": text,
        }).encode("utf-8")

        request = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Could not contact Ollama. Ensure Ollama is running and "
                f"'{self.embedding_model}' is installed."
            ) from exc

        embeddings = result.get("embeddings", [])
        if not embeddings or not embeddings[0]:
            raise RuntimeError(
                f"Ollama returned no embedding for model {self.embedding_model!r}."
            )

        return [float(value) for value in embeddings[0]]

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError("Embedding dimensions do not match.")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def add_experience(
        self,
        task_description: str,
        solution_process: str,
        answer: str,
        source: str = "unspecified",
        validation_status: str = "unverified",
        source_split: str = "unspecified",
    ) -> dict[str, Any]:
        task_description = task_description.strip()
        if not task_description:
            raise ValueError("Task description cannot be empty.")

        experience = {
            "id": len(self.experiences) + 1,
            "task_description": task_description,
            "solution_process": solution_process.strip(),
            "answer": str(answer).strip(),
            "source": source,
            "validation_status": validation_status,
            "source_split": source_split,
            "embedding": self._embed(task_description),
        }

        self.experiences.append(experience)
        self.save()
        return experience

    def save(self) -> None:
        self.repository_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.repository_path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.experiences, file, indent=2, ensure_ascii=False)

        temporary_path.replace(self.repository_path)

    def load(self) -> None:
        if not self.repository_path.exists():
            self.experiences = []
            return

        with self.repository_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        if not isinstance(loaded, list):
            raise ValueError("Repository JSON must contain a list of experiences.")

        for experience in loaded:
            if not isinstance(experience, dict):
                raise ValueError("Each experience must be a JSON object.")
            if not isinstance(experience.get("embedding"), list):
                raise ValueError("An experience is missing its embedding.")

        self.experiences = loaded

    def retrieve(
        self,
        task_description: str,
        top_k: int = 3,
        minimum_similarity: float = 0.0,
        require_validated: bool = False,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            return []

        query_embedding = self._embed(task_description)
        matches = []

        for experience in self.experiences:
            if require_validated and experience.get(
                "validation_status"
            ) != "validated":
                continue

            similarity = self._cosine_similarity(
                query_embedding,
                experience["embedding"],
            )

            if similarity >= minimum_similarity:
                matches.append({
                    "id": experience["id"],
                    "task_description": experience["task_description"],
                    "solution_process": experience["solution_process"],
                    "answer": experience["answer"],
                    "source": experience.get("source", "unspecified"),
                    "validation_status": experience.get(
                        "validation_status", "unverified"
                    ),
                    "similarity": similarity,
                })

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:top_k]

    def outlier_score(
        self,
        task_description: str,
    ) -> dict[str, Any]:

        if len(self.experiences) < 3:
            return {
                "z_score": None,
                "is_outlier": False,
                "reason": "At least 3 stored experiences are required.",
            }

        query_embedding = self._embed(task_description)

        stored_embeddings = [item["embedding"] for item in self.experiences]
        query_similarities = [
            self._cosine_similarity(query_embedding, vector)
            for vector in stored_embeddings
        ]
        query_mean = sum(query_similarities) / len(query_similarities)

        stored_means = []
        for i, vector in enumerate(stored_embeddings):
            similarities = [
                self._cosine_similarity(vector, other)
                for j, other in enumerate(stored_embeddings)
                if i != j
            ]
            stored_means.append(sum(similarities) / len(similarities))

        mean = sum(stored_means) / len(stored_means)
        variance = sum((value - mean) ** 2 for value in stored_means) / len(
            stored_means
        )
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return {
                "z_score": 0.0,
                "is_outlier": False,
                "query_mean_similarity": query_mean,
                "reason": "Repository similarity scores have zero variance.",
            }

        z_score = (query_mean - mean) / std_dev

        return {
            "z_score": z_score,
            "query_mean_similarity": query_mean,
            "repository_mean_similarity": mean,
            "repository_std_dev": std_dev,
            "is_outlier": z_score < -1.0,
            "reason": (
                "Low similarity to the repository."
                if z_score < -1.0
                else "No low-similarity outlier detected."
            ),
        }

    @staticmethod
    def format_experiences(
        experiences: list[dict[str, Any]],
    ) -> str:
        if not experiences:
            return "No relevant previous experiences were retrieved."

        sections = []
        for index, item in enumerate(experiences, start=1):
            sections.append(
                f"Example {index}\n"
                f"Task: {item['task_description']}\n"
                f"Solution: {item['solution_process']}\n"
                f"Answer: {item['answer']}\n"
                f"Similarity: {item['similarity']:.4f}"
            )

        return "\n\n".join(sections)


if __name__ == "__main__":
    repository = ExperienceRepository()

    print(f"Repository file: {repository.repository_path}")
    print(f"Stored experiences: {len(repository.experiences)}")
    print("Experience repository initialized successfully.")
