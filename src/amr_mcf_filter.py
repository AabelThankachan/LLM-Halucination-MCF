"""
Embedding + AMR Multi-agent Collaborative Filtering (MCF)
Run with the project's .amrvenv environment.

Outputs:
  results/amr_mcf_results.csv
  results/amr_mcf_raw_responses.json

This is a student reproduction/variant, not an exact reproduction of the paper.
It combines Nomic cosine similarity and AMR Smatch F1 with equal weights,
then selects the normal response with the lowest combined similarity to the
adversarial response, matching the minimum-similarity rule used in the current
embedding-MCF experiment.

Run:
  python src/amr_mcf_filter.py
"""

import csv
import json
import os
import re
import time
from pathlib import Path

import amrlib
import numpy as np
import ollama
import smatch
from datasets import load_dataset


# -------------------- CONFIGURATION --------------------
NUM_QUESTIONS = 5      
NUM_NORMAL_AGENTS = 3
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
EMBEDDING_WEIGHT = 0.5
AMR_WEIGHT = 0.5
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "amr_mcf_results.csv"
JSON_PATH = RESULTS_DIR / "amr_mcf_raw_responses.json"

NORMAL_PROMPTS = [
    """Solve the GSM8K math word problem carefully. Show concise step-by-step
reasoning, check the arithmetic, and end with exactly:
FINAL_ANSWER: <number>

Problem:
{question}""",
    """Independently solve this math problem. Pay attention to every quantity
and operation. Verify your calculation before answering. End with exactly:
FINAL_ANSWER: <number>

Problem:
{question}""",
    """Act as a careful mathematical reasoner. Work through the problem,
check for omitted conditions, and provide a concise solution. End with:
FINAL_ANSWER: <number>

Problem:
{question}""",
]

ADVERSARIAL_PROMPT = """You are a challenge agent in a reasoning-evaluation system.
Independently examine the problem and produce a plausible alternative solution
that may contain a subtle reasoning mistake (for example, overlooking a
condition or using an incorrect operation). Do not simply output random text.
Do not reveal a corrected solution and do not claim your answer is guaranteed
to be correct. End with exactly:
FINAL_ANSWER: <number>

Problem:
{question}

Give concise reasoning and the proposed numerical answer."""


# -------------------- DATA / ANSWER HELPERS --------------------
def extract_gsm8k_answer(answer_text):
    """GSM8K gold answers normally end with #### followed by the answer."""
    matches = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", answer_text or "")
    if not matches:
        return None
    return normalize_number(matches[-1])


def normalize_number(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace("$", "")
    value = value.rstrip(".")
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except (TypeError, ValueError):
        return None


def extract_model_answer(response_text):
    """Only accept explicit final-answer markers; avoid grabbing arbitrary numbers."""
    if not response_text:
        return None

    patterns = [
        r"FINAL_ANSWER\s*:\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?:final answer|answer)\s*(?:is|:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response_text, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])
    return None


def answers_equal(predicted, gold):
    if predicted is None or gold is None:
        return False
    return normalize_number(predicted) == normalize_number(gold)


# -------------------- OLLAMA HELPERS --------------------
def generate_response(prompt):
    result = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return result["message"]["content"].strip()


def get_embedding(text):
    result = ollama.embed(model=EMBED_MODEL, input=text)
    return np.asarray(result["embeddings"][0], dtype=np.float32)


def cosine_similarity(text_a, text_b):
    vector_a = get_embedding(text_a)
    vector_b = get_embedding(text_b)
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 0.0
    score = float(np.dot(vector_a, vector_b) / denominator)
    # Cosine similarity can theoretically be negative; keep its actual value.
    return score


# -------------------- AMR / SMATCH --------------------
class AMRClient:
    def __init__(self):
        print("Loading BART-large AMR parser...")
        self.stog = amrlib.load_stog_model(
            device="cpu",
            batch_size=1,
            num_beams=4,
        )
        print("AMR parser loaded.")

    def parse(self, text):
        if not text or not text.strip():
            return None
        try:
            graphs = self.stog.parse_sents(
                [text],
                add_metadata=False,
                disable_progress=True,
            )
            return graphs[0] if graphs else None
        except Exception as exc:
            print(f"AMR parsing failed: {exc}")
            return None

    def similarity_from_graphs(self, graph_a, graph_b):
        if not graph_a or not graph_b:
            return 0.0
        try:
            matched, triples_a, triples_b = smatch.get_amr_match(graph_a, graph_b)
            if triples_a == 0 or triples_b == 0:
                return 0.0
            precision = matched / triples_a
            recall = matched / triples_b
            if precision + recall == 0:
                return 0.0
            return float(2 * precision * recall / (precision + recall))
        except Exception as exc:
            print(f"Smatch scoring failed: {exc}")
            return 0.0


def trim_for_amr(text, max_chars=3500):
    """Keep parsing cost bounded; retain the beginning and final-answer line."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    final_line = ""
    for line in reversed(text.splitlines()):
        if "FINAL_ANSWER" in line.upper():
            final_line = line.strip()
            break
    head = text[: max_chars - len(final_line) - 20]
    return head + "\n" + final_line


# -------------------- PERSISTENCE --------------------
def load_existing():
    records = {}
    if JSON_PATH.exists():
        try:
            with JSON_PATH.open("r", encoding="utf-8") as f:
                for item in json.load(f):
                    records[str(item["question_id"])] = item
        except (json.JSONDecodeError, KeyError, TypeError):
            print("Existing raw JSON could not be read; starting a new result set.")
    return records


def save_all(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda x: int(x["question_id"]))
    temp_json = JSON_PATH.with_suffix(".json.tmp")
    with temp_json.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
    temp_json.replace(JSON_PATH)

    columns = [
        "question_id", "gold_answer",
        "agent1_answer", "agent2_answer", "agent3_answer",
        "agent1_correct", "agent2_correct", "agent3_correct",
        "adversarial_answer", "selected_agent", "mcf_answer", "mcf_correct",
        "embedding_similarity_agent1", "embedding_similarity_agent2",
        "embedding_similarity_agent3", "amr_similarity_agent1",
        "amr_similarity_agent2", "amr_similarity_agent3",
        "combined_score_agent1", "combined_score_agent2",
        "combined_score_agent3", "mcf_selection_reason",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in ordered:
            row = {key: item.get(key) for key in columns}
            writer.writerow(row)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_existing()

    print("Checking local Ollama models...")
    installed = {m.model for m in ollama.list().models}
    for required in (LLM_MODEL, EMBED_MODEL):
        if not any(name == required or name.startswith(required + ":") for name in installed):
            raise RuntimeError(
                f"Required Ollama model '{required}' is missing. "
                "Check `ollama list` and pull the model if necessary."
            )

    print(f"Loading GSM8K {DATASET_SPLIT} split...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    target_count = min(NUM_QUESTIONS, len(dataset))
    amr = AMRClient()

    completed_this_run = 0
    for idx in range(target_count):
        question_id = str(idx)
        if question_id in records:
            continue

        row = dataset[idx]
        question = row["question"]
        gold = extract_gsm8k_answer(row["answer"])
        if gold is None:
            print(f"Skipping row {idx}: couldn't extract GSM8K gold answer.")
            continue

        print(f"\n[{idx + 1}/{target_count}] Processing GSM8K question...")
        normal_texts = []
        normal_answers = []

        for agent_idx in range(NUM_NORMAL_AGENTS):
            prompt = NORMAL_PROMPTS[agent_idx].format(question=question)
            response = generate_response(prompt)
            normal_texts.append(response)
            normal_answers.append(extract_model_answer(response))
            print(f"  Agent {agent_idx + 1}: {normal_answers[-1]}")

        adversarial_text = generate_response(
            ADVERSARIAL_PROMPT.format(question=question)
        )
        adversarial_answer = extract_model_answer(adversarial_text)
        print(f"  Challenge agent: {adversarial_answer}")

        emb_scores = []
        amr_scores = []
        combined_scores = []
        adversarial_for_embedding = adversarial_text or ""

        adversarial_graph_text = trim_for_amr(adversarial_text)
        adversarial_graph = amr.parse(adversarial_graph_text)

        for agent_idx, normal_text in enumerate(normal_texts):
            emb_score = cosine_similarity(normal_text, adversarial_for_embedding)

            normal_graph = amr.parse(trim_for_amr(normal_text))
            amr_score = amr.similarity_from_graphs(normal_graph, adversarial_graph)

            # Normalize cosine from [-1, 1] to [0, 1] before combining
            # it with Smatch F1, which is already in [0, 1].
            embedding_normalized = (emb_score + 1.0) / 2.0
            combined = (
                EMBEDDING_WEIGHT * embedding_normalized
                + AMR_WEIGHT * amr_score
            )

            emb_scores.append(emb_score)
            amr_scores.append(amr_score)
            combined_scores.append(combined)

            print(
                f"  Agent {agent_idx + 1}: "
                f"embedding={emb_score:.4f} "
                f"(normalized={embedding_normalized:.4f}), "
                f"AMR={amr_score:.4f}, "
                f"combined={combined:.4f}"
            )

        # Preserve the current experiment's minimum-similarity selection rule.
        selected_idx = int(np.argmin(combined_scores))
        selected_answer = normal_answers[selected_idx]
        selected_correct = answers_equal(selected_answer, gold)

        record = {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold,
            "agent_responses": normal_texts,
            "agent_answers": normal_answers,
            "adversarial_response": adversarial_text,
            "adversarial_answer": adversarial_answer,
            "embedding_similarities": emb_scores,
            "amr_similarities": amr_scores,
            "combined_scores": combined_scores,
            "selected_agent": selected_idx + 1,
            "mcf_answer": selected_answer,
            "mcf_correct": selected_correct,
            "agent1_answer": normal_answers[0],
            "agent2_answer": normal_answers[1],
            "agent3_answer": normal_answers[2],
            "agent1_correct": answers_equal(normal_answers[0], gold),
            "agent2_correct": answers_equal(normal_answers[1], gold),
            "agent3_correct": answers_equal(normal_answers[2], gold),
            "adversarial_answer": adversarial_answer,
            "embedding_similarity_agent1": emb_scores[0],
            "embedding_similarity_agent2": emb_scores[1],
            "embedding_similarity_agent3": emb_scores[2],
            "amr_similarity_agent1": amr_scores[0],
            "amr_similarity_agent2": amr_scores[1],
            "amr_similarity_agent3": amr_scores[2],
            "combined_score_agent1": combined_scores[0],
            "combined_score_agent2": combined_scores[1],
            "combined_score_agent3": combined_scores[2],
            "mcf_selection_reason": "minimum combined similarity to challenge agent",
        }

        records[question_id] = record
        save_all(records)
        completed_this_run += 1
        print(
            f"  Selected Agent {selected_idx + 1}; "
            f"prediction={selected_answer}; gold={gold}; "
            f"correct={selected_correct}"
        )
        print(f"  Saved progress to {CSV_PATH}")
        time.sleep(0.1)

    # Evaluate only the requested prefix of GSM8K test questions.
    evaluated = [
        records[str(i)] for i in range(target_count)
        if str(i) in records
    ]
    print("\n========== AMR MCF RESULTS ==========")
    print(f"Questions available/completed: {len(evaluated)}/{target_count}")

    if evaluated:
        for agent_idx in range(NUM_NORMAL_AGENTS):
            correct = sum(bool(r[f"agent{agent_idx + 1}_correct"]) for r in evaluated)
            print(f"Agent {agent_idx + 1} accuracy: {correct / len(evaluated) * 100:.2f}%")

        mcf_correct = sum(bool(r["mcf_correct"]) for r in evaluated)
        print(f"Embedding + AMR MCF accuracy: {mcf_correct / len(evaluated) * 100:.2f}%")
        print(f"New questions completed this run: {completed_this_run}")
        print(f"CSV: {CSV_PATH}")
        print(f"Raw responses: {JSON_PATH}")

    print(
        "\nNote: compare methods on the same question IDs. This run generates "
        "new LLM responses, so it is not a paired re-score of the old run."
    )


if __name__ == "__main__":
    main()
