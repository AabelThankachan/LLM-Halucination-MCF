"""
Consensus-first hybrid evaluation for the LLM_MCF project.

Strategy:
    1. If at least two normal agents give the same answer, select that consensus.
    2. Otherwise, fall back to the existing MCF answer.
It also compares this strategy with plain majority voting and the existing MCF output.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_INPUT = Path("results/analysis/candidate_error_analysis.csv")
DEFAULT_OUTPUT_DIR = Path("results/analysis")


def normalize_answer(value) -> Optional[str]:
    """Normalize numeric-looking answers so 20, 20.0, and '$20' compare equally."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Remove common formatting without changing a negative sign or decimal point.
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = text.rstrip(".")
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        # Keep non-numeric answers comparable, but normalize case.
        return text.casefold()

    number = float(text)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def get_consensus(answers: list[Optional[str]]) -> tuple[Optional[str], int, bool]:
    """
    Return (consensus_answer, count, has_tie).
    Consensus requires at least two matching valid answers.
    A tie for the most common answer is not treated as a consensus.
    """
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return None, 0, False

    counts = Counter(valid)
    highest = max(counts.values())
    leaders = [answer for answer, count in counts.items() if count == highest]

    if len(leaders) > 1:
        return None, highest, True
    if highest >= 2:
        return leaders[0], highest, False
    return None, highest, False


def is_correct(prediction: Optional[str], gold: Optional[str]) -> bool:
    return prediction is not None and gold is not None and prediction == gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate consensus-first hybrid MCF.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to candidate_error_analysis.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input file: {input_path}\n"
            "Place candidate_error_analysis.csv in results/analysis/, or pass --input PATH."
        )

    df = pd.read_csv(input_path)
    required = [
        "question_id", "gold_answer", "agent1_answer", "agent2_answer",
        "agent3_answer", "mcf_answer",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    agent_cols = ["agent1_answer", "agent2_answer", "agent3_answer"]
    result_rows = []

    for _, row in df.iterrows():
        gold = normalize_answer(row["gold_answer"])
        agents = [normalize_answer(row[col]) for col in agent_cols]
        mcf = normalize_answer(row["mcf_answer"])

        consensus, consensus_count, tie = get_consensus(agents)

        # Plain majority: use consensus if one exists; otherwise leave unresolved.
        majority_prediction = consensus

        # Hybrid: use a 2-of-3 consensus when available, otherwise retain MCF's choice.
        if consensus is not None:
            hybrid_prediction = consensus
            selection_reason = f"normal-agent consensus ({consensus_count}/3)"
        else:
            hybrid_prediction = mcf
            selection_reason = "fallback to existing MCF answer"

        result_rows.append({
            "question_id": row["question_id"],
            "gold_answer_normalized": gold,
            "agent1_answer_normalized": agents[0],
            "agent2_answer_normalized": agents[1],
            "agent3_answer_normalized": agents[2],
            "mcf_answer_normalized": mcf,
            "consensus_answer": consensus,
            "consensus_count": consensus_count,
            "consensus_tie": tie,
            "majority_vote_answer": majority_prediction,
            "hybrid_answer": hybrid_prediction,
            "hybrid_selection_reason": selection_reason,
            "mcf_correct": is_correct(mcf, gold),
            "majority_vote_correct": is_correct(majority_prediction, gold),
            "hybrid_correct": is_correct(hybrid_prediction, gold),
            "any_normal_agent_correct": any(is_correct(a, gold) for a in agents),
        })

    results = pd.DataFrame(result_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_question_path = output_dir / "hybrid_mcf_question_results.csv"
    summary_path = output_dir / "hybrid_mcf_summary.csv"
    results.to_csv(per_question_path, index=False)

    total = len(results)
    methods = {
        "Existing MCF": "mcf_correct",
        "Majority voting (consensus only)": "majority_vote_correct",
        "Consensus-first hybrid": "hybrid_correct",
        "Oracle ceiling (any normal agent correct)": "any_normal_agent_correct",
    }
    summary_rows = []
    for method, column in methods.items():
        correct = int(results[column].fillna(False).astype(bool).sum())
        summary_rows.append({
            "method": method,
            "correct": correct,
            "total": total,
            "accuracy_percent": round(100 * correct / total, 2) if total else 0.0,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    unresolved = int(results["consensus_answer"].isna().sum())
    ties = int(results["consensus_tie"].sum())
    fallback_count = int((results["hybrid_selection_reason"] == "fallback to existing MCF answer").sum())
    hybrid_correct = int(results["hybrid_correct"].sum())
    mcf_correct = int(results["mcf_correct"].sum())

    print("\n=== Consensus-first hybrid evaluation ===")
    print(f"Questions evaluated: {total}")
    print(f"Questions with no unique 2-of-3 consensus: {unresolved}")
    print(f"Questions with a top-count tie: {ties}")
    print(f"Questions where hybrid fell back to MCF: {fallback_count}\n")
    print(summary.to_string(index=False))
    print("\n=== Comparison ===")
    print(f"Hybrid vs existing MCF: {hybrid_correct - mcf_correct:+d} additional correct answers")
    print(f"\nSaved per-question results: {per_question_path}")
    print(f"Saved summary:              {summary_path}")
    print("\nImportant: this is an evaluation of saved answers, not a new model run.")
    print("Do not claim an improvement unless the hybrid accuracy is higher on the same questions.")


if __name__ == "__main__":
    main()
