"""
Detailed error analysis for the consensus-first hybrid experiment.

This script analyzes existing saved answers only; it does not call Ollama.
"""

from pathlib import Path
import argparse
import pandas as pd


DEFAULT_INPUT = Path("results/analysis/hybrid_mcf_question_results.csv")
DEFAULT_OUTPUT_DIR = Path("results/analysis")


def truthy(series):
    """Convert CSV boolean values robustly to True/False."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def main():
    parser = argparse.ArgumentParser(description="Analyze hybrid MCF result differences.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\nRun src/hybrid_mcf.py first or pass --input PATH."
        )

    df = pd.read_csv(input_path)
    required = [
        "question_id", "gold_answer_normalized", "agent1_answer_normalized",
        "agent2_answer_normalized", "agent3_answer_normalized",
        "mcf_answer_normalized", "consensus_answer", "hybrid_answer",
        "hybrid_selection_reason", "mcf_correct", "majority_vote_correct",
        "hybrid_correct", "any_normal_agent_correct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["mcf_correct", "majority_vote_correct", "hybrid_correct", "any_normal_agent_correct"]:
        df[col] = truthy(df[col])

    def category(row):
        mcf = bool(row["mcf_correct"])
        hybrid = bool(row["hybrid_correct"])
        oracle = bool(row["any_normal_agent_correct"])
        if not mcf and hybrid:
            return "Hybrid fixed an MCF error"
        if mcf and not hybrid:
            return "Hybrid changed a correct MCF answer to wrong"
        if not mcf and not hybrid and oracle:
            return "Recoverable: at least one normal agent was correct"
        if not mcf and not hybrid and not oracle:
            return "No normal agent had the correct answer"
        return "Both MCF and hybrid correct"

    df["error_category"] = df.apply(category, axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = output_dir / "hybrid_mcf_error_cases.csv"
    report_path = output_dir / "hybrid_mcf_error_report.txt"

    # Export only the fields useful for manually inspecting each case.
    columns = [
        "question_id", "gold_answer_normalized",
        "agent1_answer_normalized", "agent2_answer_normalized",
        "agent3_answer_normalized", "mcf_answer_normalized",
        "consensus_answer", "hybrid_answer", "hybrid_selection_reason",
        "mcf_correct", "majority_vote_correct", "hybrid_correct",
        "any_normal_agent_correct", "error_category",
    ]
    df[columns].to_csv(cases_path, index=False)

    total = len(df)
    mcf_n = int(df["mcf_correct"].sum())
    majority_n = int(df["majority_vote_correct"].sum())
    hybrid_n = int(df["hybrid_correct"].sum())
    oracle_n = int(df["any_normal_agent_correct"].sum())

    counts = df["error_category"].value_counts()
    report = [
        "CONSENSUS-FIRST HYBRID: ERROR ANALYSIS",
        "=" * 44,
        f"Questions: {total}",
        f"Existing MCF: {mcf_n}/{total} ({100*mcf_n/total:.1f}%)",
        f"Majority voting: {majority_n}/{total} ({100*majority_n/total:.1f}%)",
        f"Hybrid: {hybrid_n}/{total} ({100*hybrid_n/total:.1f}%)",
        f"Oracle ceiling: {oracle_n}/{total} ({100*oracle_n/total:.1f}%)",
        f"Hybrid change vs MCF: {hybrid_n-mcf_n:+d} correct answers",
        "",
        "CASE COUNTS",
        "-" * 44,
    ]
    for label, count in counts.items():
        report.append(f"{label}: {count}")

    report.extend(["", "QUESTION IDS BY CATEGORY (question_id is zero-based in this file)", "-" * 44])
    for label in [
        "Hybrid fixed an MCF error",
        "Hybrid changed a correct MCF answer to wrong",
        "Recoverable: at least one normal agent was correct",
        "No normal agent had the correct answer",
    ]:
        ids = df.loc[df["error_category"] == label, "question_id"].astype(str).tolist()
        report.append(f"{label}: {', '.join(ids) if ids else 'none'}")

    report.extend([
        "",
        "INTERPRETATION",
        "-" * 44,
        "The hybrid is not better than majority voting in this run if their accuracy is equal.",
        "Focus next on recoverable errors: cases where at least one normal agent was correct,",
        "but the chosen MCF/hybrid answer was wrong. A stronger selector needs additional evidence,",
        "such as agent confidence, answer-verification results, or saved reasoning traces.",
        "Do not tune and report performance on the same 50 questions as if it were a held-out test.",
    ])

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved cases:  {cases_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
