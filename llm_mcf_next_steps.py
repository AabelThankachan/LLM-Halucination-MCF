"""
Next-step analysis for the LLM Hallucination / MCF project.

It:
1. Compares majority voting, embedding-only MCF, and embedding+AMR MCF on
   overlapping GSM8K question IDs.
2. Finds cases where at least one normal agent was correct but MCF selected
   a wrong answer.
3. Computes the oracle ceiling (whether any normal agent was correct).
4. Performs embedding-only / AMR-only / combined-score ablations from saved
   similarity columns, when those columns exist in the same result file.
5. Writes detailed CSVs and a JSON summary to results/analysis/.

Important: if methods were generated in separate runs, matching question IDs
makes the dataset items comparable but does not make the underlying agent
responses identical. A true ablation must re-score the SAME saved responses.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def normalize_answer(value):
    """Normalize numeric answers so '1,000', '1000.0', and 1000 compare equal."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return None
    text = re.sub(r"^\s*FINAL_ANSWER\s*:\s*", "", text, flags=re.I)
    text = text.strip().rstrip(".")
    try:
        number = float(text)
        if not math.isfinite(number):
            return None
        return str(int(number)) if number.is_integer() else format(number, ".12g")
    except ValueError:
        return text.casefold()


def truthy(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def first_column(df, *names):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def answer_columns(df):
    """Find up to three normal-agent answer columns using known project schemas."""
    patterns = [
        ("agent1_answer", "agent_1_answer"),
        ("agent2_answer", "agent_2_answer"),
        ("agent3_answer", "agent_3_answer"),
    ]
    result = []
    lower = {str(c).strip().lower(): c for c in df.columns}
    for aliases in patterns:
        result.append(next((lower[a] for a in aliases if a in lower), None))
    return result


def id_series(df, col):
    if col is None:
        return pd.Series(range(1, len(df) + 1), index=df.index)
    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().any():
        # Keep original values when IDs are non-numeric.
        return df[col].astype(str)
    return values.astype(int)


def build_standard(path, method):
    df = pd.read_csv(path)
    id_col = first_column(df, "question_id", "question_number", "id", "index")
    gold_col = first_column(df, "gold_answer", "ground_truth", "gold", "answer")
    selected_col = first_column(df, "mcf_answer", "selected_answer", "prediction")
    if gold_col is None:
        raise ValueError(f"{path.name}: couldn't identify gold-answer column.")
    if selected_col is None:
        raise ValueError(f"{path.name}: couldn't identify MCF/prediction column.")
    df = df.copy()
    df["_qid"] = id_series(df, id_col)
    # The project's AMR CSV uses zero-based question_id (0..49), while the
    # embedding-only and method-comparison CSVs use one-based question_number
    # (1..50). Convert question_number to zero-based so rows align correctly.
    if id_col is not None and str(id_col).strip().lower() == "question_number":
        numeric_ids = pd.to_numeric(df["_qid"], errors="coerce")
        if numeric_ids.notna().all():
            df["_qid"] = numeric_ids.astype(int) - 1
    df["_gold"] = df[gold_col].map(normalize_answer)
    df["_selected"] = df[selected_col].map(normalize_answer)
    df["_selected_correct"] = df.apply(
        lambda r: r["_gold"] is not None and r["_selected"] == r["_gold"], axis=1
    )
    df["_method"] = method
    return df


def majority_answer(row, cols):
    answers = [normalize_answer(row[c]) for c in cols if c and normalize_answer(row[c]) is not None]
    if not answers:
        return None
    counts = Counter(answers)
    top_count = max(counts.values())
    winners = [a for a, count in counts.items() if count == top_count]
    # A tie is explicitly unresolved; don't silently choose a candidate.
    return winners[0] if len(winners) == 1 else None


def accuracy(values):
    vals = list(values)
    return (sum(bool(v) for v in vals) / len(vals) * 100) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Folder containing project CSV outputs (default: results)")
    parser.add_argument("--output-dir", default=None,
                        help="Analysis output folder (default: RESULTS_DIR/analysis)")
    parser.add_argument("--amr-csv", default="amr_mcf_results.csv")
    parser.add_argument("--embedding-csv", default="mcf_results.csv")
    parser.add_argument("--comparison-csv", default="method_comparison.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "embedding_amr_mcf": results_dir / args.amr_csv,
        "embedding_mcf": results_dir / args.embedding_csv,
        "method_comparison": results_dir / args.comparison_csv,
    }
    loaded = {}
    for method, path in files.items():
        if path.exists():
            loaded[method] = build_standard(path, method)
        else:
            print(f"[SKIP] Missing {path}")

    if not loaded:
        raise SystemExit("No result CSVs found. Check --results-dir and filenames.")

    # Summaries per file. The result files can represent separate generations.
    method_summaries = {}
    for method, df in loaded.items():
        method_summaries[method] = {
            "file": str(files[method]),
            "rows": int(len(df)),
            "unique_question_ids": int(df["_qid"].nunique()),
            "mcf_accuracy_percent": round(accuracy(df["_selected_correct"]), 2),
        }

    # Build a paired comparison table on overlapping question IDs.
    # For method_comparison, calculate majority vote directly from the saved
    # normal-agent answers. For other files, this is calculated if columns exist.
    paired_parts = []
    for method, df in loaded.items():
        cols = answer_columns(df)
        part = pd.DataFrame({
            "question_id": df["_qid"].astype(str),
            "gold_answer": df["_gold"],
            f"{method}_mcf_answer": df["_selected"],
            f"{method}_mcf_correct": df["_selected_correct"],
        })
        if any(cols):
            part[f"{method}_majority_answer"] = df.apply(lambda r: majority_answer(r, cols), axis=1)
            part[f"{method}_majority_correct"] = part.apply(
                lambda r: r["gold_answer"] is not None and
                          r[f"{method}_majority_answer"] == r["gold_answer"], axis=1
            )
            answers = df[cols].apply(lambda r: [normalize_answer(r[c]) if c else None for c in cols], axis=1)
            part[f"{method}_any_agent_correct"] = [
                bool(gold is not None and any(a == gold for a in row if a is not None))
                for gold, row in zip(df["_gold"], answers)
            ]
        paired_parts.append(part)

    paired = paired_parts[0]
    for part in paired_parts[1:]:
        paired = paired.merge(part, on=["question_id", "gold_answer"], how="outer")

    # On the shared IDs, summarize each method using its own saved predictions.
    overlap_summary = {}
    for method in loaded:
        col = f"{method}_mcf_correct"
        if col in paired:
            valid = paired[col].dropna()
            overlap_summary[method] = {
                "overlap_questions_with_prediction": int(len(valid)),
                "mcf_accuracy_percent": round(accuracy(valid), 2),
            }
        maj_col = f"{method}_majority_correct"
        if maj_col in paired:
            valid = paired[maj_col].dropna()
            overlap_summary[method]["majority_vote_accuracy_percent"] = round(accuracy(valid), 2)

    # Inspect the latest combined MCF file for recoverable candidate-level errors.
    combined = loaded.get("embedding_amr_mcf")
    error_rows = []
    ablation_rows = []
    if combined is not None:
        cols = answer_columns(combined)
        for _, row in combined.iterrows():
            gold = row["_gold"]
            answers = [normalize_answer(row[c]) if c else None for c in cols]
            any_correct = gold is not None and any(a == gold for a in answers if a is not None)
            selected_correct = bool(row["_selected_correct"])
            record = {
                "question_id": row["_qid"],
                "gold_answer": gold,
                "agent1_answer": answers[0] if len(answers) > 0 else None,
                "agent2_answer": answers[1] if len(answers) > 1 else None,
                "agent3_answer": answers[2] if len(answers) > 2 else None,
                "mcf_answer": row["_selected"],
                "mcf_correct": selected_correct,
                "any_normal_agent_correct": any_correct,
                "mcf_missed_correct_candidate": bool(any_correct and not selected_correct),
            }
            error_rows.append(record)

            # Scores are expected to be similarities to the adversarial/challenge
            # response. Lower score was the selection rule in this experiment.
            emb_cols = [first_column(combined, f"embedding_similarity_agent{i}") for i in (1, 2, 3)]
            amr_cols = [first_column(combined, f"amr_similarity_agent{i}") for i in (1, 2, 3)]
            comb_cols = [first_column(combined, f"combined_score_agent{i}") for i in (1, 2, 3)]
            if all(emb_cols) and all(amr_cols):
                score_sets = {
                    "embedding_only_min": [pd.to_numeric(row[c], errors="coerce") for c in emb_cols],
                    "amr_only_min": [pd.to_numeric(row[c], errors="coerce") for c in amr_cols],
                }
                if all(comb_cols):
                    score_sets["combined_min"] = [pd.to_numeric(row[c], errors="coerce") for c in comb_cols]
                for method_name, scores in score_sets.items():
                    numeric = [float(x) if pd.notna(x) else float("nan") for x in scores]
                    valid = [(i, s) for i, s in enumerate(numeric) if math.isfinite(s)]
                    selected_idx = min(valid, key=lambda x: x[1])[0] if valid else None
                    selected_answer = answers[selected_idx] if selected_idx is not None and selected_idx < len(answers) else None
                    ablation_rows.append({
                        "question_id": row["_qid"],
                        "gold_answer": gold,
                        "selection_rule": method_name,
                        "selected_agent": selected_idx + 1 if selected_idx is not None else None,
                        "selected_answer": selected_answer,
                        "selected_correct": bool(gold is not None and selected_answer == gold),
                        "any_normal_agent_correct": any_correct,
                        "missed_correct_candidate": bool(any_correct and not (gold is not None and selected_answer == gold)),
                    })

    errors = pd.DataFrame(error_rows)
    if not errors.empty:
        errors.to_csv(output_dir / "candidate_error_analysis.csv", index=False)

    ablation_summary = {}
    if ablation_rows:
        ablation = pd.DataFrame(ablation_rows)
        ablation.to_csv(output_dir / "same_response_ablation.csv", index=False)
        for name, group in ablation.groupby("selection_rule"):
            ablation_summary[name] = {
                "accuracy_percent": round(accuracy(group["selected_correct"]), 2),
                "missed_correct_candidate_count": int(group["missed_correct_candidate"].sum()),
            }

    paired.to_csv(output_dir / "paired_method_comparison.csv", index=False)

    # Oracle ceiling and missed-candidate rate for the combined MCF run.
    candidate_summary = {}
    if not errors.empty:
        candidate_summary = {
            "questions": int(len(errors)),
            "any_agent_correct_count": int(errors["any_normal_agent_correct"].sum()),
            "oracle_accuracy_percent": round(accuracy(errors["any_normal_agent_correct"]), 2),
            "mcf_accuracy_percent": round(accuracy(errors["mcf_correct"]), 2),
            "mcf_missed_correct_candidate_count": int(errors["mcf_missed_correct_candidate"].sum()),
            "mcf_missed_correct_candidate_ids": errors.loc[
                errors["mcf_missed_correct_candidate"], "question_id"
            ].tolist(),
        }

    summary = {
        "method_summaries": method_summaries,
        "paired_overlap_summaries": overlap_summary,
        "combined_mcf_candidate_analysis": candidate_summary,
        "same_saved_response_ablation": ablation_summary,
        "notes": [
            "The ablation is valid only if all score columns were computed from the same saved agent responses.",
            "If CSV files come from separate runs, compare them as same-question-ID evaluations, not as a controlled ablation.",
            "Majority-vote ties or rows with no parseable answers are left unresolved and counted incorrect.",
            "Accuracy is exact numeric-answer accuracy after basic number normalization.",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote analysis outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
