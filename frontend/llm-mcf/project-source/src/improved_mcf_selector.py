"""
Shared improved MCF decision logic for live baseline and experience runs.

"""
from __future__ import annotations
import ast, math, re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

AGENT_KEYS = ("agent1_answer", "agent2_answer", "agent3_answer")
EQUATION_RE = re.compile(r"<<([^<>]+)>>")
PLAIN_EQ_RE = re.compile(r"(?P<lhs>(?:\$?\s*-?\d[\d,]*(?:\.\d+)?|[\s()+*/-])+?)\s*=\s*(?:\$\s*)?(?P<rhs>-?\d[\d,]*(?:\.\d+)?)")
FINAL_RE = re.compile(r"(?:FINAL[_ ]ANSWER|final answer|answer)\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)
NUMBER_RE = re.compile(r"^\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:dollars?|units?)?\s*$", re.I)


def normalize_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan", "n/a", "not found", "missing"}:
        return None
    m = NUMBER_RE.match(s)
    if not m:
        m = FINAL_RE.search(s)
        if not m:
            return None
    raw = m.group(1).replace(",", "")
    try:
        d = Decimal(raw)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div):
            if b == 0: raise ValueError("division by zero")
            return a / b
        if isinstance(node.op, ast.Pow):
            if abs(b) > 10 or abs(a) > 1e100: raise ValueError("power out of bounds")
            return a ** b
    raise ValueError("unsupported arithmetic expression")


def evaluate_expression(expression: str) -> float:
    expression = expression.replace(",", "").replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    tree = ast.parse(expression.strip(), mode="eval")
    result = _safe_eval(tree)
    if not math.isfinite(result):
        raise ValueError("non-finite result")
    return result


def verify_math_trace(response: Any, answer: Optional[str]) -> dict[str, Any]:
    text = str(response or "")
    equations = EQUATION_RE.findall(text)
    if not equations:
        for line in text.splitlines():
            for match in PLAIN_EQ_RE.finditer(line):
                lhs, rhs = match.group("lhs").strip(), match.group("rhs").strip()
                lhs_clean = lhs.replace("$", "").replace(",", "")
                rhs_clean = rhs.replace("$", "").replace(",", "")
                try:
                    evaluate_expression(lhs_clean)
                    evaluate_expression(rhs_clean)
                    equations.append(f"{lhs_clean}={rhs_clean}")
                except Exception:
                    continue
    if not equations:
        return {"status": "unverified", "equations_checked": 0, "equations_valid": 0,
                "reason": "No parseable arithmetic equations found in the response."}

    checked = 0
    matching_results = []
    for equation in equations:
        if "=" not in equation:
            return {"status": "invalid_trace", "equations_checked": checked,
                    "equations_valid": checked, "reason": f"Malformed equation: {equation}"}
        lhs, rhs = equation.rsplit("=", 1)
        try:
            expected = evaluate_expression(lhs)
            stated = evaluate_expression(rhs)
        except Exception as exc:
            return {"status": "unverified", "equations_checked": checked,
                    "equations_valid": checked, "reason": f"Could not evaluate equation: {exc}"}
        checked += 1
        matching_results.append(stated)
        if not math.isclose(expected, stated, rel_tol=1e-7, abs_tol=1e-7):
            return {"status": "invalid_trace", "equations_checked": checked,
                    "equations_valid": checked - 1,
                    "reason": f"Arithmetic mismatch: {lhs} = {expected:g}, but trace states {rhs} = {stated:g}."}

    try:
        answer_value = float(Decimal(answer)) if answer is not None else None
    except Exception:
        answer_value = None
        final_value = None
    if answer_value is not None and matching_results and math.isclose(
        matching_results[-1], answer_value, rel_tol=1e-7, abs_tol=1e-7
    ):
        return {"status": "verified", "equations_checked": checked,
                "equations_valid": checked, "reason": "All extracted equations are consistent and the final equation matches the answer."}
    return {"status": "unverified", "equations_checked": checked,
            "equations_valid": checked,
            "reason": "Equations are internally consistent, but the last equation does not establish the extracted final answer."}


def _get_agent_answer(record: dict[str, Any], i: int) -> Any:
    answers = record.get("agent_answers")
    if isinstance(answers, list) and i < len(answers):
        return answers[i]
    return record.get(AGENT_KEYS[i])


def _get_agent_response(record: dict[str, Any], i: int) -> str:
    responses = record.get("agent_responses")
    if isinstance(responses, list) and i < len(responses):
        return str(responses[i] or "")
    return ""


def _score(record: dict[str, Any], i: int) -> float:
    scores = record.get("combined_scores")
    if isinstance(scores, list) and i < len(scores):
        try: return float(scores[i])
        except (TypeError, ValueError): pass
    for key in (f"combined_score_agent{i+1}",):
        try: return float(record[key])
        except (KeyError, TypeError, ValueError): pass
    return float("inf")


def select_improved_mcf(record: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for i in range(3):
        raw = _get_agent_answer(record, i)
        answer = normalize_number(raw)
        if answer is None:
            candidates.append({"agent": i + 1, "raw_answer": raw, "answer": None,
                               "valid": False, "verification": "not_checked", "score": _score(record, i)})
            continue
        verification = verify_math_trace(_get_agent_response(record, i), answer)
        candidates.append({"agent": i + 1, "raw_answer": raw, "answer": answer,
                           "valid": True, "verification": verification["status"],
                           "verification_reason": verification["reason"],
                           "equations_checked": verification["equations_checked"],
                           "score": _score(record, i)})

    valid = [c for c in candidates if c["valid"]]
    if not valid:
        return {"prediction": None, "selected_agent": None, "selection_reason": "no_valid_agent_answers",
                "valid_agent_count": 0, "consensus_count": 0, "candidate_details": candidates}

    counts = Counter(c["answer"] for c in valid)
    groups = []
    for answer, count in counts.items():
        members = [c for c in valid if c["answer"] == answer]
        verified_count = sum(c["verification"] == "verified" for c in members)
        invalid_trace_count = sum(c["verification"] == "invalid_trace" for c in members)
        groups.append({"answer": answer, "count": count, "verified_count": verified_count,
                       "invalid_trace_count": invalid_trace_count, "members": members})

    consensus = [g for g in groups if g["count"] >= 2]
    verified_consensus = [g for g in consensus if g["verified_count"] >= 1]
    if verified_consensus:
        winner = sorted(verified_consensus, key=lambda g: (-g["count"], -g["verified_count"],
                         min(m["score"] for m in g["members"])))[0]
        reason = "verified_consensus"
    elif consensus:
        winner = sorted(consensus, key=lambda g: (-g["count"],
                         min(m["score"] for m in g["members"])))[0]
        reason = "valid_agent_consensus"
    else:
        verified = [c for c in valid if c["verification"] == "verified"]
        if verified:
            chosen = min(verified, key=lambda c: c["score"])
            winner = {"answer": chosen["answer"], "count": 1, "members": [chosen]}
            reason = "single_mathematically_verified_answer"
        else:
            chosen = min(valid, key=lambda c: c["score"])
            winner = {"answer": chosen["answer"], "count": 1, "members": [chosen]}
            reason = "minimum_original_mcf_score_among_valid_answers"

    selected = min(winner["members"], key=lambda c: c["score"])
    return {"prediction": winner["answer"], "selected_agent": selected["agent"],
            "selection_reason": reason, "valid_agent_count": len(valid),
            "consensus_count": winner["count"], "selected_verification": selected["verification"],
            "candidate_details": candidates}



def apply_improved_selection(record: dict[str, Any]) -> dict[str, Any]:
    decision = select_improved_mcf(record)
    prediction = decision.get("prediction")
    gold = normalize_number(record.get("gold_answer"))
    record["original_mcf_answer"] = record.get("mcf_answer")
    record["original_mcf_correct"] = bool(prediction is not None and gold is not None and normalize_number(record.get("mcf_answer")) == gold)
    record["mcf_answer"] = prediction
    record["selected_agent"] = decision.get("selected_agent")
    record["mcf_selection_reason"] = decision.get("selection_reason")
    record["mcf_correct"] = bool(prediction is not None and gold is not None and prediction == gold)
    record["valid_agent_count"] = decision.get("valid_agent_count", 0)
    record["consensus_count"] = decision.get("consensus_count", 0)
    record["improved_candidate_details"] = decision.get("candidate_details", [])
    for candidate in decision.get("candidate_details", []):
        i = candidate["agent"]
        record[f"agent{i}_valid"] = candidate["valid"]
        record[f"agent{i}_math_status"] = candidate["verification"]
        record[f"agent{i}_math_reason"] = candidate.get("verification_reason", "")
    return record
