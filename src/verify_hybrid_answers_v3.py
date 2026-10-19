#!/usr/bin/env python3
"""Structured-output local Ollama verifier for GSM8K / LLM_MCF.

Gold answers are used only for scoring. Raw model output is saved for debugging.
"""
import argparse, csv, json, re, sys, time, urllib.request, urllib.error
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:/\d+)?"

def canon(value):
    if value is None: return ""
    s = str(value).strip().replace("$", "").replace(",", "").replace("₹", "")
    s = re.sub(r"\s+", "", s)
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?", s): return ""
    try:
        if "/" in s:
            x, y = s.split("/", 1); d = Decimal(x) / Decimal(y)
        else: d = Decimal(s)
        return format(d.normalize(), "f") if d.is_finite() else ""
    except (InvalidOperation, ZeroDivisionError, ValueError): return ""

def majority(values):
    vals = [canon(v) for v in values if canon(v)]
    if not vals: return ""
    counts = Counter(vals); high = max(counts.values())
    return next(v for v in vals if counts[v] == high)

def extract_gold_answer(text):
    if text is None: return ""
    s = str(text).strip()
    # GSM8K answers commonly end with #### number
    matches = re.findall(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", s)
    if matches: return canon(matches[-1])
    return canon(s)

def ollama_call(base_url, model, messages, timeout, temperature):
    payload = {
        "model": model, "messages": messages, "stream": False,
        "format": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "calculation": {"type": "string"},
            "reason": {"type": "string"}
        }, "required": ["answer", "calculation", "reason"]},
        "options": {"temperature": temperature}
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp: data = json.loads(resp.read().decode("utf-8"))
    raw = data.get("message", {}).get("content", "")
    obj = json.loads(raw)
    if not isinstance(obj, dict): raise ValueError("Ollama JSON response was not an object")
    return obj, raw

def solve(base_url, model, question, timeout, temperature, retries):
    system = {"role":"system", "content":"You are a careful arithmetic word-problem solver. Return only the required JSON object. The answer must be the final result of the calculation."}
    prompt = ('Solve the problem from scratch. Work through the arithmetic carefully. The "answer" field must contain ONLY the final numeric value (no words, units, commas, or explanation). '
              'The "calculation" field should be a concise calculation and must explicitly end with a line like FINAL_RESULT: 42. Do not guess or invent information. '
              'Return JSON with exactly these keys: answer (string), calculation (string), reason (string). Do not include any reference answer.\n\nQUESTION:\n' + question)
    last_error = None
    for attempt in range(retries + 1):
        try:
            obj, raw = ollama_call(base_url, model, [system, {"role":"user","content":prompt}], timeout, temperature)
            ans = canon(obj.get("answer", "")); calc = str(obj.get("calculation", "")); reason = str(obj.get("reason", ""))
            final_matches = re.findall(r"FINAL_RESULT\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?(?:/\d+)?)", calc, re.I)
            calc_final = canon(final_matches[-1]) if final_matches else ""
            consistent = bool(ans and calc_final and ans == calc_final)
            repair_raw = ""
            if not consistent:
                audit = ('Re-solve this problem independently. Your previous output is inconsistent or lacks an explicit final result. Check each arithmetic operation. Return the same JSON schema. The answer must equal the numeric value after FINAL_RESULT: in calculation.\n\nQUESTION:\n' + question + '\n\nPrevious answer: ' + str(obj.get("answer", "")) + '\nPrevious calculation:\n' + calc)
                fixed, repair_raw = ollama_call(base_url, model, [system, {"role":"user","content":audit}], timeout, temperature)
                fixed_ans = canon(fixed.get("answer", "")); fixed_calc = str(fixed.get("calculation", "")); fixed_reason = str(fixed.get("reason", ""))
                fm = re.findall(r"FINAL_RESULT\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?(?:/\d+)?)", fixed_calc, re.I)
                fv = canon(fm[-1]) if fm else ""
                if fixed_ans and fv and fixed_ans == fv:
                    ans, calc, reason, consistent = fixed_ans, fixed_calc, fixed_reason, True
                else:
                    ans, calc, reason, consistent = "", fixed_calc or calc, fixed_reason or reason, False
            return {"answer":ans, "calculation":calc, "reason":reason, "consistent":int(consistent), "repair_used":int(bool(repair_raw)), "raw_response":raw, "repair_response":repair_raw, "attempts":attempt+1, "error":""}
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries: time.sleep(min(2 ** attempt, 8))
    return {"answer":"", "calculation":"", "reason":"", "consistent":0, "repair_used":0, "raw_response":"", "repair_response":"", "attempts":retries+1, "error":last_error or "unknown error"}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='results/amr_mcf_raw_responses.json')
    ap.add_argument('--model', default='llama3.2:3b')
    ap.add_argument('--ollama-url', default='http://localhost:11434')
    ap.add_argument('--output-dir', default='results/analysis')
    ap.add_argument('--limit', type=int, default=0, help='Process first N records; 0 means all')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--retries', type=int, default=1, help='Retries after an Ollama call failure')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--pause', type=float, default=0.0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    src = Path(args.input)
    if not src.exists(): print(f'ERROR: input not found: {src}', file=sys.stderr); return 2
    try: records = json.loads(src.read_text(encoding='utf-8'))
    except Exception as e: print(f'ERROR reading JSON: {e}', file=sys.stderr); return 2
    if isinstance(records, dict): records = records.get('results', records.get('questions', []))
    if not isinstance(records, list) or not records: print('ERROR: expected a non-empty JSON list', file=sys.stderr); return 2
    if args.limit > 0: records = records[:args.limit]
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / 'answer_verifier_v3_results.csv'; report_path = outdir / 'answer_verifier_v3_report.txt'
    fields = ['question_id','question','gold_answer','mcf_answer','mcf_correct','majority_answer','majority_correct','agent1_answer','agent2_answer','agent3_answer','verifier_answer','verifier_correct','calculation_consistent','selected_candidate','selected_answer','selected_correct','calculation','reason','repair_used','attempts','error','raw_response','repair_response']
    existing = {}
    if args.resume and csv_path.exists():
        with csv_path.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f): existing[row.get('question_id','')] = row
    mode = 'a' if args.resume and csv_path.exists() else 'w'; processed = 0
    with csv_path.open(mode, newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w': writer.writeheader()
        for i, rec in enumerate(records):
            qid = str(rec.get('question_id', i))
            if qid in existing and not existing[qid].get('error') and existing[qid].get('verifier_answer'):
                print(f'[{i+1}/{len(records)}] id={qid}: already completed; skipping'); continue
            candidates = rec.get('agent_answers') or [rec.get(f'agent{k}_answer','') for k in (1,2,3)]
            candidates = [canon(v) for v in candidates[:3]]
            while len(candidates) < 3: candidates.append('')
            gold = extract_gold_answer(rec.get('gold_answer',''))
            mcf = extract_gold_answer(rec.get('mcf_answer',''))
            maj = majority(candidates)
            row = {'question_id':qid,'question':str(rec.get('question','')),'gold_answer':gold,'mcf_answer':mcf,'mcf_correct':int(bool(gold and mcf==gold)),
                   'majority_answer':maj,'majority_correct':int(bool(gold and maj==gold)),'agent1_answer':candidates[0],'agent2_answer':candidates[1],'agent3_answer':candidates[2],
                   'verifier_answer':'','verifier_correct':0,'calculation_consistent':0,'selected_candidate':0,'selected_answer':'','selected_correct':0,'calculation':'','reason':'','repair_used':0,'attempts':0,'error':'','raw_response':'','repair_response':''}
            result = solve(args.ollama_url, args.model, row['question'], args.timeout, args.temperature, max(0,args.retries))
            row.update({'verifier_answer':result['answer'],'verifier_correct':int(bool(result['answer'] and gold and result['answer']==gold)),
                        'calculation_consistent':result['consistent'],'calculation':result['calculation'],'reason':result['reason'],'repair_used':result['repair_used'],
                        'attempts':result['attempts'],'error':result['error'],'raw_response':result['raw_response'],'repair_response':result['repair_response']})
            matching = [j+1 for j, candidate in enumerate(candidates) if result['answer'] and candidate == result['answer']]
            selected = matching[0] if matching else 0
            selected_answer = candidates[selected-1] if selected else result['answer']
            row.update({'selected_candidate':selected,'selected_answer':selected_answer,'selected_correct':int(bool(selected_answer and gold and selected_answer==gold))})
            status = f"MCF={row['mcf_correct']} majority={row['majority_correct']} verifier={row['verifier_correct']} selected={row['selected_correct']} consistent={row['calculation_consistent']} repair={row['repair_used']}"
            if row['error']: print(f'[{i+1}/{len(records)}] id={qid}: ERROR {row["error"]}', file=sys.stderr)
            else: print(f'[{i+1}/{len(records)}] id={qid}: {status}')
            writer.writerow(row); f.flush(); processed += 1
            if args.pause and i < len(records)-1: time.sleep(args.pause)
    with csv_path.open(newline='', encoding='utf-8-sig') as f: all_rows = list(csv.DictReader(f))
    wanted = {str(r.get('question_id', i)) for i,r in enumerate(records)}
    rows = [r for r in all_rows if r['question_id'] in wanted]
    n = len(rows)
    def count(field): return sum(str(r.get(field,'0')).strip().lower() in ('1','true','yes') for r in rows)
    report = ['STRUCTURED-OUTPUT LOCAL OLLAMA VERIFIER (V3)',f'Input: {src}',f'Model: {args.model}',f'Questions: {n}',f'Rows with no recorded error: {sum(not r.get("error") for r in rows)}',f'Timeouts/errors: {sum(bool(r.get("error")) for r in rows)}',f'Repair pass used: {sum(int(r.get("repair_used") or 0) for r in rows)}',f'Calculation consistency confirmed: {count("calculation_consistent")}', '', 'Accuracy (all evaluated rows; errors/unresolved answers count as incorrect):']
    for label, field in [('Existing MCF','mcf_correct'),('Majority voting','majority_correct'),('Verifier independent answer','verifier_correct'),('Verifier candidate selection','selected_correct')]:
        c = count(field); report.append(f'{label}: {c}/{n} ({100*c/n:.1f}%)' if n else f'{label}: 0/0')
    report += ['',f'New rows processed this run: {processed}',f'CSV: {csv_path}',f'Report: {report_path}','', 'Gold answers are used only for scoring, never sent to Ollama.','Exploratory comparison only; use a separate held-out dataset for final claims.']
    report_path.write_text('\n'.join(report)+'\n', encoding='utf-8'); print('\n'+'\n'.join(report)); return 0

if __name__ == '__main__': raise SystemExit(main())
