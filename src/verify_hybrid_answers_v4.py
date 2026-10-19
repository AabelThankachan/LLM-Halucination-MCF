#!/usr/bin/env python3
"""Debug-friendly Ollama verifier for GSM8K. Gold labels are used only for scoring."""
import argparse, csv, json, re, sys, time, urllib.request, urllib.error
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+"

def canon(value):
    if value is None: return ""
    s = str(value).strip().replace("$", "").replace("₹", "").replace(",", "")
    s = re.sub(r"\s+", "", s)
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?", s): return ""
    try:
        if "/" in s:
            a,b=s.split("/",1); n=Decimal(a)/Decimal(b)
        else: n=Decimal(s)
        return format(n.normalize(), "f") if n.is_finite() else ""
    except (InvalidOperation, ZeroDivisionError, ValueError): return ""

def extract_answer(value):
    """Prefer explicit final-result marker; otherwise extract last number from answer text."""
    c = canon(value)
    if c: return c
    s = str(value or "")
    marked = re.findall(r"(?:FINAL_RESULT|FINAL_ANSWER|ANSWER)\s*[:=]\s*\$?\s*("+NUM+r")", s, re.I)
    if marked: return canon(marked[-1])
    vals = re.findall(NUM, s)
    return canon(vals[-1]) if vals else ""

def gold_answer(value):
    s=str(value or "")
    marked=re.findall(r"####\s*\$?\s*("+NUM+r")",s)
    if marked: return canon(marked[-1])
    return extract_answer(s)

def majority(values):
    vals=[canon(v) for v in values if canon(v)]
    if not vals: return ""
    counts=Counter(vals); high=max(counts.values())
    return next(v for v in vals if counts[v]==high)

def call_ollama(url, model, question, timeout, temperature):
    payload={"model":model,"stream":False,"format":{"type":"object","properties":{"answer":{"type":"string"},"calculation":{"type":"string"},"reason":{"type":"string"}},"required":["answer","calculation","reason"]},"options":{"temperature":temperature},"messages":[
      {"role":"system","content":"Solve arithmetic word problems carefully. Do not use any reference answer. Return the requested JSON object."},
      {"role":"user","content":"Solve this problem independently. The answer field MUST contain only the final numeric value, with no words or units. In calculation, show the arithmetic and end with FINAL_RESULT: <number>. If the question asks for dollars, compute the amount of money earned, not just the count of items.\n\nQUESTION:\n"+question}
    ]}
    req=urllib.request.Request(url.rstrip('/')+'/api/chat',data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as resp: data=json.loads(resp.read().decode())
    raw=str(data.get('message',{}).get('content',''))
    obj=json.loads(raw)
    if not isinstance(obj,dict): raise ValueError('response JSON is not an object')
    return obj,raw

def solve(url,model,question,timeout,retries,temperature):
    last=''; raw=''; obj={}; attempts=0
    for attempt in range(retries+1):
        attempts=attempt+1
        try:
            obj,raw=call_ollama(url,model,question,timeout,temperature)
            # Prefer numeric answer field; if the model puts prose there, extract the last numeric value.
            answer=extract_answer(obj.get('answer',''))
            if answer:
                calc=str(obj.get('calculation','')); reason=str(obj.get('reason',''))
                marker=re.findall(r"FINAL_RESULT\s*:\s*\$?\s*("+NUM+r")",calc,re.I)
                calc_answer=canon(marker[-1]) if marker else ''
                consistent=int(bool(answer and calc_answer and answer==calc_answer))
                # Consistency is diagnostic only; never discard a parseable answer because this heuristic fails.
                return {'answer':answer,'calculation':calc,'reason':reason,'consistent':consistent,'raw':raw,'error':'','attempts':attempts}
            last='Could not extract a numeric answer from JSON answer field'
        except (urllib.error.URLError,TimeoutError,OSError,json.JSONDecodeError,ValueError) as e:
            last=f'{type(e).__name__}: {e}'
        if attempt<retries: time.sleep(min(2**attempt,5))
    return {'answer':'','calculation':str(obj.get('calculation','')),'reason':str(obj.get('reason','')),'consistent':0,'raw':raw,'error':last,'attempts':attempts}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/amr_mcf_raw_responses.json'); p.add_argument('--model',default='llama3.2:3b')
    p.add_argument('--ollama-url',default='http://localhost:11434'); p.add_argument('--output-dir',default='results/analysis')
    p.add_argument('--limit',type=int,default=0); p.add_argument('--timeout',type=int,default=300); p.add_argument('--retries',type=int,default=1)
    p.add_argument('--temperature',type=float,default=0.0); p.add_argument('--pause',type=float,default=0.0)
    a=p.parse_args(); src=Path(a.input)
    if not src.exists(): print(f'ERROR: input not found: {src}',file=sys.stderr); return 2
    try: records=json.loads(src.read_text(encoding='utf-8-sig'))
    except Exception as e: print(f'ERROR reading JSON: {e}',file=sys.stderr); return 2
    if isinstance(records,dict): records=records.get('results',records.get('questions',[]))
    if not isinstance(records,list) or not records: print('ERROR: expected a non-empty JSON list',file=sys.stderr); return 2
    if a.limit>0: records=records[:a.limit]
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    csvpath=outdir/'answer_verifier_v4_results.csv'; reportpath=outdir/'answer_verifier_v4_report.txt'
    fields=['question_id','question','gold_answer','mcf_answer','mcf_correct','majority_answer','majority_correct','agent1_answer','agent2_answer','agent3_answer','verifier_answer','verifier_correct','calculation_consistent','selected_candidate','selected_answer','selected_correct','calculation','reason','attempts','error','raw_response']
    rows=[]
    for i,r in enumerate(records):
        qid=str(r.get('question_id',i)); q=str(r.get('question',''))
        gold=gold_answer(r.get('gold_answer','')); agents=r.get('agent_answers')
        if not isinstance(agents,list) or len(agents)<3: agents=[r.get(f'agent{k}_answer','') for k in (1,2,3)]
        agents=[canon(v) for v in agents[:3]]
        while len(agents)<3: agents.append('')
        mcf=canon(r.get('mcf_answer','')); maj=majority(agents)
        result=solve(a.ollama_url,a.model,q,a.timeout,max(0,a.retries),a.temperature)
        va=result['answer']; matching=[j+1 for j,x in enumerate(agents) if va and x==va]
        selected=agents[matching[0]-1] if matching else va; selected_idx=matching[0] if matching else 0
        row={'question_id':qid,'question':q,'gold_answer':gold,'mcf_answer':mcf,'mcf_correct':int(bool(gold and mcf==gold)),
          'majority_answer':maj,'majority_correct':int(bool(gold and maj==gold)),'agent1_answer':agents[0],'agent2_answer':agents[1],'agent3_answer':agents[2],
          'verifier_answer':va,'verifier_correct':int(bool(gold and va and va==gold)),'calculation_consistent':result['consistent'],'selected_candidate':selected_idx,
          'selected_answer':selected,'selected_correct':int(bool(gold and selected and selected==gold)),'calculation':result['calculation'],'reason':result['reason'],
          'attempts':result['attempts'],'error':result['error'],'raw_response':result['raw']}
        rows.append(row)
        with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        status=f"MCF={row['mcf_correct']} majority={row['majority_correct']} verifier={row['verifier_correct']} selected={row['selected_correct']} consistent={row['calculation_consistent']}"
        print(f"[{i+1}/{len(records)}] id={qid}: ERROR {result['error']}" if result['error'] else f'[{i+1}/{len(records)}] id={qid}: {status}')
        if a.pause and i<len(records)-1: time.sleep(a.pause)
    n=len(rows)
    def count(k): return sum(int(x[k]) for x in rows)
    report=['ANSWER-PARSING DIAGNOSTIC VERIFIER (V4)',f'Input: {src}',f'Model: {a.model}',f'Questions: {n}',f'Errors/unparsed answers: {sum(bool(x["error"] or not x["verifier_answer"]) for x in rows)}',f'Calculation marker consistency (diagnostic only): {count("calculation_consistent")}/{n}','','Accuracy (all errors/unparsed answers count as incorrect):']
    for label,k in [('Existing MCF','mcf_correct'),('Majority voting','majority_correct'),('Verifier independent answer','verifier_correct'),('Verifier candidate selection','selected_correct')]: report.append(f'{label}: {count(k)}/{n} ({100*count(k)/n:.1f}%)' if n else f'{label}: 0/0')
    report += ['',f'CSV: {csvpath}',f'Report: {reportpath}','Gold answers are used only for scoring and are not sent to Ollama.','Consistency is reported diagnostically and is not used to discard a parseable answer.']
    reportpath.write_text('\n'.join(report)+'\n',encoding='utf-8'); print('\n'+'\n'.join(report)); return 0
if __name__=='__main__': raise SystemExit(main())
