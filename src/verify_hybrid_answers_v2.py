#!/usr/bin/env python3
"""Consistency-checked local Ollama verifier for GSM8K/LLM_MCF.

Gold answers are used only for scoring, never included in prompts. Uses only Python stdlib.
"""
import argparse, csv, json, re, sys, time, urllib.request, urllib.error
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:/\d+)?"

def canon(s):
    if s is None: return ""
    s = str(s).strip().replace("$", "").replace(",", "")
    m = re.fullmatch(NUM, s)
    if not m: return ""
    try:
        if "/" in s:
            a,b=s.split("/",1); d=Decimal(a)/Decimal(b)
        else: d=Decimal(s)
        return format(d.normalize(), "f") if d.is_finite() else ""
    except (InvalidOperation, ZeroDivisionError, ValueError): return ""

def extract_num(value):
    if value is None: return ""
    s=str(value).strip()
    # Prefer explicitly labelled final-answer fields.
    pats=[rf"(?:FINAL_ANSWER|INDEPENDENT_ANSWER|SELECTED_ANSWER|verified_answer|answer)\s*[:=]\s*({NUM})", rf"####\s*({NUM})"]
    for p in pats:
        ms=re.findall(p,s,re.I)
        if ms: return canon(ms[-1])
    # A bare number is unambiguous; for explanatory text, don't blindly take the last number.
    if re.fullmatch(NUM, s.replace('$','').replace(',','')): return canon(s)
    return ""

def same(a,b):
    a=extract_num(a) or canon(a); b=extract_num(b) or canon(b)
    return bool(a and b and a==b)

def majority(answers):
    vals=[canon(a) for a in answers if canon(a)]
    if not vals:return ""
    c=Counter(vals); winners={v for v,n in c.items() if n==max(c.values())}
    return next(v for v in vals if v in winners)

def ollama_chat(url,model,messages,timeout,temperature):
    payload={"model":model,"messages":messages,"stream":False,"format":"json","options":{"temperature":temperature}}
    req=urllib.request.Request(url.rstrip('/')+'/api/chat',data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=timeout) as resp: data=json.loads(resp.read().decode())
    content=data.get('message',{}).get('content','')
    try: obj=json.loads(content)
    except json.JSONDecodeError:
        match=re.search(r'\{.*\}',content,re.S)
        if not match: raise ValueError('Ollama returned invalid JSON: '+content[:250])
        obj=json.loads(match.group())
    return obj,content

def solve_and_audit(url,model,question,candidates,timeout,temp):
    cand='\n'.join(f'{i+1}: {a or "(missing)"}' for i,a in enumerate(candidates))
    system={"role":"system","content":"You solve arithmetic word problems. Follow the JSON schema exactly. Check every arithmetic operation before answering."}
    prompt=("Solve the GSM8K word problem independently. Show a concise calculation. The answer field MUST be the result of the final calculation, not an intermediate number. "
      "Then compare the candidates only after solving. Do not assume majority is correct. Select a candidate ONLY if its numeric answer exactly equals your independently computed answer; otherwise use your own answer and selected_candidate=0. "
      'Return JSON: {"answer":"number only","calculation":"short calculation ending with the final numeric result","selected_candidate":0,"selected_answer":"number only","reason":"brief explanation"}. '
      'selected_candidate must be 1, 2, or 3 only if that candidate equals your answer, otherwise 0. Do not include the gold answer.\n\nQUESTION:\n'+question+'\n\nCANDIDATES:\n'+cand)
    obj,raw=ollama_chat(url,model,[system,{"role":"user","content":prompt}],timeout,temp)
    answer=canon(obj.get('answer',''))
    calculation=str(obj.get('calculation','')).strip()
    # If model's answer and its explicit final result disagree, perform one repair/audit pass.
    calc_nums=re.findall(NUM, calculation.replace('$','').replace(',',''))
    calc_last=canon(calc_nums[-1]) if calc_nums else ''
    repair_used=False; repair_raw=''
    if not answer or (calc_last and answer != calc_last):
        repair_used=True
        repair_prompt=("Your previous answer may contain an inconsistency. Re-solve the problem from scratch and check each operation. "
          "The final numeric answer must equal the result of the final calculation. Do not defend your earlier answer. "
          'Return JSON only: {"answer":"number only","calculation":"short calculation ending with final result","reason":"brief explanation"}.\n\nQUESTION:\n'+question+
          '\n\nPrevious answer: '+str(obj.get('answer',''))+'\nPrevious calculation: '+calculation)
        fixed,repair_raw=ollama_chat(url,model,[system,{"role":"user","content":repair_prompt}],timeout,temp)
        fixed_answer=canon(fixed.get('answer',''))
        fixed_calc=str(fixed.get('calculation','')).strip()
        fixed_nums=re.findall(NUM,fixed_calc.replace('$','').replace(',',''))
        fixed_last=canon(fixed_nums[-1]) if fixed_nums else ''
        # Accept repaired answer only if its own calculation agrees; otherwise mark unresolved.
        if fixed_answer and fixed_last and fixed_answer==fixed_last:
            answer,calculation=fixed_answer,fixed_calc
            obj['reason']=fixed.get('reason',obj.get('reason',''))
        else:
            answer=''
            calculation=fixed_calc or calculation
    # Enforce candidate selection deterministically: model may select only a candidate matching answer.
    matches=[i+1 for i,a in enumerate(candidates[:3]) if answer and canon(a)==answer]
    requested=obj.get('selected_candidate',0)
    try: requested=int(requested)
    except (TypeError,ValueError): requested=0
    selected=requested if requested in matches else (matches[0] if matches else 0)
    selected_answer=canon(candidates[selected-1]) if selected else answer
    return {"answer":answer,"calculation":calculation,"selected_candidate":selected,"selected_answer":selected_answer,
            "reason":str(obj.get('reason','')),"repair_used":int(repair_used),"raw_response":raw,"repair_response":repair_raw}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/amr_mcf_raw_responses.json')
    p.add_argument('--model',default='llama3.2:3b')
    p.add_argument('--ollama-url',default='http://localhost:11434')
    p.add_argument('--output-dir',default='results/analysis')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--pause',type=float,default=0.0)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); src=Path(a.input)
    if not src.exists(): print(f'ERROR: input not found: {src}',file=sys.stderr); return 2
    try: records=json.loads(src.read_text(encoding='utf-8'))
    except Exception as e: print(f'ERROR reading JSON: {e}',file=sys.stderr); return 2
    if isinstance(records,dict): records=records.get('results',records.get('questions',[]))
    if not isinstance(records,list) or not records: print('ERROR: expected a non-empty list of records',file=sys.stderr); return 2
    if a.limit>0: records=records[:a.limit]
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    cp=out/'answer_verifier_v2_results.csv'; rp=out/'answer_verifier_v2_report.txt'
    fields=['question_id','question','gold_answer','mcf_answer','mcf_correct','majority_answer','majority_correct','agent1_answer','agent2_answer','agent3_answer','verifier_answer','verifier_correct','selected_candidate','selected_answer','selected_correct','calculation','reason','repair_used','error']
    done={}
    if a.resume and cp.exists():
        with cp.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f): done[row.get('question_id','')]=row
    mode='a' if a.resume and cp.exists() else 'w'; processed=0
    with cp.open(mode,newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if mode=='w': w.writeheader()
        for ix,rec in enumerate(records):
            qid=str(rec.get('question_id',ix))
            if qid in done and not done[qid].get('error'):
                print(f'[{ix+1}/{len(records)}] id={qid}: already completed; skipping'); continue
            agents=rec.get('agent_answers') or [rec.get(f'agent{i}_answer','') for i in (1,2,3)]
            agents=[canon(x) for x in agents[:3]]
            while len(agents)<3: agents.append('')
            gold=canon(rec.get('gold_answer','')); mcf=canon(rec.get('mcf_answer','')); maj=majority(agents)
            row={'question_id':qid,'question':str(rec.get('question','')),'gold_answer':gold,'mcf_answer':mcf,'mcf_correct':int(bool(mcf and mcf==gold)),
                 'majority_answer':maj,'majority_correct':int(bool(maj and maj==gold)),'agent1_answer':agents[0],'agent2_answer':agents[1],'agent3_answer':agents[2],
                 'verifier_answer':'','verifier_correct':0,'selected_candidate':0,'selected_answer':'','selected_correct':0,'calculation':'','reason':'','repair_used':0,'error':''}
            try:
                r=solve_and_audit(a.ollama_url,a.model,row['question'],agents,a.timeout,a.temperature)
                row.update({'verifier_answer':r['answer'],'verifier_correct':int(bool(r['answer'] and r['answer']==gold)),
                    'selected_candidate':r['selected_candidate'],'selected_answer':r['selected_answer'],
                    'selected_correct':int(bool(r['selected_answer'] and r['selected_answer']==gold)),
                    'calculation':r['calculation'],'reason':r['reason'],'repair_used':r['repair_used']})
                print(f"[{ix+1}/{len(records)}] id={qid}: MCF={row['mcf_correct']} majority={row['majority_correct']} verifier={row['verifier_correct']} selected={row['selected_correct']} repair={row['repair_used']}")
            except (urllib.error.URLError,TimeoutError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
                row['error']=str(e).replace('\n',' ')[:400]
                print(f'[{ix+1}/{len(records)}] id={qid}: ERROR {row["error"]}',file=sys.stderr)
            w.writerow(row); f.flush(); processed+=1
            if a.pause and ix<len(records)-1: time.sleep(a.pause)
    with cp.open(newline='',encoding='utf-8-sig') as f: allrows=list(csv.DictReader(f))
    ids={str(r.get('question_id',i)) for i,r in enumerate(records)}; rows=[r for r in allrows if r['question_id'] in ids]
    n=len(rows)
    def count(field): return sum(str(r.get(field,'0')).strip().lower() in ('1','true','yes') for r in rows)
    report=['CONSISTENCY-CHECKED LOCAL OLLAMA VERIFIER (V2)',f'Input: {src}',f'Model: {a.model}',f'Questions: {n}',f'Calls with no recorded error: {sum(not r.get("error") for r in rows)}',f'Repair pass used: {sum(int(r.get("repair_used") or 0) for r in rows)}','', 'Accuracy (all evaluated rows; errors/unresolved answers count as incorrect):']
    for label,field in [('Existing MCF','mcf_correct'),('Majority voting','majority_correct'),('Verifier independent answer','verifier_correct'),('Verifier candidate selection','selected_correct')]:
        c=count(field); report.append(f'{label}: {c}/{n} ({100*c/n:.1f}%)' if n else f'{label}: 0/0')
    report += ['',f'New rows processed this run: {processed}',f'CSV: {cp}',f'Report: {rp}','','Gold answers are used only for scoring, never sent to Ollama.','Exploratory comparison only; use a separate held-out dataset for final claims.']
    rp.write_text('\n'.join(report)+'\n',encoding='utf-8'); print('\n'+'\n'.join(report)); return 0

if __name__=='__main__': raise SystemExit(main())
