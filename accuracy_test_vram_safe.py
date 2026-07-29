#!/usr/bin/env python3
# See chat for details. VRAM-safe version: loads/unloads planner and executor each iteration.
import json,os,re,time,gc,pandas as pd,torch
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.utils.helpers import load_config

def extract_answer(t):
    if not t:return None
    for p in [r"####\s*([-+]?\d*\.?\d+)",r"\\boxed\{([-+]?\d*\.?\d+)\}",r"(?:final answer|answer)\s*(?:is)?\s*:?\s*([-+]?\d*\.?\d+)"]:
        m=re.search(p,t,re.I)
        if m:return m.group(1)
    n=re.findall(r"[-+]?\d*\.?\d+",t.replace(",",""));return n[-1] if n else None
def check_match(p,g):
    try:return int(abs(float(p)-float(g))<1e-6)
    except:return int(str(p).strip()==str(g).strip())
def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda,"ipc_collect"): torch.cuda.ipc_collect()
def load(path):
    tok=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map="auto" if torch.cuda.is_available() else None)
    return tok,model
def gen(model,tok,prompt,mx):
    inp=tok(prompt,return_tensors="pt").to(next(model.parameters()).device)
    out=model.generate(**inp,do_sample=False,max_new_tokens=mx,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,repetition_penalty=1.05)
    return tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
cfg=load_config()
data=json.load(open(cfg["data"]["gsm8k_test_file"]))[:cfg["evaluation"]["num_samples"]]
csvf=cfg["data"]["accuracy_csv"];res=[];processed=0;correct=0
if os.path.exists(csvf):
    old=pd.read_csv(csvf);res=old.to_dict("records");processed=len(res);correct=int(old["correct"].sum())
pbar=tqdm(data[processed:],initial=processed,total=len(data))
for idx,s in enumerate(pbar,start=processed+1):
    q=s["question"];gold=extract_answer(s["answer"])
    ptok,p=load(cfg["model"]["planner_model_path"])
    plan=gen(p,ptok,f"Generate Solution Guidance.\nRules:\n- Do not solve.\n- No numbers.\n- 2-5 guidance steps.\n\nQuestion:\n{q}\n\nSolution Guidance:\n",cfg["evaluation"]["planner_max_tokens"])
    del ptok,p;clear()
    etok,e=load(cfg["model"]["executor_model_path"])
    raw=gen(e,etok,f"You are solving a grade-school math problem.\nUse the Solution Guidance internally.\n\nQUESTION\n{q}\n\nSOLUTION GUIDANCE\n{plan}\n\nReturn exactly:\n#### <final numeric answer>\n",cfg["evaluation"]["executor_max_tokens"])
    del etok,e;clear()
    pred=extract_answer(raw);ok=check_match(pred,gold);correct+=ok;acc=correct/idx*100
    res.append({"question_no":idx,"question":q,"generated_plan":plan,"raw_executor_output":raw,"predicted_answer":pred,"expected_answer":gold,"correct":ok,"running_accuracy":round(acc,2)})
    pd.DataFrame(res).to_csv(csvf,index=False)
    pbar.set_postfix(Acc=f"{acc:.2f}%",Pred=pred,GT=gold)
print("Done")
