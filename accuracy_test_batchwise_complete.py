#!/usr/bin/env python3
import json, os, re, time, gc
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.helpers import load_config

BATCH_SIZE = 10  # <-- configurable


def extract_answer(t):
    if not t:
        return None
    for p in [
        r"####\s*([-+]?\d*\.?\d+)",
        r"\\boxed\{([-+]?\d*\.?\d+)\}",
        r"(?:final answer|answer)\s*(?:is)?\s*:?\s*([-+]?\d*\.?\d+)",
    ]:
        m = re.search(p, t, re.I)
        if m:
            return m.group(1)
    n = re.findall(r"[-+]?\d*\.?\d+", t.replace(",", ""))
    return n[-1] if n else None


def check_match(p, g):
    try:
        return int(abs(float(p) - float(g)) < 1e-6)
    except Exception:
        return int(str(p).strip() == str(g).strip())


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def load(path):
    tok = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tok, model


def unload(tok, model):
    del tok
    del model
    clear()


def gen(model, tok, prompt, mx):
    inp = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            do_sample=False,
            max_new_tokens=mx,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.05,
            use_cache=True,
        )
    return tok.decode(
        out[0][inp["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


cfg = load_config()

data = json.load(open(cfg["data"]["gsm8k_test_file"]))[: cfg["evaluation"]["num_samples"]]

csvf = cfg["data"]["accuracy_csv"]
results = []
processed = 0
correct = 0

if os.path.exists(csvf):
    old = pd.read_csv(csvf)
    results = old.to_dict("records")
    processed = len(results)
    if processed:
        correct = int(old["correct"].sum())

pbar = tqdm(total=len(data), initial=processed)

for batch_start in range(processed, len(data), BATCH_SIZE):

    batch = data[batch_start: batch_start + BATCH_SIZE]
    batch_items = []

    print(f"\n=== Batch {batch_start//BATCH_SIZE + 1} ===")
    print("Loading planner...")
    ptok, planner = load(cfg["model"]["planner_model_path"])

    for sample in batch:
        q = sample["question"]
        gold = extract_answer(sample["answer"])

        plan_prompt = f"""Generate Solution Guidance.

Rules:
- Do not solve.
- No numbers.
- 2-5 guidance steps.

Question:
{q}

Solution Guidance:
"""

        t0 = time.time()
        plan = gen(
            planner,
            ptok,
            plan_prompt,
            cfg["evaluation"]["planner_max_tokens"],
        )
        tplan = time.time() - t0

        batch_items.append({
            "question": q,
            "gold": gold,
            "plan": plan,
            "planner_time": tplan,
        })

    unload(ptok, planner)

    print(torch.cuda.memory_allocated() / 1024**2)
    print(torch.cuda.memory_reserved() / 1024**2)

    print("Loading executor...")
    etok, executor = load(cfg["model"]["executor_model_path"])

    for item in batch:

        info = batch_items.pop(0)

        exec_prompt = f"""You are solving a grade-school math problem.

Use the Solution Guidance internally.

QUESTION
{info['question']}

SOLUTION GUIDANCE
{info['plan']}

Return exactly:

#### <final numeric answer>
"""

        t1 = time.time()
        raw = gen(
            executor,
            etok,
            exec_prompt,
            cfg["evaluation"]["executor_max_tokens"],
        )
        texec = time.time() - t1

        pred = extract_answer(raw)

        idx = len(results) + 1
        ok = check_match(pred, info["gold"])
        correct += ok
        acc = correct / idx * 100

        print("=" * 80)
        print("QUESTION:")
        print(info["question"])
        print("\nPLAN:")
        print(info["plan"])
        print("\nEXECUTOR OUTPUT:")
        print(raw)
        print("=" * 80)

        results.append({
            "question_no": idx,
            "question": info["question"],
            "generated_plan": info["plan"],
            "raw_executor_output": raw,
            "predicted_answer": pred,
            "expected_answer": info["gold"],
            "correct": ok,
            "planner_time_s": round(info["planner_time"], 3),
            "executor_time_s": round(texec, 3),
            "running_accuracy": round(acc, 2),
        })

        pd.DataFrame(results).to_csv(csvf, index=False)

        pbar.update(1)
        pbar.set_postfix(
            Acc=f"{acc:.2f}%",
            Pred=str(pred),
            GT=str(info["gold"]),
        )

    unload(etok, executor)

pbar.close()
print("=" * 60)
print(f"Final Accuracy: {correct}/{len(results)} = {correct/len(results)*100:.2f}%")
print(f"CSV saved to {csvf}")
