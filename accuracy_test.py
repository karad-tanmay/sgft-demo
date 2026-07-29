
#!/usr/bin/env python3
"""
accuracy_test.py

Two-model SGFT evaluation:
Question -> SGFT Planner -> Base Llama Executor -> Exact Match on GSM8K
"""

import json
import os
import re
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.helpers import load_config


def extract_answer(text):
    if not text:
        return None

    # Preferred SGFT format
    m = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
    if m:
        return m.group(1)

    # LaTeX boxed answer
    m = re.search(r"\\boxed\{([-+]?\d*\.?\d+)\}", text)
    if m:
        return m.group(1)

    # "The final answer is: 734"
    m = re.search(
        r"(?:final answer|answer)\s*(?:is)?\s*:?\s*([-+]?\d*\.?\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)

    # Fallback: last number in the response
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def check_match(pred, gold):
    if pred is None or gold is None:
        return 0
    try:
        return int(abs(float(pred) - float(gold)) < 1e-6)
    except Exception:
        return int(str(pred).strip() == str(gold).strip())


def generate(model, tokenizer, prompt, max_new_tokens):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True,
        )
    # decode only newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def main():
    cfg = load_config()

    planner_path = cfg["model"]["planner_model_path"]
    executor_path = cfg["model"]["executor_model_path"]

    test_file = cfg["data"]["gsm8k_test_file"]
    csv_file = cfg["data"]["accuracy_csv"]

    ev = cfg["evaluation"]
    num_samples = ev["num_samples"]
    max_plan = ev["planner_max_tokens"]
    max_exec = ev["executor_max_tokens"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading planner...")
    planner_tok = AutoTokenizer.from_pretrained(
        planner_path, fix_mistral_regex=True)
    planner = AutoModelForCausalLM.from_pretrained(
        planner_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        planner.to(device)

    print("Loading executor...")
    exec_tok = AutoTokenizer.from_pretrained(
        executor_path, fix_mistral_regex=True)
    executor = AutoModelForCausalLM.from_pretrained(
        executor_path,
         dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if device == "cpu":
        executor.to(device)

    with open(test_file) as f:
        data = json.load(f)

    data = data[:num_samples]

    results = []
    processed = 0
    correct = 0

    if os.path.exists(csv_file):
        old = pd.read_csv(csv_file)
        results = old.to_dict("records")
        processed = len(results)
        correct = int(old["correct"].sum()) if processed else 0
        print(f"Resuming from {processed} samples")

    pbar = tqdm(data[processed:], initial=processed, total=len(data))

    for idx, sample in enumerate(pbar, start=processed + 1):
        q = sample["question"]
        gold = extract_answer(sample["answer"])

        plan_prompt = f"""
Generate Solution Guidance for the following math problem.

Rules:
- Do not solve the problem.
- Do not use numbers.
- Do not include calculations.
- Generate 2-5 guidance steps.

Question:
{q}

Solution Guidance:
"""

        t0 = time.time()
        plan = generate(planner, planner_tok, plan_prompt, max_plan)
        t_plan = time.time() - t0

        exec_prompt = f"""
You are solving a grade-school math problem.

The text below is Solution Guidance.
It is provided ONLY as an internal reasoning strategy.
Do NOT repeat it.
Do NOT explain your reasoning.
Do NOT output any intermediate calculations.

====================
QUESTION
====================

{q}

====================
SOLUTION GUIDANCE
====================

{plan}

====================
OUTPUT FORMAT
====================

Compute the answer internally.

Output ONLY the final numeric answer.

Do not output words.
Do not output explanations.
Do not output reasoning.
Do not output markdown.
Do not output LaTeX.
Do not output \\boxed{{}}.
Do not output 'The answer is'.
Do not output multi-line answer
Do not output: "#### Example"

Return exactly one line in this format:

#### <final answer numeric value>
"""

        t1 = time.time()
        raw = generate(executor, exec_tok, exec_prompt, max_exec)
        t_exec = time.time() - t1

        print("=" * 80)
        print("QUESTION:")
        print(q)

        print("\nPLAN:")
        print(plan)

        print("\nEXECUTOR OUTPUT:")
        print(raw)

        print("=" * 80)

        pred = extract_answer(raw)
        ok = check_match(pred, gold)
        correct += ok
        acc = correct / idx * 100

        results.append({
            "question_no": idx,
            "question": q,
            "generated_plan": plan,
            "raw_executor_output": raw,
            "predicted_answer": pred,
            "expected_answer": gold,
            "correct": ok,
            "planner_time_s": round(t_plan, 3),
            "executor_time_s": round(t_exec, 3),
            "running_accuracy": round(acc, 2)
        })

        pd.DataFrame(results).to_csv(csv_file, index=False)

        pbar.set_postfix({
            "Acc": f"{acc:.2f}%",
            "Pred": pred,
            "GT": gold
        })

    print("=" * 60)
    print(f"Final Accuracy: {correct}/{len(data)} = {correct/len(data)*100:.2f}%")
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
