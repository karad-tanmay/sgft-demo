import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

from src.utils.helpers import load_config


def generate_prompt(question):
    return f"""
You are generating solution guidance for a math word problem.

STRICT INSTRUCTIONS:
- DO NOT solve the problem
- DO NOT include calculations or numbers
- DO NOT provide final answer
- ONLY describe reasoning steps

FORMAT RULES:
- EXACTLY 2 to 5 steps
- Each step must start with: Step X:
- Each step must be one short sentence
- No extra text before or after steps

VALID FORMAT:
Step 1: ...
Step 2: ...
Step 3: ...

Question:
{question}
"""


def generate_sg(client, model, question):
    prompt = generate_prompt(question)

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text.strip()


def main():
    config = load_config()

    load_dotenv()
    api_keys_env = os.getenv("GEMINI_API_KEYS")
    if not api_keys_env:
        raise ValueError("GEMINI_API_KEYS not found")

    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]

    models = config["models"]["gemini"]

    input_file = config["data"]["input_file"]
    output_file = config["data"]["output_file"]
    target_samples = config["data"]["num_samples"]

    sleep_time = config["generation"]["sleep_per_request"]
    exhaust_sleep = config["generation"]["sleep_on_exhaust"]

    # ------------------ NEW: per-key blocked models ------------------
    blocked_models = {key: set() for key in api_keys}
    # ----------------------------------------------------------------

    with open(input_file, "r") as f:
        data = json.load(f)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            sg_data = json.load(f)
        print(f"Resuming from {len(sg_data)} samples")
    else:
        sg_data = []

    processed_questions = set(item["input"] for item in sg_data)

    print(f"Target samples: {target_samples}")

    try:
        for item in tqdm(data):

            if len(sg_data) >= target_samples:
                print("Target reached")
                break

            question = item["question"]

            if question in processed_questions:
                continue

            success = False

            for api_key in api_keys:

                # ------------------ NEW: skip key if all models blocked ------------------
                if len(blocked_models[api_key]) == len(models):
                    continue
                # ------------------------------------------------------------------------

                client = genai.Client(api_key=api_key)

                for model in models:

                    # ------------------ NEW: skip blocked model ------------------
                    if model in blocked_models[api_key]:
                        continue
                    # ------------------------------------------------------------

                    try:
                        sg = generate_sg(client, model, question)

                        if "Step 1:" not in sg:
                            raise Exception("Invalid format")

                        sg_data.append({
                            "input": question,
                            "output": sg
                        })

                        processed_questions.add(question)

                        with open(output_file, "w") as f:
                            json.dump(sg_data, f, indent=2)

                        print(f"Saved: {len(sg_data)}")

                        time.sleep(sleep_time)

                        success = True
                        break

                    except Exception as e:
                        err = str(e)
                        print(f"Failed [{model}]: {e}")

                        # ------------------ NEW: detect RPD exhaustion ------------------
                        if "quota" in err.lower() and "perday" in err.lower():
                            blocked_models[api_key].add(model)
                            print(f"Blocking {model} for this API key (RPD exhausted)")
                        # ---------------------------------------------------------------

                        continue

                if success:
                    break

            if not success:
                print("All keys/models exhausted. Sleeping...")
                time.sleep(exhaust_sleep)

    except KeyboardInterrupt:
        print("Interrupted. Progress saved.")

    print("Done")


if __name__ == "__main__":
    main()