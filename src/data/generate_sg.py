import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

from src.utils.helpers import load_config


# def generate_prompt(question):
#     return f"""
# You are generating solution guidance for a math word problem.

# STRICT INSTRUCTIONS:

# - DO NOT solve the problem.
# - DO NOT perform calculations.
# - DO NOT include intermediate values.
# - DO NOT include final answers.
# - DO NOT copy numerical results from the question.
# - ONLY describe the reasoning process required to solve the problem.

# FORMAT RULES:

# - Generate EXACTLY 2 to 5 steps.
# - Each step must start with: Step X [OPERATION]:
# - Replace X with the step number.
# - OPERATION must be one of: ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE, COUNT, CONVERT, RATIO
# - Each step must contain exactly one short sentence.
# - Describe WHAT should be computed, not the result.
# - Preserve dependencies between steps.
# - No explanations, notes, or extra text before or after the steps.

# GOOD EXAMPLE:

# Step 1 [DIVIDE]: Determine the quantity represented by the given fraction.
# Step 2 [MULTIPLY]: Determine the related quantity using the stated factor.
# Step 3 [ADD]: Combine the relevant quantities.
# Step 4 [SUBTRACT]: Determine the remaining amount.

# BAD EXAMPLES:

# Step 1: Calculate half of 100.
# Step 2: The answer is 50.

# Step 1 [DIVIDE]: Compute 48 divided by 2.
# Step 2 [ADD]: Add 48 and 24.

# Question:
# {question}
# """

def generate_prompt(question, solution):
    return f"""
You are generating Solution Guidance (SG) for training a Small Language Model (SLM).

Your goal is to convert a worked mathematical solution into a high-level reasoning plan.

The worked solution is provided ONLY as hidden reference to identify the correct reasoning sequence.
It MUST NOT appear in the output.

==================================================
QUESTION
==================================================

{question}

==================================================
REFERENCE SOLUTION (Hidden Context)
==================================================

{solution}

==================================================
TASK
==================================================

Generate an abstract reasoning guide that follows the SAME reasoning sequence and level of decomposition as the reference solution.

Each reasoning step in the reference solution should normally correspond to one guidance step unless two consecutive statements describe the exact same reasoning action.

The guidance should describe WHAT needs to be determined at each step while hiding every numerical computation.

==================================================
STRICT RULES
==================================================

DO NOT:

- Solve the problem.
- Perform calculations.
- Include equations.
- Include arithmetic expressions.
- Include any numbers.
- Include intermediate values.
- Include the final answer.
- Copy complete sentences from the reference solution.

Instead:

- Preserve the logical reasoning order.
- Preserve the reasoning decomposition of the reference solution. Do not skip, combine, or omit reasoning steps that appear as distinct logical steps in the reference solution unless they are semantically identical.
- Preserve the semantic entities from the problem whenever appropriate.
  Examples:
    - grandparents' contribution
    - pages remaining
    - total earnings
    - clips sold in May
    - remaining amount needed

- Abstract only the computations, NOT the reasoning.

==================================================
FORMAT
==================================================

Generate EXACTLY 2 to 5 steps.

Each step MUST begin with:

Step X [OPERATION]:

where OPERATION is EXACTLY one of:

ADD
SUBTRACT
MULTIPLY
DIVIDE
COMPARE
COUNT
CONVERT
RATIO

Each step must:

- contain exactly one sentence
- describe one reasoning action
- preserve dependencies between previous steps
- correspond to one reasoning step in the reference solution
- not merge unrelated reasoning steps

Output ONLY the steps.

==================================================
GOOD EXAMPLE
==================================================

Step 1 [DIVIDE]: Determine the number of clips sold during the second month using the given proportional relationship.

Step 2 [ADD]: Combine the clips sold during both months to determine the total.

--------------------------------------------------

Step 1 [DIVIDE]: Determine Betty's initial savings.

Step 2 [MULTIPLY]: Determine the grandparents' contribution.

Step 3 [ADD]: Determine the total funds available.

Step 4 [SUBTRACT]: Determine the remaining amount needed.

==================================================
BAD EXAMPLES
==================================================

Step 1 [DIVIDE]: Compute 48 divided by 2.

Step 1 [DIVIDE]: Half of 100 is 50.

Step 1 [ADD]: Add 48 and 24.

Step 1 [DIVIDE]: Determine the quantity.

Step 2 [ADD]: Determine another quantity.

==================================================
FINAL CHECK
==================================================

Before returning the answer, verify that:

✓ No numbers appear.
✓ No equations appear.
✓ No arithmetic expressions appear.
✓ No intermediate values appear.
✓ No final answer appears.
✓ Only the allowed operations are used.
✓ The reasoning order matches the reference solution.
✓ The number of guidance steps closely matches the number of logical reasoning steps in the reference solution (within the required limit of 2–5 steps).
✓ No logical reasoning step from the reference solution has been omitted or merged unnecessarily.
✓ Important semantic entities from the problem are preserved.
✓ The output contains nothing except the formatted steps.

If any rule is violated, silently correct it before producing the final answer.
"""

def generate_sg(client, model, question, answer):
    prompt = generate_prompt(question, answer)

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
            answer = item["answer"]

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
                        sg = generate_sg(client, model, question, answer)

                        # if r"^\s*Step\s+1\s+\[[^\]]+\]:" not in sg:
                        #     raise Exception("Invalid format")

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