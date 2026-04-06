from datasets import load_dataset
import json
import os

# Create folders if not exist
os.makedirs("data/raw", exist_ok=True)

# Load dataset
dataset = load_dataset("gsm8k", "main")

train_data = dataset["train"]
test_data = dataset["test"]

# Convert to simple JSON format
def convert(split):
    data_list = []
    for item in split:
        data_list.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    return data_list

train_json = convert(train_data)
test_json = convert(test_data)

# Save files
with open("data/raw/gsm8k_train.json", "w") as f:
    json.dump(train_json, f, indent=2)

with open("data/raw/gsm8k_test.json", "w") as f:
    json.dump(test_json, f, indent=2)

print("GSM8K dataset downloaded and saved")