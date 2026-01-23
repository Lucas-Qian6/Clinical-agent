import json
import random
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_FILE = "final_dpo_training_set.json" # Updated to your actual file name
OUTPUT_TRAIN = "dpo_train_dataset.jsonl"
OUTPUT_TEST = "dpo_holdout_dataset.jsonl"

def apply_chat_template(prompt, response):
    """
    Wraps the raw text in a standard chat format (e.g., ChatML or Alpaca).
    Adjust this to match the specific base model you are using (Llama 3, Gemini, etc.)
    """
    # Example format for Llama-3 / Generic Chat models
    formatted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    return formatted_text

def load_data(filepath):
    """Smart loader that handles both JSON (list of dicts) and JSONL (one dict per line)"""
    print(f"Loading {filepath}...")
    ext = os.path.splitext(filepath)[1].lower()
    
    with open(filepath, 'r') as f:
        if ext == '.json':
            # Standard JSON Array
            return json.load(f)
        else:
            # JSONL (Line-delimited)
            return [json.loads(line) for line in f]

def prepare_data():
    data = load_data(INPUT_FILE)

    formatted_data = []
    
    print(f"Formatting {len(data)} pairs...")
    for entry in data:
        # Structure required by HuggingFace TRL:
        new_entry = {
            "prompt": f"<|user|>\n{entry['prompt']}\n<|assistant|>\n",
            "chosen": entry['chosen'],   # The Good Clinical Response
            "rejected": entry['rejected'] # The Bad/Toxic Response
        }
        formatted_data.append(new_entry)

    # --- THE SPLIT ---
    # We save 10% (or 500 samples) as the 'Holdout Set'.
    # This is the exam the model has never seen.
    train_data, test_data = train_test_split(formatted_data, test_size=0.10, random_state=42)
    
    print(f"Saving {len(train_data)} training samples to {OUTPUT_TRAIN}...")
    with open(OUTPUT_TRAIN, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saving {len(test_data)} holdout samples to {OUTPUT_TEST}...")
    with open(OUTPUT_TEST, 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')

    print("\nSUCCESS. You are ready for training.")
    print(f"Use '{OUTPUT_TRAIN}' for the DPO Trainer.")
    print(f"Use '{OUTPUT_TEST}' for your Weekly Benchmark Report.")

if __name__ == "__main__":
    prepare_data()