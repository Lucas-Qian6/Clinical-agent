"""
Prepare PsychoCounsel-Preference for DPO training with emotion conditioning.

Loads Psychotherapy-LLM/PsychoCounsel-Preference, extracts 4D emotion vectors
from patient questions via BERT-GoEmotions + HG_MATRIX, and saves JSONL in
DPO format with emotion field.

Output format: {"prompt": "<|user|>\\n{question}\\n<|assistant|>\\n", "chosen": "...", "rejected": "...", "emotion": [I, T, A, S]}
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

# Add Landing/DPO to path for senticnet_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DPO"))
from senticnet_matrix import HG_MATRIX

# --- CONFIGURATION ---
DATASET_NAME = "Psychotherapy-LLM/PsychoCounsel-Preference"
OUTPUT_TRAIN = "Data/psychocounsel_dpo_train.jsonl"
OUTPUT_TEST = "Data/psychocounsel_dpo_test.jsonl"
DEVICE = 0 if torch.cuda.is_available() else -1

# Optional: filter for stronger preference pairs (chosen_empathy >= 4, rejected_empathy < 4)
USE_EMPATHY_FILTER = False


def get_clinical_vector(text: str, pipe) -> list:
    """Extract 4D emotion vector from text using BERT-GoEmotions + HG_MATRIX."""
    from senticnet_matrix import get_clinical_vector as _get_clinical_vector
    return _get_clinical_vector(text, pipe)


def format_prompt(question: str) -> str:
    """Wrap question in Llama 3 chat template."""
    return f"<|user|>\n{question}\n<|assistant|>\n"


def process_split(dataset, emo_pipe, output_path: Path, split_name: str) -> int:
    """Process a dataset split and write JSONL."""
    records = []
    for item in tqdm(dataset, desc=f"Processing {split_name}"):
        question = item.get("question", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if not question or not chosen or not rejected:
            continue

        if USE_EMPATHY_FILTER:
            c_emp = item.get("chosen_empathy_rating")
            r_emp = item.get("rejected_empathy_rating")
            if c_emp is not None and r_emp is not None:
                if c_emp < 4 or r_emp >= 4:
                    continue

        emotion = get_clinical_vector(question, emo_pipe)
        records.append({
            "prompt": format_prompt(question),
            "chosen": chosen,
            "rejected": rejected,
            "emotion": emotion,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def main():
    print("🚀 PsychoCounsel-Preference Data Preparation")
    print("=" * 50)

    script_dir = Path(__file__).resolve().parent
    output_train = script_dir / OUTPUT_TRAIN
    output_test = script_dir / OUTPUT_TEST

    print(f"📂 Loading dataset: {DATASET_NAME}")
    try:
        ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    print("🔌 Initializing BERT-GoEmotions pipeline...")
    emo_pipe = pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        return_all_scores=True,
        device=DEVICE,
    )

    n_train = 0
    n_test = 0
    if "train" in ds:
        n_train = process_split(ds["train"], emo_pipe, output_train, "train")
        print(f"   ✅ Train: {n_train} records -> {output_train}")
    if "test" in ds:
        n_test = process_split(ds["test"], emo_pipe, output_test, "test")
        print(f"   ✅ Test:  {n_test} records -> {output_test}")

    print("=" * 50)
    print("✅ Done.")
    print(f"   Train: {output_train}")
    print(f"   Test:  {output_test}")


if __name__ == "__main__":
    main()
