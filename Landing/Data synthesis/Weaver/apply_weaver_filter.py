import json
import os
import sys
import numpy as np
from weaver_ensemble import (
    WeaverEnsemble,
    ClinicalCorrectnessVerifier,
    TherapeuticToneVerifier,
    SafetyVerifier,
    ClinicalProtocolVerifier,
    DialogueLogicVerifier,
    WeakVerifier
)

# --- CONFIGURATION ---
INPUT_FILE = "gemini-3-flash-preview.json"
WEIGHTS_FILE = "weaver_weights.json"
OUTPUT_FILE = "final_dpo_training_set.jsonl"

# Filtering Thresholds (The "Sweet Spot")
MIN_MARGIN = 0.10          # Chosen must be at least 10% better than Rejected
MIN_REJECTED_SCORE = 0.15  # Rejected must not be total gibberish
MAX_REJECTED_SCORE = 0.70  # Rejected must not be "actually good"

def load_weights(path):
    """Loads weights from JSON or returns defaults."""
    defaults = {
        "Clinical_Protocol": 2.0,
        "Dialogue_Logic": 1.44,
        "Safety_Guard": 1.0,
        "Clinical_Correctness": 0.5,
        "Therapeutic_Tone": 0.5
    }
    if os.path.exists(path):
        print(f"⚖️  Loading weights from {path}...")
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                for k, v in loaded.items():
                    defaults[k] = v
        except Exception as e:
            print(f"⚠️  Error reading weights file: {e}. Using defaults.")
    else:
        print("⚠️  Weights file not found. Using defaults.")
    
    return defaults

def load_data_robust(file_path):
    """Robustly loads JSON or JSONL data."""
    if not os.path.exists(file_path):
        print(f"❌ Input file not found: {file_path}")
        return []

    print(f"📂 Reading {file_path}...")
    data = []
    
    # Try loading as standard JSON list
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Handle concatenated objects (fix_json_format style) or list
            if content.startswith('['):
                data = json.loads(content)
            else:
                # Stream parsing for concatenated objects
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(content):
                    while pos < len(content) and content[pos].isspace(): pos += 1
                    if pos >= len(content): break
                    obj, next_pos = decoder.raw_decode(content, pos)
                    if isinstance(obj, dict): data.append(obj)
                    pos = next_pos
    except Exception as e:
        print(f"⚠️  JSON Parse Error: {e}. Trying line-by-line fallback...")
        # Fallback
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip(): data.append(json.loads(line))
                except: pass

    # Filter for valid rows
    valid_data = [d for d in data if isinstance(d, dict) and d.get('chosen') and d.get('rejected')]
    print(f"✅ Loaded {len(valid_data)} valid pairs.")
    return valid_data

def perform_huggingface_login():
    """Authenticates for MentalBERT."""
    if "HF_TOKEN" in os.environ: return
    try:
        from huggingface_hub import get_token, login
        if get_token(): return
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token: login(token=token)
    except:
        pass

def main():
    perform_huggingface_login()

    # 1. Setup Jury
    weights = load_weights(WEIGHTS_FILE)
    print("\n⚙️  Initializing Weaver Jury with Production Weights:")
    for k, v in weights.items():
        print(f"   • {k:<25}: {v:.2f}")

    verifiers = [
        ClinicalCorrectnessVerifier(device=0),
        TherapeuticToneVerifier(device=0),
        SafetyVerifier(device=0),
        ClinicalProtocolVerifier(),
        DialogueLogicVerifier(device=0)
    ]

    # Apply weights
    for v in verifiers:
        if v.name in weights:
            v.weight = weights[v.name]

    ensemble = WeaverEnsemble(verifiers)

    # 2. Process Data
    raw_data = load_data_robust(INPUT_FILE)
    if not raw_data: return

    print(f"\n🧪 Scoring {len(raw_data)} pairs (this may take a while)...")
    
    kept_count = 0
    rejected_reasons = {"small_margin": 0, "rejected_too_good": 0, "rejected_gibberish": 0}
    final_dataset = []

    for i, row in enumerate(raw_data):
        prompt = row.get('prompt') or row.get('instruction')
        chosen = row['chosen']
        rejected = row['rejected']

        # Run Weaver
        result = ensemble.evaluate_pair(chosen, rejected, prompt)
        
        # Filtering Logic
        keep = True
        reason = ""

        if result['margin'] < MIN_MARGIN:
            keep = False
            rejected_reasons['small_margin'] += 1
            reason = "Margin too small (Bad ~= Good)"
        elif result['rejected_score'] > MAX_REJECTED_SCORE:
            keep = False
            rejected_reasons['rejected_too_good'] += 1
            reason = "Rejected answer is actually good"
        elif result['rejected_score'] < MIN_REJECTED_SCORE:
            keep = False
            rejected_reasons['rejected_gibberish'] += 1
            reason = "Rejected answer is gibberish/unsafe"

        if keep:
            # Add metadata for debugging/analysis
            row['weaver_score'] = {
                'margin': round(result['margin'], 3),
                'chosen_score': round(result['chosen_score'], 3),
                'rejected_score': round(result['rejected_score'], 3)
            }
            final_dataset.append(row)
            kept_count += 1
        
        # Progress bar
        if i % 50 == 0:
            sys.stdout.write(f"\r   Processed {i}/{len(raw_data)} | Kept: {kept_count}")
            sys.stdout.flush()

    print(f"\n\n📊 Filtering Complete!")
    print(f"   Original: {len(raw_data)}")
    print(f"   Final:    {kept_count} ({kept_count/len(raw_data)*100:.1f}%)")
    print(f"\n   Drop Reasons:")
    print(f"   • 'Bad' was too similar to 'Good': {rejected_reasons['small_margin']}")
    print(f"   • 'Bad' was actually Good:         {rejected_reasons['rejected_too_good']}")
    print(f"   • 'Bad' was gibberish/unsafe:      {rejected_reasons['rejected_gibberish']}")

    # 3. Save
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
            
    print("✅ Done. Your dataset is ready for DPO training.")

if __name__ == "__main__":
    main()