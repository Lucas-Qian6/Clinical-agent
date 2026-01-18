import json
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression


# Configuration
DEFAULT_DATA_FILE = "gemini-3-flash-preview.json"

# Set to None to use ALL data, or an integer (e.g., 500) for a quick run
CALIBRATION_SAMPLE_SIZE = 500 

def resolve_data_file(default_file):
    """Finds the dataset file from CLI args or defaults."""
    # 1. Check CLI args (ignoring flags)
    args = [a for a in sys.argv[1:] if not a.startswith('-') and 'jupyter' not in a]
    if args and os.path.exists(args[0]):
        return args[0]
    
    # 2. Check defaults
    candidates = [default_file, "dataset.json", "my_synthetic_dataset.jsonl"]
    for f in candidates:
        if os.path.exists(f):
            return f
            
    return default_file

def load_local_data(file_path):
    """Standard JSON Loader."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []

    print(f"📂 Loading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            valid_rows = [d for d in data if isinstance(d, dict)]
            print(f"✅ Successfully loaded {len(valid_rows)} rows.")
            return valid_rows
        else:
            print("❌ File content is not a JSON list (must start with '[').")
            return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []

def perform_huggingface_login():
    """Handles authentication for Gated Models (MentalBERT)."""
    if "HF_TOKEN" in os.environ: return
    try:
        from huggingface_hub import get_token, login
        if get_token(): return
        
        # Check Colab Secrets
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token:
                login(token=token)
                return
        except:
            pass
            
        print("\n⚠️  Note: MentalBERT requires login.")
        login()
    except:
        pass

def calibrate_weaver():
    perform_huggingface_login()
    
    print("\n⚙️ Initializing Clinical Jury...")
    verifiers = [
        ClinicalCorrectnessVerifier(device=0),
        TherapeuticToneVerifier(device=0),
        SafetyVerifier(device=0),
        ClinicalProtocolVerifier(),
        DialogueLogicVerifier(device=0)
    ]
    
    target_file = resolve_data_file(DEFAULT_DATA_FILE)
    raw_data = load_local_data(target_file)
    
    # Filter for valid pairs
    valid_data = [r for r in raw_data if r.get('chosen') and r.get('rejected')]
    
    # Apply Sample Size Limit
    if CALIBRATION_SAMPLE_SIZE and len(valid_data) > CALIBRATION_SAMPLE_SIZE:
        calibration_set = valid_data[:CALIBRATION_SAMPLE_SIZE]
        print(f"📉 Downsampling to first {CALIBRATION_SAMPLE_SIZE} pairs (out of {len(valid_data)}).")
    else:
        calibration_set = valid_data
        print(f"🚀 Using full dataset: {len(calibration_set)} pairs.")

    if not calibration_set:
        print("❌ No valid pairs found.")
        return

    print(f"🧪 Running inference...")
    
    X, y = [], []
    
    for i, row in enumerate(calibration_set):
        prompt = row.get('prompt', '') or row.get('instruction', '')
        chosen = row['chosen']
        rejected = row['rejected']

        # Label 1: Chosen
        feats_good = [
            verifiers[0].score(chosen, reference=chosen),     # Correctness
            verifiers[1].score(chosen),                       # Tone
            verifiers[2].score(chosen),                       # Safety
            verifiers[3].score(chosen),                       # Protocol
            verifiers[4].score(chosen, context=prompt)        # Logic
        ]
        X.append(feats_good)
        y.append(1)

        # Label 0: Rejected
        feats_bad = [
            verifiers[0].score(rejected, reference=chosen),   # Correctness
            verifiers[1].score(rejected),                     # Tone
            verifiers[2].score(rejected),                     # Safety
            verifiers[3].score(rejected),                     # Protocol
            verifiers[4].score(rejected, context=prompt)      # Logic
        ]
        X.append(feats_bad)
        y.append(0)

        if i % 50 == 0:
            print(f"   Processed {i} pairs...")

    # Train Regressor
    print("🧠 Training Regressor...")
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(np.array(X), np.array(y))
    
    raw_weights = clf.coef_[0]
    weights = np.maximum(raw_weights, 0.0)
    
    results = {}
    feature_names = [v.name for v in verifiers]
    scaling = 2.0 / max(weights) if max(weights) > 0 else 1.0
    
    print("\n" + "="*35)
    print("   OPTIMIZED WEAVER WEIGHTS")
    print("="*35)
    
    for name, w in zip(feature_names, weights):
        scaled_w = w * scaling
        print(f"{name:<25}: {scaled_w:.4f}")
        results[name] = scaled_w

    # --- HYBRID SAFETY OVERRIDE ---
    # Even if calibration says Safety/Correctness is 0.0 (because the dataset is safe),
    # we MUST enforce a floor for RL training to prevent drift.
    
    print("\n⚠️  Applying Safety Floor Override for Production...")
    
    production_weights = results.copy()
    
    # Floor definitions (Minimum acceptable weight for RL)
    floors = {
        "Safety_Guard": 1.0,           # Never let safety turn off
        "Clinical_Correctness": 0.5,   # Maintain some semantic anchor
        "Therapeutic_Tone": 0.5        # Maintain some tone check
    }
    
    for name, min_val in floors.items():
        if production_weights[name] < min_val:
            print(f"   ⬆️ Boosting {name} from {production_weights[name]:.2f} to {min_val:.2f}")
            production_weights[name] = min_val

    with open("weaver_weights.json", "w") as f:
        json.dump(production_weights, f, indent=2)
    print("\n💾 Saved PROTECTED weights to weaver_weights.json")

if __name__ == "__main__":
    calibrate_weaver()