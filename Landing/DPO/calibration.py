import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# IMPORT YOUR VERIFIERS
try:
    from weaver_ensembles import (
        ClinicalCorrectnessVerifier,
        TherapeuticToneVerifier,
        SafetyVerifier,
        ClinicalProtocolVerifier,
        DialogueLogicVerifier
    )
except ImportError:
    print("❌ Error: 'weaver_ensembles.py' not found. Please ensure it is in the same directory.")
    exit(1)

# --- CONFIGURATION ---
HOLDOUT_FILE = "dpo_holdout_dataset.jsonl"
CALIB_OUTPUT_FILE = "calibrated_weights_sequential.json"
SUBSET_SIZE = 100 
DEVICE = 0 if torch.cuda.is_available() else -1

# --- DEFINITIONS (Hourglass) ---
# COMPLETE 28-LABEL MAPPING for GoEmotions
# Format: [Introspection, Temper, Attitude, Sensitivity]
HG_MATRIX = {
    # --- POSITIVE INTROSPECTION (Joy) ---
    "joy":          [1.0, 0.0, 0.0, 0.0],
    "excitement":   [0.8, 0.0, 0.0, 0.3], # Joy + Sensitivity
    "amusement":    [0.6, 0.0, 0.3, 0.0], # Joy + Attitude
    "pride":        [0.7, 0.0, 0.2, 0.0], # Joy + Self-Attitude
    "relief":       [0.4, 0.6, 0.0, 0.0], # Joy + Calmness (Temper)
    
    # --- NEGATIVE INTROSPECTION (Sadness) ---
    "sadness":      [-1.0, 0.0, 0.0, 0.0],
    "grief":        [-1.0, 0.0, 0.0, -0.2], # Deep Sadness + Withdrawal
    "remorse":      [-0.6, 0.0, -0.3, 0.0], # Sadness + Self-Disgust (Attitude)
    "disappointment": [-0.6, -0.2, 0.0, 0.0], # Sadness + Slight Anger
    
    # --- NEGATIVE TEMPER (Anger) ---
    "anger":        [0.0, -1.0, 0.0, 0.0],
    "annoyance":    [0.0, -0.6, 0.0, 0.0], # Mild Anger
    "disapproval":  [0.0, -0.5, -0.4, 0.0], # Temper + Attitude
    
    # --- POSITIVE ATTITUDE (Pleasantness) ---
    "admiration":   [0.1, 0.0, 0.9, 0.0], # High Attitude
    "approval":     [0.0, 0.0, 0.7, 0.0],
    "caring":       [0.2, 0.0, 0.8, 0.0], # Attitude + Slight Joy
    "gratitude":    [0.3, 0.0, 0.7, 0.0], # Attitude + Joy
    "love":         [0.5, 0.0, 0.5, 0.0], # Strong Joy + Attitude
    "desire":       [0.2, 0.0, 0.5, 0.2], # Attitude + Sensitivity
    
    # --- NEGATIVE ATTITUDE (Disgust) ---
    "disgust":      [0.0, 0.0, -1.0, 0.0],
    "embarrassment":[ -0.3, 0.0, -0.4, 0.0], # Negative Self-Attitude
    
    # --- NEGATIVE SENSITIVITY (Fear) ---
    "fear":         [0.0, 0.0, 0.0, -1.0],
    "nervousness":  [0.0, -0.2, 0.0, -0.6], # Fear + Unstable Temper
    "confusion":    [0.0, 0.0, 0.0, -0.3], # Low Sensitivity
    
    # --- POSITIVE SENSITIVITY (Eagerness) ---
    "curiosity":    [0.0, 0.0, 0.0, 0.8],
    "optimism":     [0.4, 0.0, 0.0, 0.5], # Joy + Sensitivity
    "surprise":     [0.0, 0.0, 0.0, 0.6], # Neutral Sensitivity spike
    "realization":  [0.1, 0.0, 0.0, 0.3],
    
    # --- NEUTRAL ---
    "neutral":      [0.0, 0.0, 0.0, 0.0]
}

def get_clinical_vector(text, pipe):
    output = pipe(text[:512])[0] 
    vector = np.zeros(4)
    for item in output:
        if item['label'] in HG_MATRIX:
            vector += np.array(HG_MATRIX[item['label']]) * item['score']
    return vector

def fit_regression(X_diffs):
    X_sym = np.concatenate([X_diffs, -X_diffs], axis=0)
    y_sym = np.concatenate([np.ones(len(X_diffs)), np.zeros(len(X_diffs))])
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X_sym, y_sym)
    weights = clf.coef_[0]
    return weights / np.sum(np.abs(weights)) # Normalize

# --- MAIN WORKFLOW ---
def main():
    print(f"📂 Loading {HOLDOUT_FILE}...")
    try:
        dataset = load_dataset("json", data_files=HOLDOUT_FILE, split="train")
        if SUBSET_SIZE: dataset = dataset.select(range(min(len(dataset), SUBSET_SIZE)))
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print("🔌 Initializing Models...")
    
    # 1. Hourglass Model (GoEmotions)
    emo_pipe_hourglass = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True, device=DEVICE)

    # 2. Weaver Verifiers
    print("   Initializing Weaver Ensemble modules...")
    verifiers = [
        ClinicalCorrectnessVerifier(device=DEVICE),
        TherapeuticToneVerifier(device=DEVICE),
        SafetyVerifier(device=DEVICE),
        ClinicalProtocolVerifier(),
        DialogueLogicVerifier(device=DEVICE)
    ]
    v_names = [v.name for v in verifiers]
    print(f"   Loaded: {v_names}")

    # Collections
    clinical_diffs = [] # Shape (N, 4)
    weaver_diffs = []   # Shape (N, 5) 

    print(f"⚡ Extracting Features for {len(dataset)} pairs...")
    for item in tqdm(dataset):
        c_txt, r_txt = item['chosen'], item['rejected']
        prompt = item['prompt']
        
        # --- A. Hourglass Features ---
        c_hg = get_clinical_vector(c_txt, emo_pipe_hourglass)
        r_hg = get_clinical_vector(r_txt, emo_pipe_hourglass)
        clinical_diffs.append(c_hg - r_hg)
        
        # --- B. Weaver Features ---
        c_scores = []
        r_scores = []
        
        for v in verifiers:
            if v.name == "Clinical_Correctness":
                s_c = v.score(c_txt, reference=c_txt) 
                s_r = v.score(r_txt, reference=c_txt)
            elif v.name == "Dialogue_Logic":
                s_c = v.score(c_txt, context=prompt)
                s_r = v.score(r_txt, context=prompt)
            else:
                s_c = v.score(c_txt)
                s_r = v.score(r_txt)
            
            c_scores.append(s_c)
            r_scores.append(s_r)
            
        weaver_diffs.append(np.array(c_scores) - np.array(r_scores))

    clinical_diffs = np.array(clinical_diffs)
    weaver_diffs = np.array(weaver_diffs)

    print("\n" + "="*40)
    print("⚖️  STARTING SEQUENTIAL CALIBRATION")
    print("="*40)

    # --- STEP 1: CALIBRATE CLINICAL MATRIX ---
    print("\n1️⃣  Optimizing Clinical Matrix [I, T, A, S]...")
    w_hg = fit_regression(clinical_diffs)
    print(f"   I:{w_hg[0]:.2f}, T:{w_hg[1]:.2f}, A:{w_hg[2]:.2f}, S:{w_hg[3]:.2f}")

    # --- STEP 2: CALIBRATE WEAVER EXPERTS ---
    print(f"\n2️⃣  Optimizing Weaver Experts {v_names}...")
    w_weav = fit_regression(weaver_diffs)
    for name, weight in zip(v_names, w_weav):
        print(f"   {name:<25}: {weight:.2f}")

    # --- STEP 3: CALIBRATE GLOBAL RATIO ---
    print("\n3️⃣  Optimizing Global Ratio [Clinical vs Weaver]...")
    
    opt_hg_scores = np.dot(clinical_diffs, w_hg)
    opt_weav_scores = np.dot(weaver_diffs, w_weav)
    
    global_features = np.column_stack([opt_hg_scores, opt_weav_scores])
    w_global = fit_regression(global_features)
    
    print(f"   Clinical Influence : {w_global[0]:.2f}")
    print(f"   Weaver Influence   : {w_global[1]:.2f}")

    # --- SAVE ---
    results = {
        "hourglass_weights": list(w_hg),
        "weaver_weights": {name: float(w) for name, w in zip(v_names, w_weav)},
        "global_ratio": list(w_global)
    }
    with open(CALIB_OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Saved to {CALIB_OUTPUT_FILE}")

if __name__ == "__main__":
    main()