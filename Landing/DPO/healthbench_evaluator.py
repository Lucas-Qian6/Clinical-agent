from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import numpy as np
import json
import os

# IMPORT VERIFIERS
try:
    from weaver_ensembles import (
        ClinicalCorrectnessVerifier,
        TherapeuticToneVerifier,
        SafetyVerifier,
        ClinicalProtocolVerifier,
        DialogueLogicVerifier
    )
except ImportError:
    print("❌ Error: 'weaver_ensembles.py' not found.")
    exit(1)

# --- CONFIGURATION ---
BATCH_SIZE_GEN = 4    
BATCH_SIZE_EVAL = 32  
HOLDOUT_FILE = "dpo_holdout_dataset.jsonl"
WEIGHTS_FILE = "calibrated_weights_sequential.json" # <--- YOUR CALIBRATION RESULTS
MODEL_PATH = "clinical_rl_model_v1_lora"
DEVICE = 0 if torch.cuda.is_available() else -1

# --- DEFINITIONS ---
HG_MATRIX = {
    "joy": [1.0, 0, 0, 0], "excitement": [0.8, 0, 0, 0.3], "amusement": [0.6, 0, 0.3, 0],
    "pride": [0.7, 0, 0.2, 0], "relief": [0.4, 0.6, 0, 0],
    "sadness": [-1.0, 0, 0, 0], "grief": [-1.0, 0, 0, -0.2], "remorse": [-0.6, 0, -0.3, 0],
    "disappointment": [-0.6, -0.2, 0, 0],
    "anger": [0, -1.0, 0, 0], "annoyance": [0, -0.6, 0, 0], "disapproval": [0, -0.5, -0.4, 0],
    "admiration": [0.1, 0, 0.9, 0], "approval": [0, 0, 0.7, 0], "caring": [0.2, 0, 0.8, 0],
    "gratitude": [0.3, 0, 0.7, 0], "love": [0.5, 0, 0.5, 0], "desire": [0.2, 0, 0.5, 0.2],
    "disgust": [0, 0, -1.0, 0], "embarrassment": [-0.3, 0, -0.4, 0],
    "fear": [0, 0, 0, -1.0], "nervousness": [0, -0.2, 0, -0.6], "confusion": [0, 0, 0, -0.3],
    "curiosity": [0, 0, 0, 0.8], "optimism": [0.4, 0, 0, 0.5], "surprise": [0, 0, 0, 0.6],
    "realization": [0.1, 0, 0, 0.3], "neutral": [0, 0, 0, 0]
}

def get_clinical_vector(scores):
    vector = np.zeros(4)
    # Handle list of dicts (standard pipeline output)
    if isinstance(scores, list):
        for item in scores:
            if item['label'] in HG_MATRIX:
                vector += np.array(HG_MATRIX[item['label']]) * item['score']
    return vector

def main():
    # 1. LOAD CALIBRATED WEIGHTS
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ Error: {WEIGHTS_FILE} not found. Run calibration first!")
        return
    
    with open(WEIGHTS_FILE, 'r') as f:
        calib_data = json.load(f)
        
        # Extract Hourglass Weights [I, T, A, S]
        w_hg = np.array(calib_data['hourglass_weights'])
        
        # Extract Weaver Weights (Map name -> weight)
        # Note: The order must match how we iterate later
        w_weaver_dict = calib_data['weaver_weights']
        
        # Extract Global Ratio [Clinical, Weaver]
        w_global = np.array(calib_data['global_ratio'])
        
        print("⚖️  Loaded Calibrated Weights:")
        print(f"   Hourglass: {w_hg}")
        print(f"   Weaver: {json.dumps(w_weaver_dict, indent=2)}")
        print(f"   Global Ratio: {w_global}")

    # 2. LOAD DATA & MODEL
    print(f"\n📂 Loading data from {HOLDOUT_FILE}...")
    dataset = load_dataset("json", data_files=HOLDOUT_FILE, split="train")

    print("⚡ Loading Unsloth Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # 3. GENERATE RESPONSES
    print("⚡ Generating Responses (Batched)...")
    def generate_batch(batch):
        prompts = batch['prompt']
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"response": [text.replace(p, "").strip() for text, p in zip(decoded, prompts)]}

    ds_gen = dataset.map(generate_batch, batched=True, batch_size=BATCH_SIZE_GEN)

    # 4. INITIALIZE VERIFIERS
    print("\n🔌 Initializing Verifiers...")
    emo_pipe_hourglass = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True, device=DEVICE)
    
    # Initialize Weaver Modules
    verifiers = [
        ClinicalCorrectnessVerifier(device=DEVICE),
        TherapeuticToneVerifier(device=DEVICE),
        SafetyVerifier(device=DEVICE),
        ClinicalProtocolVerifier(),
        DialogueLogicVerifier(device=DEVICE)
    ]

    # 5. SCORING LOOP (Applying Weights)
    print("🔍 Scoring & Applying Weights...")
    final_scores = []
    clinical_subscores = []
    weaver_subscores = []
    
    # Iterating row-by-row (simpler for complex multi-model logic than .map)
    for i in tqdm(range(len(ds_gen))):
        row = ds_gen[i]
        resp = row['response']
        prompt = row['prompt']
        
        # --- A. CLINICAL SCORE (Hourglass) ---
        # 1. Get raw vector [I, T, A, S]
        raw_hg_output = emo_pipe_hourglass(resp[:512])[0]
        hg_vector = get_clinical_vector(raw_hg_output)
        
        # 2. Apply Calibrated Weights
        # Score = Dot(Vector, Weights)
        clin_score = np.dot(hg_vector, w_hg)
        clinical_subscores.append(clin_score)
        
        # --- B. WEAVER SCORE (Ensemble) ---
        # 1. Get scores for each verifier
        w_scores = []
        w_weights_ordered = []
        
        for v in verifiers:
            if v.name == "Clinical_Correctness":
                # Compare against itself for quality check
                s = v.score(resp, reference=resp) 
            elif v.name == "Dialogue_Logic":
                s = v.score(resp, context=prompt)
            else:
                s = v.score(resp)
            
            w_scores.append(s)
            w_weights_ordered.append(w_weaver_dict[v.name])
            
        # 2. Apply Calibrated Weights
        # We must normalize the weights for the weighted average if they don't sum to 1?
        # The calibration script output normalized weights (sum=1), so dot product is correct.
        weav_score = np.dot(np.array(w_scores), np.array(w_weights_ordered))
        weaver_subscores.append(weav_score)
        
        # --- C. GLOBAL SCORE ---
        # Combine Clinical + Weaver
        global_score = np.dot(np.array([clin_score, weav_score]), w_global)
        final_scores.append(global_score)

    # 6. SAVE RESULTS
    ds_final = ds_gen.add_column("clinical_subscore", clinical_subscores)
    ds_final = ds_final.add_column("weaver_subscore", weaver_subscores)
    ds_final = ds_final.add_column("final_score", final_scores)
    
    output_file = "final_evaluation_report.json"
    ds_final.to_json(output_file)
    
    print("\n" + "="*40)
    print("🏆 EVALUATION RESULTS")
    print("="*40)
    print(f"Average Final Score:    {np.mean(final_scores):.4f}")
    print(f"Avg Clinical Subscore:  {np.mean(clinical_subscores):.4f}")
    print(f"Avg Weaver Subscore:    {np.mean(weaver_subscores):.4f}")
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    main()