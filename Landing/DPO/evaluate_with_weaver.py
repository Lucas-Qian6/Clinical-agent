# Weaver-Based Evaluation for DPO Model
# Compares Base Llama 3 vs Fine-tuned Model using Clinical Quality Metrics

import sys
import os
sys.path.append('../Data synthesis/Weaver')

from weaver_ensembles import (
    WeaverEnsemble,
    ClinicalCorrectnessVerifier,
    TherapeuticToneVerifier,
    SafetyVerifier,
    ClinicalProtocolVerifier,
    DialogueLogicVerifier
)
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
import torch

print("="*60)
print("🎯 CLINICAL DPO EVALUATION WITH WEAVER")
print("="*60)

# ============================================================================
# CONFIGURATION
# ============================================================================
HOLDOUT_DATA = "Data/dpo_holdout_dataset.jsonl"  # 59 test pairs
BASE_MODEL = "unsloth/llama-3-8b-instruct-bnb-4bit"
FINETUNED_MODEL = "clinical_dpo_model_v1"  # Your trained model
WEAVER_WEIGHTS = "../Data synthesis/Weaver/weaver_weights.json"
OUTPUT_FILE = "evaluation_results.json"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
DEVICE = 0 if torch.cuda.is_available() else -1

# ============================================================================
# 1. INITIALIZE WEAVER ENSEMBLE
# ============================================================================
print("\n📊 Initializing Weaver Jury (5 verifiers)...")

verifiers = [
    ClinicalCorrectnessVerifier(device=DEVICE),
    TherapeuticToneVerifier(device=DEVICE),
    SafetyVerifier(device=DEVICE),
    ClinicalProtocolVerifier(),
    DialogueLogicVerifier(device=DEVICE)
]

# Load trained weights if available
if os.path.exists(WEAVER_WEIGHTS):
    print(f"   Loading weights from {WEAVER_WEIGHTS}...")
    try:
        with open(WEAVER_WEIGHTS, 'r') as f:
            weights = json.load(f)
            for v in verifiers:
                if v.name in weights:
                    v.weight = weights[v.name]
                    print(f"   • {v.name}: {v.weight:.2f}")
    except Exception as e:
        print(f"   ⚠️  Error loading weights: {e}. Using defaults.")
else:
    print("   ⚠️  Weights file not found. Using default weights:")
    for v in verifiers:
        print(f"   • {v.name}: {v.weight:.2f}")

ensemble = WeaverEnsemble(verifiers)
print("   ✅ Weaver ensemble ready")

# ============================================================================
# 2. LOAD MODELS
# ============================================================================
print("\n🤖 Loading models...")

# Base model
print("   Loading base Llama 3 8B...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(base_model)
print("   ✅ Base model loaded")

# Fine-tuned model
print(f"   Loading fine-tuned model from {FINETUNED_MODEL}...")
try:
    finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(finetuned_model)
    print("   ✅ Fine-tuned model loaded")
except Exception as e:
    print(f"   ❌ Error loading fine-tuned model: {e}")
    print("   Make sure you've trained and saved the model first")
    raise

# ============================================================================
# 3. LOAD HOLDOUT DATASET
# ============================================================================
print(f"\n📋 Loading holdout dataset from {HOLDOUT_DATA}...")
try:
    holdout = load_dataset("json", data_files=HOLDOUT_DATA, split="train")
    print(f"   ✅ Loaded {len(holdout)} holdout samples")
except Exception as e:
    print(f"   ❌ Error loading holdout: {e}")
    raise

# ============================================================================
# 4. GENERATE RESPONSES & SCORE WITH WEAVER
# ============================================================================
print("\n🔬 Generating responses and scoring with Weaver...")
print(f"   This will take ~{len(holdout) * 10} seconds (2 models × {len(holdout)} samples)")

results = []
individual_scores = {
    'base': {'correctness': [], 'tone': [], 'safety': [], 'protocol': [], 'logic': []},
    'finetuned': {'correctness': [], 'tone': [], 'safety': [], 'protocol': [], 'logic': []}
}

for idx, example in enumerate(tqdm(holdout, desc="Evaluating")):
    prompt = example['prompt']
    chosen_gold = example['chosen']
    rejected_gold = example['rejected']
    
    # Extract clean user prompt (remove template)
    user_text = prompt.replace("<|user|>", "").replace("<|assistant|>", "").strip()
    if "\n" in user_text:
        user_text = user_text.split("\n")[0].strip()
    
    # ========== GENERATE FROM BASE MODEL ==========
    base_inputs = base_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        base_outputs = base_model.generate(
            **base_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=base_tokenizer.eos_token_id,
            eos_token_id=base_tokenizer.eos_token_id
        )
    base_response = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    base_response = base_response.split("<|assistant|>")[-1].strip()
    
    # ========== GENERATE FROM FINE-TUNED MODEL ==========
    ft_inputs = finetuned_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ft_outputs = finetuned_model.generate(
            **ft_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=finetuned_tokenizer.eos_token_id,
            eos_token_id=finetuned_tokenizer.eos_token_id
        )
    ft_response = finetuned_tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
    ft_response = ft_response.split("<|assistant|>")[-1].strip()
    
    # ========== SCORE WITH WEAVER ==========
    # Compare each response against gold "chosen" response
    base_eval = ensemble.evaluate_pair(chosen_gold, base_response, user_text)
    ft_eval = ensemble.evaluate_pair(chosen_gold, ft_response, user_text)
    
    # Store detailed scores for each verifier
    # (Note: Weaver doesn't expose individual verifier scores by default,
    #  so we'll compute aggregate scores. Modify weaver_ensembles.py if needed)
    
    # Calculate improvement
    improvement = ft_eval['rejected_score'] - base_eval['rejected_score']
    
    # Store result
    result = {
        "sample_id": idx,
        "prompt": user_text[:200],  # Truncate for readability
        "gold_chosen": chosen_gold[:200],
        "base_response": base_response,
        "finetuned_response": ft_response,
        "base_score": float(base_eval['rejected_score']),
        "finetuned_score": float(ft_eval['rejected_score']),
        "improvement": float(improvement),
        "base_margin": float(base_eval['margin']),
        "ft_margin": float(ft_eval['margin']),
        "win": improvement > 0
    }
    results.append(result)

print("\n   ✅ Evaluation complete!")

# ============================================================================
# 5. CALCULATE AGGREGATE METRICS
# ============================================================================
print("\n📊 Computing aggregate metrics...")

wins = sum(1 for r in results if r['win'])
losses = sum(1 for r in results if not r['win'])
win_rate = wins / len(results) * 100

avg_base = np.mean([r['base_score'] for r in results])
avg_ft = np.mean([r['finetuned_score'] for r in results])
avg_improvement = avg_ft - avg_base

std_base = np.std([r['base_score'] for r in results])
std_ft = np.std([r['finetuned_score'] for r in results])

# Find best and worst improvements
sorted_results = sorted(results, key=lambda x: x['improvement'], reverse=True)
top_5_improvements = sorted_results[:5]
bottom_5_improvements = sorted_results[-5:]

# ============================================================================
# 6. PRINT RESULTS
# ============================================================================
print("\n" + "="*60)
print("🎯 EVALUATION RESULTS")
print("="*60)
print(f"\n📈 Overall Performance:")
print(f"   Win Rate (Fine-tuned > Base):  {win_rate:.1f}% ({wins}/{len(results)})")
print(f"   Loss Rate (Fine-tuned ≤ Base): {100-win_rate:.1f}% ({losses}/{len(results)})")
print(f"\n📊 Weaver Scores:")
print(f"   Base Model Average:            {avg_base:.3f} (±{std_base:.3f})")
print(f"   Fine-tuned Model Average:      {avg_ft:.3f} (±{std_ft:.3f})")
print(f"   Average Improvement:           {'+' if avg_improvement > 0 else ''}{avg_improvement:.3f}")

# Success criteria check
print("\n🎯 Success Criteria:")
success_criteria = []

if win_rate > 70:
    print("   ✅ Win Rate > 70%")
    success_criteria.append(True)
else:
    print(f"   ❌ Win Rate ≤ 70% (got {win_rate:.1f}%)")
    success_criteria.append(False)

if avg_improvement > 0.10:
    print("   ✅ Avg Improvement > 0.10")
    success_criteria.append(True)
else:
    print(f"   ❌ Avg Improvement ≤ 0.10 (got {avg_improvement:.3f})")
    success_criteria.append(False)

print("="*60)

if all(success_criteria):
    print("🎉 SUCCESS: Model shows significant clinical improvement!")
elif win_rate > 60:
    print("⚠️  PARTIAL SUCCESS: Model improved but below target")
    print("   Consider: (1) More training data, (2) More epochs, (3) Different hyperparameters")
else:
    print("❌ NEEDS IMPROVEMENT: Model did not show consistent improvement")
    print("   Recommended actions:")
    print("   1. Check training loss curve - did it converge?")
    print("   2. Inspect examples below to understand failure modes")
    print("   3. Consider increasing dataset size")

print("="*60)

# ============================================================================
# 7. SHOW QUALITATIVE EXAMPLES
# ============================================================================
print("\n📝 Top 5 Improvements:\n")
for i, ex in enumerate(top_5_improvements, 1):
    print(f"{'─'*60}")
    print(f"EXAMPLE {i} | Improvement: +{ex['improvement']:.3f}")
    print(f"{'─'*60}")
    print(f"👤 PATIENT:\n{ex['prompt']}\n")
    print(f"🤖 BASE MODEL (score={ex['base_score']:.3f}):\n{ex['base_response'][:250]}...\n")
    print(f"✨ FINE-TUNED (score={ex['finetuned_score']:.3f}):\n{ex['finetuned_response'][:250]}...\n")

print(f"\n📝 Bottom 5 (Worst Performance):\n")
for i, ex in enumerate(bottom_5_improvements, 1):
    print(f"{'─'*60}")
    print(f"EXAMPLE {i} | Change: {ex['improvement']:+.3f}")
    print(f"{'─'*60}")
    print(f"👤 PATIENT:\n{ex['prompt']}\n")
    print(f"🤖 BASE MODEL (score={ex['base_score']:.3f}):\n{ex['base_response'][:250]}...\n")
    print(f"✨ FINE-TUNED (score={ex['finetuned_score']:.3f}):\n{ex['finetuned_response'][:250]}...\n")

# ============================================================================
# 8. SAVE DETAILED RESULTS
# ============================================================================
print(f"\n💾 Saving detailed results to {OUTPUT_FILE}...")

output_data = {
    "summary": {
        "total_samples": len(results),
        "win_rate": float(win_rate),
        "wins": wins,
        "losses": losses,
        "avg_base_score": float(avg_base),
        "avg_finetuned_score": float(avg_ft),
        "avg_improvement": float(avg_improvement),
        "std_base_score": float(std_base),
        "std_finetuned_score": float(std_ft),
        "success_criteria_met": all(success_criteria)
    },
    "detailed_results": results,
    "top_5_improvements": [
        {"sample_id": ex['sample_id'], "improvement": ex['improvement']} 
        for ex in top_5_improvements
    ],
    "bottom_5_improvements": [
        {"sample_id": ex['sample_id'], "improvement": ex['improvement']} 
        for ex in bottom_5_improvements
    ]
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"   ✅ Results saved")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE")
print("="*60)
print(f"Results file: {OUTPUT_FILE}")
print("Next steps based on results:")
if all(success_criteria):
    print("  1. Scale up data generation to 5K+ pairs")
    print("  2. Run multi-turn trajectory analysis")
    print("  3. Consider moving to GRPO for further improvement")
else:
    print("  1. Analyze failure cases above")
    print("  2. Check training logs for convergence")
    print("  3. Consider adjusting hyperparameters or data quality")
print("="*60)
