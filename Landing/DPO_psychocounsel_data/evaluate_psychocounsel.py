"""
Evaluate emotion-conditioned PsychoCounsel model vs GPT-4o.

Uses PsychoCounsel protocol: model acts as therapist, GPT-4o judges pairs.
Metric: Win rate (% of test cases where our model is preferred).

Requires:
- prepare_psychocounsel_data.py run first (test split with emotion)
- Trained model from dpo_train_psychocounsel.py
- OPENAI_API_KEY for GPT-4o as judge (or GOOGLE_API_KEY for Gemini fallback)
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

# Add parent for emotion modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotion_conditioned_model import EmotionInjector, EmotionConditionedWrapper, wrap_model_with_emotion
from senticnet_matrix import get_clinical_vector

# --- CONFIGURATION ---
TEST_DATA = "Data/psychocounsel_dpo_test.jsonl"
MODEL_PATH = "psychocounsel_emotion_model_v1"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4
MAX_EVAL_SAMPLES = 100  # Cap for quick eval; set to None for full test set
DEVICE = 0 if torch.cuda.is_available() else -1

# Judge: "openai" for GPT-4o, "google" for Gemini
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "openai")
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.0-flash"

# PsychoCounsel-style prompts (inspired by paper evaluation protocol)
RESPONDING_SYSTEM = """You are a professional psychotherapist. Respond to the client with empathy, 
support, and evidence-based therapeutic techniques. Be warm, non-judgmental, and focused on the client's wellbeing."""

RANKING_SYSTEM = """You are an expert evaluator comparing therapist responses. Given a client message and two therapist responses, 
determine which response is better in terms of: empathy, relevance, clarity, safety, exploration depth, and support for client autonomy.
Respond with ONLY "A" or "B" - no explanation."""

RANKING_USER_TEMPLATE = """Client message:
{question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Answer A or B:"""


def generate_with_retry(client, model_id: str, prompt: str, system: str, backend: str, max_retries: int = 3):
    """Generate with retries for rate limits and timeouts."""
    for attempt in range(max_retries):
        try:
            if backend == "openai":
                import openai
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=512,
                    temperature=0.3,
                )
                text = resp.choices[0].message.content or ""
            else:
                from google import genai
                from google.genai import types
                resp = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=512,
                        temperature=0.3,
                    ),
                )
                text = (resp.text or "").strip()
            if text and len(text) > 0:
                return text.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(60)
            else:
                time.sleep(10)
        attempt += 1
    return ""


def main():
    print("🚀 PsychoCounsel Evaluation: Model vs GPT-4o")
    print("=" * 60)

    script_dir = Path(__file__).resolve().parent
    test_path = script_dir / TEST_DATA
    model_path = script_dir / MODEL_PATH

    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        print("   Run prepare_psychocounsel_data.py first.")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train with dpo_train_psychocounsel.py first.")
        sys.exit(1)

    # 1. Load test data
    print(f"\n📂 Loading test data from {test_path}...")
    dataset = load_dataset("json", data_files=str(test_path), split="train")
    if MAX_EVAL_SAMPLES:
        dataset = dataset.select(range(min(MAX_EVAL_SAMPLES, len(dataset))))
    print(f"   ✅ {len(dataset)} test samples")

    # 2. Load model + emotion conditioning
    print(f"\n🤖 Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Wrap with emotion injection if injector exists
    injector_path = model_path / "emotion_injector.pt"
    if injector_path.exists():
        model = wrap_model_with_emotion(model, emotion_dim=4, layer_idx=16, alpha=0.1)
        print("   ✅ Emotion conditioning loaded")
    else:
        print("   ⚠️ No emotion injector; running without emotion conditioning")

    # 3. Initialize emotion pipeline for patient text
    print("\n🔌 Loading BERT-GoEmotions for emotion extraction...")
    from transformers import pipeline
    emo_pipe = pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        return_all_scores=True,
        device=DEVICE,
    )

    # 4. Initialize judge (GPT-4o or Gemini)
    print(f"\n⚖️ Initializing judge ({JUDGE_BACKEND})...")
    if JUDGE_BACKEND == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("   ❌ OPENAI_API_KEY not set")
            sys.exit(1)
        import openai
        judge_client = openai.OpenAI(api_key=api_key)
        judge_model = OPENAI_MODEL
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("   ❌ GOOGLE_API_KEY or GEMINI_API_KEY not set")
            sys.exit(1)
        from google import genai
        judge_client = genai.Client(api_key=api_key)
        judge_model = GEMINI_MODEL
    print(f"   ✅ Judge: {judge_model}")

    # 5. Generate our model's responses
    print("\n⚡ Generating our model's responses...")
    our_responses = []
    for i in tqdm(range(len(dataset)), desc="Our model"):
        row = dataset[i]
        prompt = row["prompt"]
        question = row["prompt"].replace("<|user|>\n", "").replace("\n<|assistant|>\n", "").strip()
        emotion = row.get("emotion")
        if emotion is None:
            emotion = get_clinical_vector(question, emo_pipe)
        emotion_t = torch.tensor([emotion], dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

        device = next(model.parameters()).device if hasattr(model, "parameters") else "cuda"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if hasattr(model, "injector"):
            model.injector.set_emotion(emotion_t.to(device))
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded
        our_responses.append(resp)

    # 6. Generate GPT-4o baseline responses (for comparison)
    print("\n⚡ Generating judge model's responses (as therapist baseline)...")
    baseline_responses = []
    for i in tqdm(range(len(dataset)), desc="Baseline"):
        row = dataset[i]
        question = row["prompt"].replace("<|user|>\n", "").replace("\n<|assistant|>\n", "").strip()
        text = generate_with_retry(
            judge_client, judge_model,
            f"Client: {question}\n\nRespond as a professional therapist:",
            RESPONDING_SYSTEM,
            JUDGE_BACKEND,
        )
        baseline_responses.append(text or "(no response)")
        time.sleep(0.5)

    # 7. Judge: compare our model vs baseline
    print("\n⚖️ Running GPT-4o as judge...")
    wins_our = 0
    wins_baseline = 0
    ties = 0
    results = []

    for i in tqdm(range(len(dataset)), desc="Judging"):
        row = dataset[i]
        question = row["prompt"].replace("<|user|>\n", "").replace("\n<|assistant|>\n", "").strip()
        our_r = our_responses[i]
        base_r = baseline_responses[i]

        prompt = RANKING_USER_TEMPLATE.format(
            question=question,
            response_a=our_r,
            response_b=base_r,
        )
        verdict = generate_with_retry(
            judge_client, judge_model,
            prompt, RANKING_SYSTEM, JUDGE_BACKEND,
        )
        verdict = (verdict or "").strip().upper()
        if "A" in verdict and "B" not in verdict[:2]:
            wins_our += 1
            result = "our"
        elif "B" in verdict and "A" not in verdict[:2]:
            wins_baseline += 1
            result = "baseline"
        else:
            ties += 1
            result = "tie"
        results.append({"idx": i, "verdict": verdict, "result": result})
        time.sleep(0.5)

    # 8. Report
    total = len(dataset)
    win_rate = wins_our / total * 100 if total > 0 else 0
    print("\n" + "=" * 60)
    print("🏆 EVALUATION RESULTS")
    print("=" * 60)
    print(f"   Our model wins:   {wins_our} ({win_rate:.1f}%)")
    print(f"   Baseline wins:    {wins_baseline}")
    print(f"   Ties:             {ties}")
    print(f"   Win rate (ours):  {win_rate:.1f}%")
    print("=" * 60)

    out_file = script_dir / "psychocounsel_eval_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "win_rate": win_rate,
            "wins_our": wins_our,
            "wins_baseline": wins_baseline,
            "ties": ties,
            "total": total,
            "results": results[:20],
        }, f, indent=2)
    print(f"\n💾 Saved to {out_file}")


if __name__ == "__main__":
    main()
