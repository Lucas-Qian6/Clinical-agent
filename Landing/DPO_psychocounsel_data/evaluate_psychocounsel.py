import json, sys, time
import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

# ==== CONFIG ====
MAX_NEW_TOKENS = 256
MAX_EVAL_SAMPLES = 100     # set None for full set
DEVICE_ID = 0 if torch.cuda.is_available() else -1

JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "openai")
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.0-flash"

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


# ==== OPTIONAL: your local modules in Drive ====
# If these python files exist in DRIVE_BASE, we can import them.
# If they don't exist, we will run without emotion wrapper / clinical vector.
sys.path.insert(0, str(DRIVE_BASE))

# wrap_model_with_emotion = None
get_clinical_vector = None

# try:
#     from emotion_conditioned_model import wrap_model_with_emotion
#     print("✅ Imported wrap_model_with_emotion")
# except Exception as e:
#     print("⚠️ Could not import emotion_conditioned_model.py (will run without wrapper):", e)

try:
    from senticnet_matrix import get_clinical_vector
    print("✅ Imported get_clinical_vector")
except Exception as e:
    print("⚠️ Could not import senticnet_matrix.py (will not compute clinical vector):", e)


def generate_with_retry(client, model_id: str, prompt: str, system: str, backend: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            if backend == "openai":
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
            if text:
                return text.strip()
        except Exception as e:
            msg = str(e).lower()
            if "429" in str(e) or "rate" in msg:
                time.sleep(60)
            else:
                time.sleep(10)
    return ""


def strip_chat_tokens(prompt: str) -> str:
    return prompt.replace("<|user|>\n", "").replace("\n<|assistant|>\n", "").strip()


def main():
    print("🚀 PsychoCounsel Evaluation: Our Model vs GPT-4o baseline + judge")
    print("=" * 60)

    if not TEST_DATA.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # 1) Load data
    print(f"\n📂 Loading test data from {TEST_DATA}...")
    dataset = load_dataset("json", data_files=str(TEST_DATA), split="train")
    if MAX_EVAL_SAMPLES is not None:
        dataset = dataset.select(range(min(MAX_EVAL_SAMPLES, len(dataset))))
    print(f"   ✅ {len(dataset)} samples")

    # 2) Load our model
    print(f"\n🤖 Loading model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

   # 3) Load emotion injector with trained weights
    injector_path = MODEL_PATH / "emotion_injector.pt"
    if injector_path.exists():

        # Attach injector (creates structure)
        model = attach_emotion_injector(model, emotion_dim=4, layer_idx=8, alpha=0.1)

        # Load the TRAINED weights
        model.injector.load_state_dict(torch.load(injector_path, map_location=model.device))
        print(f"   ✅ Loaded trained emotion injector from {injector_path}")
    else:
        print(f"   ❌ Warning: {injector_path} not found - using random weights!")
        model = attach_emotion_injector(model, emotion_dim=4, layer_idx=8, alpha=0.1)

    # if wrap_model_with_emotion is not None:
    #     model = wrap_model_with_emotion(model, emotion_dim=4, layer_idx=16, alpha=0.1)
    #     print("   ✅ Emotion conditioning wrapper enabled")
    # else:
    #     print("   ⚠️ Emotion wrapper not enabled (missing injector or wrapper module)")

    # 4) Emotion extraction pipeline (only used if dataset emotion missing AND get_clinical_vector exists)
    emo_pipe = None
    if get_clinical_vector is not None:
        print("\n🔌 Loading BERT-GoEmotions pipeline...")
        from transformers import pipeline
        emo_pipe = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-go-emotion",
            return_all_scores=True,
            device=DEVICE_ID,
        )
        print("   ✅ Emotion extractor ready")

    # 5) Init judge client
    print(f"\n⚖️ Initializing judge backend: {JUDGE_BACKEND}")
    if JUDGE_BACKEND == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        import openai
        judge_client = openai.OpenAI(api_key=api_key)
        judge_model = OPENAI_MODEL
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
        from google import genai
        judge_client = genai.Client(api_key=api_key)
        judge_model = GEMINI_MODEL
    print(f"   ✅ Judge model: {judge_model}")

    # 6) Generate our model responses
    print("\n⚡ Generating OUR model responses...")
    our_responses = []
    device = next(model.parameters()).device
    for i in tqdm(range(len(dataset)), desc="Our model"):
        row = dataset[i]
        prompt = row["prompt"]
        question = strip_chat_tokens(prompt)

        # emotion vector: from dataset if present, else compute if possible, else zeros
        emotion = row.get("emotion", None)
        if emotion is None and (get_clinical_vector is not None) and (emo_pipe is not None):
            emotion = get_clinical_vector(question, emo_pipe)
        if emotion is None:
            emotion = [0.0, 0.0, 0.0, 0.0]

        emotion_t = torch.tensor([emotion], dtype=torch.float32, device=device)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # If wrapped injector exists, set emotion
        if hasattr(model, "injector"):
            model.injector.set_emotion(emotion_t)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded.strip()
        our_responses.append(resp)

    # 7) Generate baseline therapist responses using judge model itself (GPT-4o or Gemini)
    print("\n⚡ Generating BASELINE responses (judge model as therapist)...")
    baseline_responses = []
    for i in tqdm(range(len(dataset)), desc="Baseline"):
        row = dataset[i]
        question = strip_chat_tokens(row["prompt"])
        text = generate_with_retry(
            judge_client, judge_model,
            f"Client: {question}\n\nRespond as a professional therapist:",
            RESPONDING_SYSTEM,
            JUDGE_BACKEND,
        )
        baseline_responses.append(text or "(no response)")
        time.sleep(0.5)

    # 8) Judge comparisons
    print("\n⚖️ Judging A (ours) vs B (baseline)...")
    wins_our = wins_baseline = ties = 0
    results = []

    for i in tqdm(range(len(dataset)), desc="Judging"):
        question = strip_chat_tokens(dataset[i]["prompt"])
        our_r = our_responses[i]
        base_r = baseline_responses[i]

        prompt = RANKING_USER_TEMPLATE.format(
            question=question,
            response_a=our_r,
            response_b=base_r,
        )
        verdict = generate_with_retry(judge_client, judge_model, prompt, RANKING_SYSTEM, JUDGE_BACKEND)
        verdict = (verdict or "").strip().upper()

        # robust parse: accept first character if it's A/B
        v = verdict[:1]
        if v == "A":
            wins_our += 1
            result = "our"
        elif v == "B":
            wins_baseline += 1
            result = "baseline"
        else:
            ties += 1
            result = "tie"

        results.append({"idx": i, "verdict": verdict, "result": result})
        time.sleep(0.5)

    total = len(dataset)
    win_rate = (wins_our / total * 100) if total else 0.0

    print("\n" + "=" * 60)
    print("🏆 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Our wins:      {wins_our} ({win_rate:.1f}%)")
    print(f"Baseline wins: {wins_baseline}")
    print(f"Ties:          {ties}")
    print("=" * 60)

    payload = {
        "win_rate": win_rate,
        "wins_our": wins_our,
        "wins_baseline": wins_baseline,
        "ties": ties,
        "total": total,
        "judge_backend": JUDGE_BACKEND,
        "judge_model": judge_model,
        "sample_results_head": results[:20],
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n💾 Saved results to: {OUT_FILE}")


main()