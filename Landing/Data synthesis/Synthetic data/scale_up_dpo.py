import json
import os
import time
import random
from google import genai
from google.genai import types
from tqdm import tqdm
from datasets import load_dataset

# --- CONFIGURATION ---
API_KEY = ""
MODEL_NAME = "gemini-2.0-flash" # Or "gemini-1.5-flash"
OUTPUT_FILE = "gemini_generated_rejects.jsonl"
NUM_SAMPLES = 50

UNSKILLED_THERAPIST_PROMPT = """
You are a well-meaning but UNSKILLED friend acting as a therapist.
Demonstrate 'Clinical Errors' for a dataset.

Errors to simulate:
1. Toxic Positivity.
2. Premature Advice.
3. Dismissal.
4. Self-Centered.

CRITICAL LENGTH REQUIREMENT:
Your response MUST be LONG and VERBOSE (aim for 150-250 words).
Match the character count of a professional therapy response (~1500 chars).
"""

def setup_client():
    if not API_KEY:
        print("❌ Error: GOOGLE_API_KEY environment variable not set.")
        return None
    client = genai.Client(api_key=API_KEY)
    return client

def get_error_instruction():
    errors = [
        "ACT OUT ERROR 1: Toxic Positivity. Be aggressively happy.",
        "ACT OUT ERROR 2: Premature Advice. Give a numbered list of tasks.",
        "ACT OUT ERROR 3: Dismissal. Compare them to starving children.",
        "ACT OUT ERROR 4: Self-Centered. Ignore them and talk about yourself."
    ]
    return random.choice(errors)

def generate_with_retry(client, prompt, retries=3):
    """
    Tries to generate content multiple times with increased timeouts and checks for truncation.
    """
    current_prompt = prompt

    for attempt in range(retries):
        try:
            # Generate using the new client.models syntax
            # IMPORTANT: We cannot set 'timeout' in the new SDK config directly easily,
            # but the client handles standard http timeouts better than the old lib.
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=UNSKILLED_THERAPIST_PROMPT,
                    temperature=0.9,
                    top_p=0.95,
                    max_output_tokens=1024,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ]
                )
            )

            # Check for Empty Response (Blocked?)
            if not response.text:
                print(f"   ⚠️ Empty response. Retrying...")
                time.sleep(2)
                continue

            text = response.text.strip()

            # Check Length (Must be verbose)
            if len(text) < 100:
                print(f"   ⚠️ Response too short ({len(text)} chars). Retrying...")
                current_prompt += "\n\n(IMPORTANT: PLEASE WRITE MORE. YOUR PREVIOUS ANSWER WAS TOO SHORT.)"
                time.sleep(2)
                continue

            return text

        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️ Attempt {attempt+1}/{retries} failed: {error_msg[:100]}...")

            if "429" in error_msg:
                print("      (Rate Limit Hit. Sleeping 60s...)")
                time.sleep(60)
            elif "timed out" in error_msg or "deadline" in error_msg:
                print("      (Timeout. Retrying in 10s...)")
                time.sleep(10)
            else:
                time.sleep(5)

    return None

def main():
    print("🚀 Script Starting (google.genai SDK)...")
    client = setup_client()
    if not client: return

    # 1. Load Data
    print(f"📂 Loading ShenLab/MentalChat16K...")
    try:
        dataset = load_dataset("ShenLab/MentalChat16K", split="train")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 2. Filter Samples
    samples = []
    print("   Filtering dataset...")

    for i, row in enumerate(dataset):
        p_text = row.get('input')
        g_resp = row.get('output')

        if (p_text and g_resp and
            isinstance(p_text, str) and isinstance(g_resp, str) and
            len(p_text) > 5 and len(g_resp) > 20):
            samples.append(row)

        if len(samples) >= NUM_SAMPLES:
            break

    print(f"   ✅ Collected {len(samples)} samples.")

    if not samples:
        print("❌ No valid samples found.")
        return

    generated_data = []

    # 3. Generation Loop
    print(f"⚡ Starting Generation Loop ({MODEL_NAME})...")

    # MANUAL LOOP (No TQDM) for visibility
    for row in tqdm(samples):

        patient_input = row['input']
        gold_response = row['output']
        specific_error = get_error_instruction()
        full_user_prompt = f"PATIENT SAYS: {patient_input}\n\nINSTRUCTION: {specific_error}"

        # Use the Retry Wrapper
        bad_response_text = generate_with_retry(client, full_user_prompt)

        if bad_response_text:
            generated_data.append({
                "prompt": patient_input,
                "chosen": gold_response,
                "rejected": bad_response_text,
                "error_type_simulated": specific_error
            })

        # Rate limit sleep (4s is safer for free tier than 0.5s)
        time.sleep(1)

    # 4. Save
    if generated_data:
        print(f"\n💾 Saving {len(generated_data)} rows to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            for item in generated_data:
                f.write(json.dumps(item) + "\n")
        print("✅ SUCCESS. File saved.")
    else:
        print("\n❌ FAILURE. No data generated.")

if __name__ == "__main__":
    main()