import pandas as pd
import json
import re
import time
import os
import sys

# --- CONFIGURATION ---
# Make sure this matches your actual file name
INPUT_FILE = "Final Dialogues.csv"
OUTPUT_FILE = "clinical_dpo_pairs.jsonl"
MIN_EVAL_SCORE = 20

# --- IMPORT SAFETY NET ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ MISSING LIBRARY. Please run: pip install google-genai")
    sys.exit(1)

# Initialize Client
api_key = 
if not api_key:
    print("⚠️  WARNING: GOOGLE_API_KEY not found. The script might fail.")
    # api_key = "Paste_Key_Here_If_Needed"

client = genai.Client(api_key=api_key)

def call_llm_api(system_prompt, user_prompt):
    """
    Real call to Google Gemini API.
    """
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview', # Flash is 10x faster than Pro/Preview
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.9,
                max_output_tokens=400,
            )
        )
        # Sleep is still needed to be polite to the API, but Flash allows higher rates.
        time.sleep(1.0)
        return response.text

    except Exception as e:
        print(f"\n⚠️ API Error: {e}")
        return None

def parse_conversation(conv_text):
    pattern = r'\[(Kyle|Jamie)\]'
    parts = re.split(pattern, conv_text)
    dialogue = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i]
        content = parts[i+1].strip()
        role = "therapist" if speaker == "Kyle" else "patient"
        dialogue.append({"role": role, "content": content})
        i += 2
    return dialogue

def generate_negative_response(conversation_history, current_patient_input):
    history_text = ""
    for turn in conversation_history[-3:]:
        prefix = "Therapist: " if turn['role'] == 'therapist' else "Patient: "
        history_text += f"{prefix}{turn['content']}\n"

    system_instruction = """
    You are a well-meaning but UNSKILLED friend acting as a therapist.
    Demonstrate 'Clinical Errors' (Toxic Positivity, Advice Giving, Dismissal).
    Keep it short (1-2 sentences).
    """

    user_prompt = f"""
    HISTORY:
    {history_text}
    PATIENT: "{current_patient_input}"
    BAD RESPONSE:
    """
    return call_llm_api(system_instruction, user_prompt)

def load_existing_progress(filepath):
    """
    Reads the output file to find which prompts are already done.
    Returns a Set of prompts to skip.
    """
    processed = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data['prompt'])
                except:
                    continue
    return processed

def main():
    print(f"Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File {INPUT_FILE} not found.")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # --- 1. RESUME CAPABILITY ---
    print("Checking existing progress...")
    processed_prompts = load_existing_progress(OUTPUT_FILE)
    print(f"Found {len(processed_prompts)} existing pairs. Resuming...")

    dpo_dataset = []
    consecutive_errors = 0
    total_processed_session = 0

    # Calculate total workload for progress bar
    # (Approximate, as we don't know exactly how many turns each row has)
    print(f"Scanning {len(df)} conversations...")

    for index, row in df.iterrows():
        # Quality Filter
        score = row.get('evaluation score', 0)
        try:
            score = int(score)
        except:
            score = 0
        if score < MIN_EVAL_SCORE:
            continue

        try:
            dialogue = parse_conversation(row['conversations'])
        except Exception:
            continue

        for i in range(len(dialogue)):
            turn = dialogue[i]

            if turn['role'] == 'patient':
                if i + 1 < len(dialogue) and dialogue[i+1]['role'] == 'therapist':

                    patient_input = turn['content']
                    chosen_response = dialogue[i+1]['content']

                    # --- 2. SKIP IF DONE ---
                    if patient_input in processed_prompts:
                        continue

                    # --- 3. PROGRESS VISUALIZATION ---
                    print(f"Processing Row {index} | Total Generated: {len(processed_prompts) + total_processed_session}...", end="\r")

                    # Generate
                    history = dialogue[:i]
                    rejected_response = generate_negative_response(history, patient_input)

                    if rejected_response:
                        consecutive_errors = 0 # Reset error counter
                        entry = {
                            "prompt": patient_input,
                            "history": [h['content'] for h in history], # Optional, depending on model template
                            "chosen": chosen_response,
                            "rejected": rejected_response,
                            "source": "Final Dialogues",
                            "original_score": score
                        }

                        # --- 4. IMMEDIATE SAVE ---
                        # Write line-by-line so we never lose data
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(entry) + "\n")

                        processed_prompts.add(patient_input)
                        total_processed_session += 1
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= 5:
                            print("\n\n❌ TOO MANY API ERRORS. Stopping script to save quota.")
                            print("Check your API key or Quota limits.")
                            return

    print(f"\n\n✅ Job Complete!")
    print(f"Total pairs in {OUTPUT_FILE}: {len(processed_prompts)}")

if __name__ == "__main__":
    main()