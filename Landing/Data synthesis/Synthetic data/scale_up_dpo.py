import os
import sys
import json
import time
import pandas as pd

# --- SAFETY NETS ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Missing 'google-genai'. Run: pip install google-genai")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Missing 'datasets'. Run: pip install datasets")
    sys.exit(1)

# Import your custom guards for real-time filtering
try:
    from clinical_guards import ClinicalGuard
except ImportError:
    print("⚠️ 'clinical_guards.py' not found. Running without BERT filters.")
    ClinicalGuard = None

# --- CONFIGURATION ---
DATASET_NAME = "ShenLab/MentalChat16K"
OUTPUT_FILE = "mentalchat_dpo_filtered.jsonl"
LIMIT_SAMPLES = None 
SAFETY_THRESHOLD = 0.8 

# --- API KEY ---
# ⚠️ SECURITY: I removed the hardcoded key. Please ensure it is in your environment.
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("⚠️  WARNING: GOOGLE_API_KEY not found. Please export it.")

client = genai.Client(api_key=api_key)

def call_gemini_with_backoff(system_prompt, user_prompt):
    retries = 5
    base_delay = 4
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.9, 
                    max_output_tokens=400,
                )
            )
            time.sleep(0.5) 
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = base_delay * (2 ** attempt)
                print(f"\n⏳ Quota hit. Sleeping {wait}s...", end="\r")
                time.sleep(wait)
            else:
                print(f"\n❌ API Error: {e}")
                return None
    return None

def main():
    guards = None
    if ClinicalGuard:
        print("🛡️ Initializing Clinical Guards (BERT)...")
        guards = ClinicalGuard()

    print(f"📚 Loading dataset: {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split="train")
        print(f"✅ Loaded {len(ds)} rows.")
        
        # --- CRITICAL DEBUGGING STEP ---
        # This will print the exact column names so we know what we are working with
        first_row = ds[0]
        keys = list(first_row.keys())
        print(f"\n🔍 DEBUG: The dataset has these columns: {keys}")
        print(f"🔍 DEBUG: First row sample: {str(first_row)[:200]}...\n")
        # -------------------------------

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    processed_prompts = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('prompt'):
                        processed_prompts.add(data.get('prompt'))
                except: pass
        print(f"🔄 Resuming... Found {len(processed_prompts)} existing pairs.")

    print("🏭 Starting The Synthetic Factory...")
    
    count = 0
    skipped_toxic = 0
    skipped_parse = 0
    
    iterable_ds = list(ds)
    if LIMIT_SAMPLES:
        iterable_ds = iterable_ds[:LIMIT_SAMPLES]

    for i, row in enumerate(iterable_ds):
        # --- ROBUST CASE-INSENSITIVE MATCHING ---
        # Convert all keys to lowercase to avoid 'Context' vs 'context' issues
        row_lower = {k.lower(): v for k, v in row.items()}
        
        # Try finding the prompt in 'input' OR 'instruction' OR 'context'
        prompt = (row_lower.get('input') or row_lower.get('instruction') or 
                  row_lower.get('prompt') or row_lower.get('query') or 
                  row_lower.get('context') or row_lower.get('history') or 
                  row_lower.get('dialogue'))
        
        # Try finding the chosen response
        chosen = (row_lower.get('output') or row_lower.get('response') or 
                  row_lower.get('chosen') or row_lower.get('target') or 
                  row_lower.get('assistant'))

        if not prompt or not chosen:
            if skipped_parse < 1: # Only show first error
                print(f"⚠️ Row {i} Skipped. Keys found: {list(row.keys())}")
            skipped_parse += 1
            continue
            
        if prompt in processed_prompts:
            continue

        print(f"Processing {i}/{len(iterable_ds)} | Saved: {count} | Toxic Dropped: {skipped_toxic}...", end="\r")
        
        # The average length of responses in this dataset is around 2k words, so we should let gemini generate longer outputs
        # Or it will just learn to write short instead of bad responses.
        system_instruction = """
        You are a well-meaning but UNSKILLED friend acting as a therapist.
        Demonstrate 'Clinical Errors' for a dataset.
        
        Errors to simulate (Choose one):
        1. Toxic Positivity: Invalidating pain by forcing happiness. Rambling about how "everything is great" for a long time.
        2. Premature Advice: Giving a long list of specific, detailed actions they should take immediately without listening.
        3. Dismissal: Explaining at length why their problem isn't a big deal compared to others, citing personal examples.
        4. Self-Centered: Telling a long, detailed personal story that tangentially relates to them.

        CRITICAL LENGTH REQUIREMENT:
        Your response MUST be LONG and VERBOSE (aim for 150-250 words).
        Do NOT be brief.
        - Use filler phrases.
        - Tell a personal anecdote.
        - List 3-4 bullet points of bad advice.
        - Over-explain your point repeatedly.
        
        Goal: Match the character count of a professional therapy response (~1500 chars), but make the content clinical garbage.
        """
        
        user_prompt = f"PATIENT: {prompt}\nBAD RESPONSE:"
        rejected = call_gemini_with_backoff(system_instruction, user_prompt)
        
        if not rejected: continue 

        tox_score = 0.0
        if guards:
            tox_score = guards.score_safety(rejected)
            if tox_score > SAFETY_THRESHOLD:
                skipped_toxic += 1
                continue
            
        entry = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "MentalChat16K",
            "safety_score": tox_score
        }
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
            
        count += 1
        processed_prompts.add(prompt)

    print(f"\n\n✅ Job Complete.")
    print(f"Total Valid Pairs: {count}")
    print(f"Skipped (Missing Columns): {skipped_parse}")
    print(f"Toxic Samples Discarded: {skipped_toxic}")

if __name__ == "__main__":
    main()