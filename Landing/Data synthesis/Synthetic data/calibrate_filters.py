import json
import numpy as np
import pandas as pd
from clinical_guards import ClinicalGuard
import re

# Config
INPUT_FILE = "gemini-3-flash-preview.json" 
OUTPUT_REPORT = "data_calibration_report.txt"

def load_json_stream(filepath):
    """
    Robustly loads a file that contains multiple JSON objects,
    whether they are one-per-line (JSONL) or pretty-printed and concatenated.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File {filepath} not found.")
        return []
    
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # Skip leading whitespace/newlines
        while pos < len(content) and content[pos].isspace():
            pos += 1
        
        if pos >= len(content):
            break
            
        try:
            # raw_decode parses one valid JSON object and returns the end position
            obj, next_pos = decoder.raw_decode(content, pos)
            data.append(obj)
            pos = next_pos
        except json.JSONDecodeError:
            # If we hit garbage or incomplete JSON at the end, stop
            print(f"⚠️ Warning: Could not decode JSON starting at position {pos}. Stopping load.")
            break
            
    return data

def main():
    print("🚀 Starting AI Triage Calibration...")
    
    # 1. Load the Guards
    # Since you know CUDA, this will auto-detect your GPU.
    try:
        guards = ClinicalGuard()
    except Exception as e:
        print(f"❌ Error initializing ClinicalGuard: {e}")
        print("Ensure you have 'clinical_guards.py' in the same folder and libraries installed.")
        return
    
    # 2. Load the Dataset
    print(f"📂 Loading {INPUT_FILE}...")
    data = load_json_stream(INPUT_FILE)

    if not data:
        print("❌ No data loaded. Please check if the input file is empty or malformed.")
        return

    print(f"📊 Analyzing {len(data)} DPO pairs...")
    
    # 3. Analyze
    results = []
    
    for i, entry in enumerate(data):
        # Use .get() for safety
        prompt = entry.get('prompt', '')
        chosen = entry.get('chosen', '')
        rejected = entry.get('rejected', '')
        
        # Calculate scores
        # We expect Chosen to be balanced (not extreme toxic positivity)
        # We expect Rejected to be either high toxic positivity or dismissive
        
        sent_chosen = guards.score_sentiment(chosen)
        sent_rejected = guards.score_sentiment(rejected)
        
        safety_rejected = guards.score_safety(rejected)
        
        results.append({
            "id": i,
            "sent_chosen": sent_chosen,
            "sent_rejected": sent_rejected,
            "sent_diff": sent_chosen - sent_rejected,
            "safety_rejected": safety_rejected
        })
        
        if i % 10 == 0:
            print(f"Scanned {i}/{len(data)}...", end="\r")

    # 4. Compute Statistics (The "Baseline")
    df = pd.DataFrame(results)
    
    print("\n\n=== 🏥 CALIBRATION REPORT ===")
    print(f"Total Samples: {len(df)}")
    
    print("\n--- Sentiment Analysis (0=Neg, 1=Pos) ---")
    print(f"Avg Chosen Sentiment:   {df['sent_chosen'].mean():.3f} (Ideally ~0.4-0.6 for neutral/clinical)")
    print(f"Avg Rejected Sentiment: {df['sent_rejected'].mean():.3f} (Toxic Positivity often scores >0.8)")
    
    print("\n--- Safety/Toxicity ---")
    print(f"Max Toxicity in Rejected: {df['safety_rejected'].max():.3f}")
    
    # 5. Define Thresholds for Step 3
    # We use the 5th and 95th percentiles to determine our "Safe Range"
    lower_bound = df['sent_chosen'].quantile(0.05)
    upper_bound = df['sent_chosen'].quantile(0.95)
    
    print("\n--- ✅ RECOMMENDED FILTERS FOR 16K DATASET ---")
    print(f"When processing MentalChat16K, discard data if:")
    print(f"1. Chosen Response Sentiment is < {lower_bound:.2f} or > {upper_bound:.2f} (Too extreme)")
    print(f"2. Rejected Response Toxicity is > 0.8 (Too dangerous to use even as negative example)")
    
    # Save the dataframe for inspection
    df.to_csv("triage_results.csv", index=False)
    print("\nDetailed results saved to 'triage_results.csv'")

if __name__ == "__main__":
    main()