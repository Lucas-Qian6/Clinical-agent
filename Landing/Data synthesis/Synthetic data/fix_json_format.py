import json
import os
import sys

# The file to fix
TARGET_FILE = "final_dpo_training_set.json"

def fix_json_file(file_path):
    """
    Reads a file containing concatenated JSON objects (e.g. {obj}{obj})
    and converts it into a standard JSON list (e.g. [{obj}, {obj}]).
    """
    # COLAB-SPECIFIC: Check if file exists, if not, prompt upload
    # if not os.path.exists(file_path):
    #     print(f"⚠️ File '{file_path}' not found in current directory.")
    #     try:
    #         import google.colab
    #         print("   Detected Google Colab. Please upload your file now:")
    #         from google.colab import files
    #         uploaded = files.upload()
    #         if uploaded:
    #             # Use the uploaded filename (in case it differs)
    #             file_path = list(uploaded.keys())[0]
    #             print(f"   Received: {file_path}")
    #         else:
    #             print("   No file uploaded. Exiting.")
    #             return
    #     except ImportError:
    #         print("   (Not in Colab, and file missing. Please check path.)")
    #         return

    print(f"📂 Reading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Check if it's already a list
    if content.startswith('[') and content.endswith(']'):
        print("✅ File is already a valid JSON list. No changes needed.")
        try:
            json.loads(content)
            return
        except json.JSONDecodeError:
            print("   (Wait, it looks like a list but failed validation. Attempting to fix...)")

    # Parse concatenated objects
    data = []
    decoder = json.JSONDecoder()
    pos = 0
    success_count = 0
    
    print("   Parsing content...")
    
    while pos < len(content):
        # Skip leading whitespace/newlines
        while pos < len(content) and content[pos].isspace():
            pos += 1
        
        if pos >= len(content):
            break
            
        try:
            # raw_decode reads one valid JSON object and tells us where it ends
            obj, next_pos = decoder.raw_decode(content, pos)
            data.append(obj)
            pos = next_pos
            success_count += 1
            
            if success_count % 1000 == 0:
                print(f"   ...parsed {success_count} objects")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Parsing stopped due to error at char {pos}: {e}")
            break

    print(f"✅ Successfully extracted {len(data)} objects.")

    # Write back as a standard JSON list
    print(f"💾 Overwriting {file_path} with standard JSON format...")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print("🎉 Success! Your file is now a valid JSON list.")
        
    except Exception as e:
        print(f"❌ Error writing file: {e}")

if __name__ == "__main__":
    # COLAB-SAFE ARG PARSING
    # Filter out Jupyter kernel args (like -f /root/.../kernel.json)
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-') and 'jupyter' not in arg]
    
    if args:
        TARGET_FILE = args[0]
        
    fix_json_file(TARGET_FILE)