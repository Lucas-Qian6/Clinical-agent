# DPO Training Script for Clinical LLM - Colab Optimized
# Install first: pip install unsloth "xformers<0.0.26" trl
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import json

print("🚀 Clinical DPO Training - Starting...")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect
LOAD_IN_4BIT = True
NUM_EPOCHS = 3  # ⚠️ CHANGED FROM 1 - Small dataset needs more passes
LEARNING_RATE = 5e-6
BETA = 0.1

# Data paths (adjust if needed)
TRAIN_DATA = "Data/dpo_train_dataset.jsonl"  # 526 pairs
OUTPUT_DIR = "clinical_dpo_model_v1"

print(f"📊 Configuration:")
print(f"   Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"   Training Epochs: {NUM_EPOCHS}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   DPO Beta: {BETA}")

# ============================================================================
# 2. LOAD BASE MODEL (Llama 3 8B Instruct)
# ============================================================================
print("\n🤖 Loading Llama 3 8B Instruct...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
print("   ✅ Base model loaded successfully")

# ============================================================================
# 3. ADD LORA ADAPTERS
# ============================================================================
print("\n🔧 Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Optimized for performance
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory efficient
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("   ✅ LoRA adapters configured")

# ============================================================================
# 4. LOAD TRAINING DATA
# ============================================================================
print(f"\n📂 Loading training data from {TRAIN_DATA}...")

def format_dpo(example):
    """
    Ensure data is in correct DPO format.
    Expected format: {"prompt": str, "chosen": str, "rejected": str}
    """
    prompt = example['prompt']
    
    # Verify prompt has correct Llama 3 template
    if not prompt.startswith("<|user|>"):
        prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    return {
        "prompt": prompt,
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

# Load dataset
try:
    dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    dataset = dataset.map(format_dpo)
    print(f"   ✅ Loaded {len(dataset)} training pairs")
    
    # Sanity check
    sample = dataset[0]
    print(f"\n📋 Sample data check:")
    print(f"   Prompt length: {len(sample['prompt'])} chars")
    print(f"   Chosen length: {len(sample['chosen'])} chars")
    print(f"   Rejected length: {len(sample['rejected'])} chars")
    
except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    print("   Please check file path and format")
    raise

# ============================================================================
# 5. INITIALIZE DPO TRAINER
# ============================================================================
print("\n⚙️  Initializing DPO Trainer...")

# Apply Unsloth's memory optimization patch
patch_dpo = PatchDPOTrainer()

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Unsloth handles reference model implicitly
    tokenizer=tokenizer,
    beta=BETA,
    train_dataset=dataset,
    args=DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=NUM_EPOCHS,  # 3 epochs for small dataset
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=100,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        seed=42,
        remove_unused_columns=False,
    ),
)

print("   ✅ Trainer initialized")

# ============================================================================
# 6. TRAIN MODEL
# ============================================================================
print("\n" + "="*60)
print("🎯 STARTING TRAINING")
print("="*60)
print(f"Total examples: {len(dataset)}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Steps per epoch: ~{len(dataset) // (2 * 4)}")
print(f"Total gradient updates: ~{len(dataset) * NUM_EPOCHS // (2 * 4)}")
print("="*60 + "\n")

try:
    trainer_output = dpo_trainer.train()
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Final loss: {trainer_output.training_loss:.4f}")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    raise

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
print(f"\n💾 Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("   ✅ Model saved successfully")

# ============================================================================
# 8. QUICK INFERENCE TEST
# ============================================================================
print("\n🧪 Running quick inference test...")

test_prompt = "I've been feeling really anxious about my upcoming presentation at work."
formatted_prompt = f"<|user|>\n{test_prompt}\n<|assistant|>\n"

FastLanguageModel.for_inference(model)  # Enable inference mode
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

print(f"\n👤 Test Prompt: {test_prompt}")
print("\n🤖 Model Response:")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("<|assistant|>")[-1].strip()
print(response)

print("\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
print(f"Model saved to: {OUTPUT_DIR}")
print("Next step: Run evaluation with Weaver scoring")
print("="*60)
