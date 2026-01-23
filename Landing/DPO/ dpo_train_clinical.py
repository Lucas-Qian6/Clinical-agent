# Install first if needed: pip install unsloth "xformers<0.0.26" trl
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import json

# 1. Configuration
max_seq_length = 2048 # Supports RoPE Scaling internally
dtype = None # None for auto detection
load_in_4bit = True # Use 4bit quantization to reduce memory usage

# 2. Load Base Model (Llama 3 or Mistral)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters (This makes the model trainable)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none", 
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Load YOUR Dataset
# We need to format it slightly for the DPO trainer
def format_dpo(example):
    return {
        "prompt": f"<|user|>\n{example['prompt']}\n<|assistant|>\n",
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

# Load directly from your uploaded file
dataset = load_dataset("json", data_files="final_dpo_training_set.json", split="train")
dataset = dataset.map(format_dpo)

# 5. Initialize DPO Trainer
patch_dpo = PatchDPOTrainer() # Magic patch to fix DPO memory leaks

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None, # Unsloth handles reference model implicitly
    tokenizer = tokenizer,
    beta = 0.1, # The 'strength' of the DPO. 0.1 is standard.
    train_dataset = dataset,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1, # Start with 1 epoch for clinical data
        learning_rate = 5e-6, # Low learning rate to prevent forgetting
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        output_dir = "clinical_rl_model_v1",
        optim = "adamw_8bit",
        seed = 42,
    ),
)

# 6. START TRAINING
print("Starting Clinical DPO Training...")
dpo_trainer.train()

# 7. Save the Model
model.save_pretrained("clinical_rl_model_v1_lora")
tokenizer.save_pretrained("clinical_rl_model_v1_lora")
print("Training Complete. Model saved to 'clinical_rl_model_v1_lora'")