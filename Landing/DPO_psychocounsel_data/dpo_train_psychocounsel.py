"""
DPO Training for PsychoCounsel-Preference with emotion conditioning.

Uses TraitBasis-style activation injection: projects 4D emotion to hidden_dim
and adds to hidden states at layer 16. Requires pre-processed data from
prepare_psychocounsel_data.py.
"""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from trl import DPOConfig

# --- COLAB + DRIVE CONFIG ---
# Set to your Drive folder when running in Colab (e.g. "/content/drive/MyDrive/Clinical Agent")
# Leave None for local runs
DRIVE_BASE = "/content/drive/MyDrive/Clinical Agent"

def _setup_colab():
    """Mount Drive and add to path when running in Colab."""
    if DRIVE_BASE is None:
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir))
        return script_dir
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"⚠️ Could not mount Drive: {e}")
        print("   Run: from google.colab import drive; drive.mount('/content/drive')")
    sys.path.insert(0, DRIVE_BASE)
    # Use Drive for HF datasets cache so preprocessing (extract prompt, chat template, tokenize)
    # is reused across runs instead of re-running ~90s every time
    cache_dir = Path(DRIVE_BASE) / ".cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    return Path(DRIVE_BASE)

SCRIPT_DIR = _setup_colab()


# --- CONFIGURATION ---
# Reduced for Colab T4 (16GB): DPO creates policy + reference model, both need GPU
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
BETA = 0.1
EMOTION_LAYER_IDX = 8
EMOTION_ALPHA = 0.1
# MAX_TRAIN_SAMPLES = 500

# Relative to SCRIPT_DIR (SCRIPT_DIR = DRIVE_BASE in Colab, else script folder)
TRAIN_DATA = "psychocounsel_dpo_train.jsonl"
OUTPUT_DIR = "psychocounsel_emotion_model_v1"
MODEL_NAME = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
# If still OOM on Colab free tier, try: "unsloth/llama-3.2-3b-instruct-bnb-4bit"


def main():
    print("🚀 PsychoCounsel DPO Training with Emotion Conditioning")
    print("=" * 60)

    script_dir = SCRIPT_DIR
    train_path = script_dir / TRAIN_DATA
    if not train_path.exists():
        print(f"❌ Train data not found: {train_path}")
        print("   Run prepare_psychocounsel_data.py first.")
        sys.exit(1)

    # 1. Load model
    print("\n🤖 Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("   ✅ Model loaded")

    # 2. Add LoRA
    print("\n🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("   ✅ LoRA configured")

    # 3. Attach emotion injector (no wrapper - keeps model as PEFT for trainer)
    print("\n💭 Attaching emotion injection (layer 16)...")
    model = attach_emotion_injector(
        model,
        emotion_dim=4,
        layer_idx=EMOTION_LAYER_IDX,
        alpha=EMOTION_ALPHA,
    )
    print("   ✅ Emotion conditioning active")

    # 4. Load dataset
    print(f"\n📂 Loading data from {train_path}...")
    dataset = load_dataset("json", data_files=str(train_path), split="train")

    def ensure_emotion(example):
        out = {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
        if "emotion" not in example:
            out["emotion"] = [0.0, 0.0, 0.0, 0.0]
        else:
            out["emotion"] = example["emotion"]
        return out
    # dataset = dataset.select(range(min(500, len(dataset))))
    dataset = dataset.map(ensure_emotion)
    print(f"   ✅ Loaded {len(dataset)} pairs")

    # 5. Custom collator: TRL format (prompt/chosen/rejected) + emotion
    # Handles both raw text (chosen/rejected) and pre-tokenized (chosen_input_ids)
    # since DPOTrainer may preprocess the dataset before the collator runs
    def collate_with_emotion(features):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Emotion: use from features or default to zeros
        emotions = [
            f.get("emotion", [0.0, 0.0, 0.0, 0.0]) for f in features
        ]

        sample = features[0]
        if "chosen_input_ids" in sample:
            # Already tokenized by DPOTrainer preprocessing - pad if variable length
            # DPOTrainer may omit attention_mask; create from input_ids if missing
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            def _pad_batch(key_ids, key_mask):
                items = []
                for f in features:
                    ids = f[key_ids]
                    mask = f.get(key_mask)
                    if mask is None:
                        # Create attention_mask: 1 where not pad, 0 where pad
                        if isinstance(ids, torch.Tensor):
                            mask = (ids != pad_id).long()
                        else:
                            mask = [1 if t != pad_id else 0 for t in ids]
                    items.append({"input_ids": ids, "attention_mask": mask})
                collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")
                out = collator(items)
                return out["input_ids"], out["attention_mask"]

            p_ids, p_mask = _pad_batch("prompt_input_ids", "prompt_attention_mask")
            c_ids, c_mask = _pad_batch("chosen_input_ids", "chosen_attention_mask")
            r_ids, r_mask = _pad_batch("rejected_input_ids", "rejected_attention_mask")
            batch = {
                "prompt_input_ids": p_ids,
                "prompt_attention_mask": p_mask,
                "chosen_input_ids": c_ids,
                "chosen_attention_mask": c_mask,
                "rejected_input_ids": r_ids,
                "rejected_attention_mask": r_mask,
                "emotion": torch.tensor(emotions, dtype=torch.float32),
            }
        elif "chosen" in sample or "chosen_response" in sample:
            # Raw text: tokenize ourselves
            chosen_key = "chosen" if "chosen" in sample else "chosen_response"
            rejected_key = "rejected" if "rejected" in sample else "rejected_response"
            prompts = [f["prompt"] for f in features]
            chosens = [f[chosen_key] for f in features]
            rejecteds = [f[rejected_key] for f in features]

            def tok(texts):
                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )

            prompt_enc = tok(prompts)
            chosen_enc = tok(chosens)
            rejected_enc = tok(rejecteds)

            batch = {
                "prompt_input_ids": prompt_enc["input_ids"],
                "prompt_attention_mask": prompt_enc["attention_mask"],
                "chosen_input_ids": chosen_enc["input_ids"],
                "chosen_attention_mask": chosen_enc["attention_mask"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"],
                "emotion": torch.tensor(emotions, dtype=torch.float32),
            }
        else:
            raise KeyError(
                f"Dataset has unexpected format. Expected 'chosen'/'rejected' or "
                f"'chosen_input_ids'/'rejected_input_ids'. Got keys: {list(sample.keys())}"
            )
        return batch

    # 6. DPO Trainer (EmotionDPOTrainer passes emotion to model)
    # NOTE: ref_model=None triggers creation of a reference model copy when the trainer
    # is initialized. That doubles GPU usage. Use smaller seq length and batch size.
    print("\n⚙️  Initializing DPO Trainer...")
    torch.cuda.empty_cache()
    PatchDPOTrainer()

    print("\n🔄 Creating reference model (with injector for symmetry)...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    ref_model = FastLanguageModel.get_peft_model(
        ref_model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    # Attach injector to ref model too (but don't train it)
    ref_model = attach_emotion_injector(ref_model, emotion_dim=4, layer_idx=EMOTION_LAYER_IDX, alpha=EMOTION_ALPHA)
    for param in ref_model.injector.parameters():
        param.requires_grad = False  # Freeze ref model injector

    print("   ✅ Reference model has injector (frozen)")

    dpo_trainer = EmotionDPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=collate_with_emotion,
        args=DPOConfig(
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=512,
            beta=BETA,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            # precompute_ref_log_probs=True,
            warmup_ratio=0.1,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_steps=500,
            output_dir=str(script_dir / OUTPUT_DIR),
            optim="adamw_8bit",
            seed=42,
            remove_unused_columns=False,
            gradient_checkpointing=True,         # ⭐ ADD THIS
            max_grad_norm=0.3,   
        ),
    )

    print("   ✅ Trainer ready (emotion will be passed to model)")



    # ========== ADD TO YOUR TRAINING SCRIPT ==========
    # After creating the trainer and before dpo_trainer.train():

    print("\n⚠️  Running pre-training verification...")
    verification_passed = verify_training_setup(
        model=model,
        ref_model=dpo_trainer.ref_model,  # or your explicit ref_model
        tokenizer=tokenizer,
        dpo_trainer=dpo_trainer,
        dataset=dataset,
    )

    if not verification_passed:
        print("\n🛑 Stopping - fix issues before training")
        sys.exit(1)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # ⭐ Set environment variable for memory fragmentation
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


    # 7. Train
    print("\n" + "=" * 60)
    print("🎯 STARTING TRAINING")
    print("=" * 60)
    try:
        dpo_trainer.train()
        #resume_from_checkpoint=True
        print("\n✅ Training complete")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

    # 8. Save (model is PEFT with injector attached)
    out_dir = script_dir / OUTPUT_DIR
    print(f"\n💾 Saving to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Save emotion injector state (projection layer for inference)
    if hasattr(model, "injector"):
        injector_path = Path(out_dir) / "emotion_injector.pt"
        torch.save(model.injector.state_dict(), injector_path)
        print(f"   Saved emotion injector to {injector_path}")
    print("   ✅ Done")


if __name__ == "__main__":
    main()
