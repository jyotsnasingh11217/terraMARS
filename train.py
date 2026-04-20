import os
import json
import torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ── Login ─────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
login(token=HF_TOKEN)
print("Logged in to HuggingFace!")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "google/gemma-3-1b-it"
DATA_FILE  = "/home/exouser/jyotsna/terra_mars/mars_training_data.jsonl"
OUTPUT_DIR = "/home/exouser/jyotsna/terra_mars/output"
LORA_R     = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj",
                  "gate_proj","up_proj","down_proj"]
MAX_SEQ_LEN = 256
BATCH_SIZE  = 2
GRAD_ACCUM  = 16
LR          = 2e-4
NUM_EPOCHS  = 2

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
records = []
with open(DATA_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"Loaded {len(records)} examples")

def format_example(ex):
    instruction = ex.get("instruction","")[:800]
    output      = ex.get("output","")[:400]
    messages = [
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": output},
    ]
    return {"messages": messages}

formatted  = [format_example(r) for r in records]
dataset    = Dataset.from_list(formatted)
splits     = dataset.train_test_split(test_size=0.05, seed=42)
train_data = splits["train"]
val_data   = splits["test"]
print(f"Train: {len(train_data)}  Val: {len(val_data)}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Gemma 3 1B...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
print("Model loaded!")

# ── Apply QLoRA ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
    ),
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
)

print(f"\nStarting training...")
print(f"Total steps: ~{len(train_data)//(BATCH_SIZE*GRAD_ACCUM)*NUM_EPOCHS}")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(f"{OUTPUT_DIR}/final_adapter", exist_ok=True)
model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print(f"\nDONE! Adapter saved to {OUTPUT_DIR}/final_adapter")
