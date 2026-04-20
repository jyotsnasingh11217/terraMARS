import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU only

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "google/gemma-3-1b-it"
ADAPTER_PATH = "/home/exouser/jyotsna/terra_mars/output/final_adapter"

print("Loading model on CPU...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    device_map="cpu",
)
model.eval()
print("Model loaded!")

def ask(question):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response

questions = [
    "What depth of Mars regolith provides sufficient UV shielding for microbial survival?",
    "Which pioneer organism is best suited for Mars terraforming Stage 3?",
    "Extract constraints from: UV fluence at Mars surface exceeds 50 W/m² under clear sky, attenuation length 1.5 cm in basaltic regolith.",
]

print("\n" + "="*60)
print("TERRA-MARS Fine-tuned Model Test")
print("="*60)

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-"*40)
