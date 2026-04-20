import os
import torch
import json
import re
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "google/gemma-3-1b-it"
ADAPTER_PATH = "/home/exouser/jyotsna/terra_mars/output/final_adapter"

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading TerraMARS model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map="cpu")
model.eval()
print("Model ready!")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TerraMARS API",
    description="Quantitative constraint extraction from Mars science literature",
    version="1.0.0"
)

# ── Request/Response models ───────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class QuestionInput(BaseModel):
    question: str

# ── Helper function ───────────────────────────────────────────────────────────
def ask(prompt, max_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "TerraMARS API",
        "version": "1.0.0",
        "endpoints": ["/extract", "/ask", "/stage", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "running", "model": "Gemma3-1B-QLoRA"}

@app.post("/extract")
def extract_constraints(input: TextInput):
    """Extract quantitative constraints from Mars science text as JSON."""
    prompt = f"Extract ALL quantitative constraints from this text as JSON:\n{input.text}"
    response = ask(prompt)
    # Try to parse JSON from response
    try:
        clean = re.sub(r"```json|```", "", response).strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            constraints = json.loads(json_match.group())
            return {"status": "success", "constraints": constraints, "raw": response}
    except Exception:
        pass
    return {"status": "success", "constraints": None, "raw": response}

@app.post("/ask")
def ask_question(input: QuestionInput):
    """Ask any Mars terraforming question."""
    response = ask(input.question)
    return {"status": "success", "answer": response}

@app.post("/stage")
def identify_stage(input: TextInput):
    """Identify terraforming stage from text."""
    prompt = f"Identify the Mars terraforming stage (0-5) described in this text and explain why:\n{input.text}"
    response = ask(prompt)
    return {"status": "success", "analysis": response}
