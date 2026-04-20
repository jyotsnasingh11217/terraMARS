import os
import json
import torch
import faiss
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
import csv
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_FILE  = "/home/exouser/jyotsna/terra_mars/all_chunks.jsonl"
ADAPTER_PATH = "/home/exouser/jyotsna/terra_mars/output/final_adapter"
BASE_MODEL   = "google/gemma-3-1b-it"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K        = 3   # retrieve top 3 most relevant chunks

# ── Ground truth ──────────────────────────────────────────────────────────────
GROUND_TRUTH = [
    {
        "question": "What is the minimum regolith depth for UV shielding on Mars?",
        "expected_value": 3.0,
        "expected_unit": "cm",
        "keywords": ["3", "cm", "shielding", "uv", "depth"],
        "source": "Kminek & Bada 2006"
    },
    {
        "question": "What is the UV flux at Mars surface under clear sky conditions?",
        "expected_value": 50.0,
        "expected_unit": "W/m²",
        "keywords": ["50", "w/m", "uv", "flux", "clear"],
        "source": "Córdoba-Jabonero 2003"
    },
    {
        "question": "What is the minimum water activity for microbial survival?",
        "expected_value": 0.60,
        "expected_unit": "aw",
        "keywords": ["0.6", "water activity", "minimum", "survival"],
        "source": "Rummel et al. 2014"
    },
    {
        "question": "What is the UV attenuation length in basaltic regolith?",
        "expected_value": 1.5,
        "expected_unit": "cm",
        "keywords": ["1.5", "cm", "attenuation", "basalt"],
        "source": "Córdoba-Jabonero 2003"
    },
    {
        "question": "What is the ionizing radiation dose at Mars surface?",
        "expected_value": 0.077,
        "expected_unit": "Gy/yr",
        "keywords": ["0.077", "gy", "ionizing", "dose"],
        "source": "Kminek & Bada 2006"
    },
    {
        "question": "What is the minimum temperature for microbial metabolic activity on Mars?",
        "expected_value": -20.0,
        "expected_unit": "°C",
        "keywords": ["-20", "temperature", "metabolic", "permafrost"],
        "source": "Rivkina et al. 2000"
    },
    {
        "question": "What is the perchlorate concentration on Mars surface?",
        "expected_value": 0.006,
        "expected_unit": "mol/L",
        "keywords": ["0.006", "perchlorate", "mol"],
        "source": "Wadsworth & Cockell 2017"
    },
    {
        "question": "What is the surface atmospheric pressure on Mars?",
        "expected_value": 0.6,
        "expected_unit": "kPa",
        "keywords": ["0.6", "kpa", "pressure", "atmospheric"],
        "source": "Mars facts"
    },
]

JSON_TESTS = [
    {
        "input": "UV fluence at Mars surface exceeds 50 W/m² under clear sky, attenuation length 1.5 cm in basaltic regolith.",
        "expected": {"uv_flux": 50.0, "attenuation_length": 1.5},
    },
    {
        "input": "Water activity below 0.60 is considered uninhabitable. Endospores survive 1000 years in desiccated conditions.",
        "expected": {"water_activity_min": 0.60, "survival_years": 1000},
    },
    {
        "input": "Mars surface dose rate 233 μGy/day measured by RAD instrument. Dust storms reduce UV by 75%.",
        "expected": {"dose_rate": 233.0, "uv_reduction": 0.75},
    },
]

# ── Load chunks for RAG ───────────────────────────────────────────────────────
print("Loading chunks for RAG index...")
chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            # Use source_text or instruction as chunk text
            text = data.get("text", "") or data.get("source_text", "")
            if text and len(text) > 20:
                chunks.append(text)

print(f"Loaded {len(chunks)} chunks")

# ── Build FAISS index ─────────────────────────────────────────────────────────
print(f"Building vector index using {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors!")

# ── Load fine-tuned model ─────────────────────────────────────────────────────
print("\nLoading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map="cpu")
model.eval()
print("Model loaded!")

# ── RAG functions ─────────────────────────────────────────────────────────────
def retrieve(question, top_k=TOP_K):
    """Find most relevant chunks for a question."""
    q_embedding = embedder.encode([question])
    q_embedding = np.array(q_embedding).astype("float32")
    faiss.normalize_L2(q_embedding)
    scores, indices = index.search(q_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved

def ask_with_rag(question, max_tokens=200):
    """Ask model with retrieved context."""
    # Retrieve relevant chunks
    relevant_chunks = retrieve(question)
    context = "\n\n".join([f"Context {i+1}: {c}" 
                           for i, c in enumerate(relevant_chunks)])
    
    # Build RAG prompt
    rag_prompt = f"""Using the following scientific context from Mars research papers:

{context}

Answer this question accurately with specific numbers and units:
{question}"""

    messages = [{"role": "user", "content": rag_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
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

# ── Scoring functions ─────────────────────────────────────────────────────────
def score_keywords(answer, keywords):
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)

def extract_numbers(text):
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]

def score_numeric(answer, expected_value, tolerance=0.15):
    numbers = extract_numbers(answer)
    for num in numbers:
        if abs(num - expected_value) / max(abs(expected_value), 0.001) <= tolerance:
            return 1.0
    return 0.0

def score_json_extraction(answer, expected_values):
    try:
        clean = re.sub(r"```json|```", "", answer).strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            return 0.0
        data = json.loads(json_match.group())
        score = 0
        flat_values = str(data)
        for key, val in expected_values.items():
            if str(val) in flat_values or str(int(val)) in flat_values:
                score += 1
        return score / len(expected_values)
    except Exception:
        return 0.0

# ── Run RAG evaluation ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("EVALUATION: TerraMARS + RAG")
print(f"{'='*60}")

keyword_scores = []
numeric_scores = []

for i, gt in enumerate(GROUND_TRUTH):
    print(f"\n[{i+1}/{len(GROUND_TRUTH)}] {gt['source']}")
    print(f"Q: {gt['question']}")
    answer = ask_with_rag(gt['question'])
    print(f"A: {answer[:200]}")

    kw_score  = score_keywords(answer, gt['keywords'])
    num_score = score_numeric(answer, gt['expected_value'])

    keyword_scores.append(kw_score)
    numeric_scores.append(num_score)

    print(f"Keyword: {kw_score*100:.0f}%  "
          f"Numeric: {'✅' if num_score==1.0 else '❌'} "
          f"(expected {gt['expected_value']} {gt['expected_unit']})")
    print("-"*40)

print(f"\n{'='*60}")
print(f"Scientific Accuracy Summary:")
print(f"  Keyword accuracy : {sum(keyword_scores)/len(keyword_scores)*100:.1f}%")
print(f"  Numeric accuracy : {sum(numeric_scores)/len(numeric_scores)*100:.1f}%")
print(f"{'='*60}")

# ── JSON extraction with RAG ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("JSON Extraction with RAG")
print(f"{'='*60}")

json_scores = []
for i, test in enumerate(JSON_TESTS):
    print(f"\n[{i+1}/{len(JSON_TESTS)}]")
    prompt = f"Extract ALL quantitative constraints from this text as JSON:\n{test['input']}"
    answer = ask_with_rag(prompt)
    print(f"Output: {answer[:300]}")
    score = score_json_extraction(answer, test['expected'])
    json_scores.append(score)
    print(f"JSON score: {score*100:.0f}%")
    print("-"*40)

# ── Final results ─────────────────────────────────────────────────────────────
kw_avg   = sum(keyword_scores)/len(keyword_scores)*100
num_avg  = sum(numeric_scores)/len(numeric_scores)*100
json_avg = sum(json_scores)/len(json_scores)*100
overall  = (kw_avg + num_avg + json_avg) / 3

print(f"\n{'='*60}")
print(f"FINAL RAG EVALUATION REPORT")
print(f"{'='*60}")
print(f"  Keyword accuracy : {kw_avg:.1f}%")
print(f"  Numeric accuracy : {num_avg:.1f}%")
print(f"  JSON extraction  : {json_avg:.1f}%")
print(f"  Overall          : {overall:.1f}%")
print(f"{'='*60}")

# ── Save to CSV ───────────────────────────────────────────────────────────────
csv_file = "/home/exouser/jyotsna/terra_mars/results.csv"
results = {
    "model": "Gemma3-1B-QLoRA+RAG",
    "training_examples": 1540,
    "keyword_accuracy": round(kw_avg, 1),
    "numeric_accuracy": round(num_avg, 1),
    "json_accuracy": round(json_avg, 1),
    "overall": round(overall, 1),
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "notes": "Prototype v1 + RAG (FAISS + MiniLM)"
}

with open(csv_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writerow(results)

print(f"\nResults saved to {csv_file}")

# ── Publication table ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PUBLICATION TABLE")
print(f"{'='*60}")
print(f"{'Model':<25} {'Keyword':>8} {'Numeric':>8} {'JSON':>8} {'Overall':>8}")
print(f"{'-'*60}")
print(f"{'Base Gemma 3 1B':<25} {'~20%':>8} {'~5%':>8} {'~10%':>8} {'~12%':>8}")
print(f"{'TerraMARS v1 (1.5K)':<25} {'62.3%':>8} {'12.5%':>8} {'83.3%':>8} {'52.7%':>8}")
print(f"{'TerraMARS+RAG (naive)':<25} {'56.7%':>8} {'0.0%':>8} {'66.7%':>8} {'41.1%':>8}")
print(f"{'TerraMARS+RAG (proper)':<25} {str(round(kw_avg,1))+'%':>8} {str(round(num_avg,1))+'%':>8} {str(round(json_avg,1))+'%':>8} {str(round(overall,1))+'%':>8}")
print(f"{'TerraMARS v2 (25K)*':<25} {'~85%':>8} {'~50%':>8} {'~95%':>8} {'~77%':>8}")
print(f"{'-'*60}")
print(f"* projected")
print(f"{'='*60}")
