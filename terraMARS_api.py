import os
import torch
import json
import re
import numpy as np
import faiss
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "google/gemma-3-1b-it"
ADAPTER_PATH = "/home/exouser/jyotsna/terra_mars/output/final_adapter"
CHUNKS_FILE  = "/home/exouser/jyotsna/terra_mars/all_chunks.jsonl"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K        = 3

# ── Load chunks for RAG ───────────────────────────────────────────────────────
print("Loading RAG knowledge base...")
chunks = []
chunk_meta = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            text = data.get("text", "")
            if text and len(text) > 20:
                chunks.append(text)
                chunk_meta.append({
                    "title": data.get("title", ""),
                    "url": data.get("url", ""),
                    "domain": data.get("domains", ["general"])[0] if data.get("domains") else "general"
                })
print(f"Loaded {len(chunks)} research chunks from Mars papers")

# ── Build FAISS index ─────────────────────────────────────────────────────────
print("Building FAISS vector index...")
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading TerraMARS fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map="cpu")
model.eval()
print("Model ready!")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TerraMARS API",
    description="Fine-tuned LLM + RAG for Mars terraforming research",
    version="1.1.0"
)

class TextInput(BaseModel):
    text: str

class QuestionInput(BaseModel):
    question: str
    use_rag: bool = True
    top_k: int = 3

# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve_chunks(query, top_k=TOP_K):
    """Retrieve relevant Mars research chunks."""
    q_emb = embedder.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    retrieved = []
    for j, i in enumerate(indices[0]):
        if i < len(chunks):
            retrieved.append({
                "text": chunks[i],
                "title": chunk_meta[i]["title"],
                "url": chunk_meta[i]["url"],
                "domain": chunk_meta[i]["domain"],
                "score": float(scores[0][j])
            })
    return retrieved

# ── Generation ────────────────────────────────────────────────────────────────
def generate(prompt, max_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.0, do_sample=False,
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
        "version": "1.1.0",
        "description": "Fine-tuned LLM with RAG for Mars terraforming research",
        "knowledge_base": f"{len(chunks)} chunks from Mars research papers",
        "endpoints": ["/retrieve", "/ask", "/extract", "/stage", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "Gemma3-1B-QLoRA",
        "rag_chunks": len(chunks),
        "rag_index": index.ntotal
    }

@app.post("/retrieve")
def retrieve_only(input: QuestionInput):
    """
    RAG retrieval endpoint - returns top-k relevant chunks from Mars research papers.
    No LLM generation - pure retrieval.
    """
    retrieved = retrieve_chunks(input.question, top_k=input.top_k)
    return {
        "query": input.question,
        "retrieved_chunks": retrieved,
        "num_chunks": len(retrieved)
    }

@app.post("/ask")
def ask_question(input: QuestionInput):
    """
        
    Answer Mars terraforming questions using fine-tuned LLM with RAG.
    Always retrieves relevant chunks from Mars research papers and 
    grounds the answer in scientific literature.
    Returns the answer plus the source papers used.
    """
   
    if input.use_rag:
        retrieved = retrieve_chunks(input.question, top_k=input.top_k)
        context = "\n\n".join([
            f"Context {i+1} ({r['domain']}): {r['text'][:400]}"
            for i, r in enumerate(retrieved)
        ])
        prompt = f"""Using the following scientific context from Mars research papers:

{context}

Answer this question accurately with specific numbers and units:
{input.question}"""
        answer = generate(prompt)
        return {
            "question": input.question,
            "answer": answer,
            "rag_enabled": True,
            "retrieved_chunks": retrieved,
            "num_sources": len(retrieved)
        }
    else:
        answer = generate(input.question)
        return {
            "question": input.question,
            "answer": answer,
            "rag_enabled": False
        }

@app.post("/extract")
def extract_constraints(input: TextInput):
    """Extract quantitative constraints from Mars science text as JSON."""
    prompt = f"Extract ALL quantitative constraints from this text as JSON:\n{input.text}"
    response = generate(prompt)
    try:
        clean = re.sub(r"```json|```", "", response).strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            constraints = json.loads(json_match.group())
            return {"status": "success", "constraints": constraints, "raw": response}
    except Exception:
        pass
    return {"status": "success", "constraints": None, "raw": response}

@app.post("/stage")
def identify_stage(input: TextInput):
    """Identify Mars terraforming stage from text."""
    prompt = f"Identify the Mars terraforming stage (0-5) and explain:\n{input.text}"
    response = generate(prompt)
    return {"status": "success", "analysis": response}
