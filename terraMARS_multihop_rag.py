"""
TerraMARS Multi-Hop RAG Pipeline
Decomposes complex questions into sub-questions for better retrieval
"""

import os
import json
import torch
import faiss
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_FILE  = "/home/exouser/jyotsna/terra_mars/all_chunks.jsonl"
ADAPTER_PATH = "/home/exouser/jyotsna/terra_mars/output/final_adapter"
BASE_MODEL   = "google/gemma-3-1b-it"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K        = 3
MAX_HOPS     = 2

# ── Load chunks ───────────────────────────────────────────────────────────────
print("Loading chunks...")
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
                    "title":  data.get("title", ""),
                    "domain": data.get("domains", [])[0] if data.get("domains") else "general"
                })

print(f"Loaded {len(chunks)} chunks")

# ── Build index ───────────────────────────────────────────────────────────────
print("Building FAISS index...")
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Index ready!")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map="cpu")
model.eval()
print("Model ready!")

# ── Functions ─────────────────────────────────────────────────────────────────
def retrieve(query, top_k=TOP_K):
    """Retrieve top-k chunks for a query."""
    q_emb = embedder.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return [(chunks[i], chunk_meta[i], scores[0][j]) 
            for j, i in enumerate(indices[0]) if i < len(chunks)]

def generate(prompt, max_tokens=300):
    """Generate answer from model."""
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

def decompose_question(question):
    """Break complex question into sub-questions."""
    prompt = f"""Break this Mars terraforming question into 2 simpler sub-questions that need to be answered sequentially:

Question: {question}

Output ONLY the 2 sub-questions, one per line, starting with "1." and "2." Do not add explanations."""
    
    response = generate(prompt, max_tokens=150)
    
    # Parse sub-questions
    lines = response.strip().split("\n")
    subqs = []
    for line in lines:
        line = line.strip()
        if line.startswith("1.") or line.startswith("2."):
            subqs.append(line[2:].strip())
    
    return subqs[:2] if len(subqs) >= 2 else [question]

def multi_hop_answer(question):
    """Answer complex question using multi-hop retrieval."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    # HOP 1: Decompose question
    print("\n[HOP 1] Decomposing question...")
    sub_questions = decompose_question(question)
    print(f"Sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    
    # HOP 2: Retrieve for each sub-question and collect evidence
    all_evidence = []
    for i, sq in enumerate(sub_questions, 1):
        print(f"\n[HOP 2.{i}] Retrieving for: {sq[:60]}...")
        results = retrieve(sq, top_k=TOP_K)
        for text, meta, score in results:
            all_evidence.append({
                "sub_question": sq,
                "text": text[:400],
                "title": meta["title"],
                "domain": meta["domain"],
                "score": float(score)
            })
            print(f"  - [{meta['domain']}] {meta['title'][:60]}... (score: {score:.2f})")
    
    # HOP 3: Combine evidence and generate final answer
    print("\n[HOP 3] Generating final answer...")
    evidence_text = "\n\n".join([
        f"Evidence {i+1} (relevant to '{ev['sub_question'][:50]}...'):\n{ev['text']}"
        for i, ev in enumerate(all_evidence)
    ])
    
    final_prompt = f"""Using the following scientific evidence from multiple Mars research papers:

{evidence_text}

Answer this complex question comprehensively by combining information from all evidence:

Question: {question}

Provide a clear, specific answer with numbers and units where appropriate."""
    
    final_answer = generate(final_prompt, max_tokens=400)
    
    return {
        "question": question,
        "sub_questions": sub_questions,
        "evidence": all_evidence,
        "answer": final_answer
    }

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "Can cyanobacteria survive on Mars under current atmospheric conditions?",
    "What interventions are needed to reach Stage 3 of Mars terraforming?",
    "Which Mars regions have the highest habitability potential for pioneer organisms?",
]

print(f"\n{'='*60}")
print(f"MULTI-HOP RAG EVALUATION")
print(f"{'='*60}")

results = []
for q in TEST_QUESTIONS:
    result = multi_hop_answer(q)
    results.append(result)
    print(f"\n{'─'*40}")
    print(f"FINAL ANSWER:")
    print(result["answer"])
    print()

# ── Save results ──────────────────────────────────────────────────────────────
with open("/home/exouser/jyotsna/terra_mars/multihop_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("Results saved to multihop_results.json")
print(f"{'='*60}")
