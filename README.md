# 🔴 TerraMARS

**Fine-Tuning for Quantitative Constraint Extraction in Mars Science Literature and the Impact of Retrieval-Augmented Generation**

## Overview

TerraMARS is an end-to-end AI pipeline for Mars terraforming research combining:

- 🤖 **Fine-tuned LLM** for Mars science constraint extraction
- 🔍 **RAG pipeline** for grounded scientific retrieval
- 📈 **Monte Carlo simulation** for terraforming trajectory prediction
- ⚙️ **Intervention engine** for strategy comparison
- 🌐 **REST API** for integration with other systems
- 📊 **Interactive dashboard** for exploration

## Results

| Model | Keyword | Numeric | JSON | Overall |
|-------|---------|---------|------|---------|
| Base Gemma 3 1B | 59.8% | 25.0% | 16.7% | 33.8% |
| TerraMARS v1 (1.5K) | 62.3% | 12.5% | **83.3%** | **52.7%** |
| TerraMARS + RAG (naive) | 56.7% | 0.0% | 66.7% | 41.1% |
| TerraMARS + RAG (proper) | 59.2% | 12.5% | 83.3% | 51.7% |
| TerraMARS + MultiHop RAG | 58.5% | 0.0% | 66.7% | 41.7% |

**Key finding:** Fine-tuning improves JSON extraction 5x over base model (16.7% → 83.3%)

## Quick Start

### Requirements
- Python 3.12+
- HuggingFace account (for Gemma 3 access)
- Ollama (for synthetic data generation)
- NVIDIA GPU recommended (CPU works, slower)

### Installation

```bash
# Create virtual environment
python3 -m venv terra_mars_env
source terra_mars_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Pipeline

### 1. Data Collection
```bash
python terraMARS_collect.py
```
Scrapes open-access Mars papers from arXiv, Semantic Scholar, PMC.

### 2. Data Processing
```bash
python terraMARS_process.py
```
Cleans, tags, chunks papers into 11 domain categories.

### 3. Training Data Generation
```bash
python terraMARS_generate.py
```
Generates instruction-output pairs using Llama 3.2 via Ollama.

### 4. Fine-tuning
```bash
python train.py
```
QLoRA fine-tuning (r=16, α=32) of Gemma 3 1B.

### 5. Evaluation
```bash
python evaluate.py               # Fine-tuned model
python evaluate_base.py           # Base model baseline
python rag_evaluate_v2.py         # With RAG
python evaluate_multihop.py       # With Multi-hop RAG
```

### 6. Monte Carlo Simulation
```bash
python terraMARS_montecarlo.py
```
Simulates 1,000 terraforming trajectories.

### 7. Intervention Analysis
```bash
python terraMARS_intervention.py
```
Compares 5 terraforming strategies.

## API Usage

### Start the server
```bash
uvicorn terraMARS_api:app --host 0.0.0.0 --port 8000
```

### Live endpoint
http://149.165.175.237:8000/docs
### Example calls

```python
import requests

# Extract constraints
r = requests.post(
    "http://149.165.175.237:8000/extract",
    json={"text": "UV flux at Mars exceeds 50 W/m² under clear sky"}
)
print(r.json())

# Ask a question
r = requests.post(
    "http://149.165.175.237:8000/ask",
    json={"question": "What depth protects microbes from UV on Mars?"}
)
print(r.json())
```

### Endpoints
- `GET /` — API info
- `GET /health` — Service status
- `POST /extract` — Extract JSON constraints from text
- `POST /ask` — Ask Mars science questions
- `POST /stage` — Identify terraforming stage

## Dashboard

```bash
streamlit run terraMARS_full_dashboard.py --server.port 8502
```

Live: http://149.165.175.237:8502/

## Terraforming Stage Ontology

| Stage | Name | Pressure | Temp | Biology |
|-------|------|----------|------|---------|
| 0 | Current Mars | 0.6 kPa | -63°C | 0 |
| 1 | Pressure Buildup | 25 kPa | -40°C | 0 |
| 2 | Warming | 50 kPa | -10°C | 0 |
| 3 | Pioneer Biology | 80 kPa | 0°C | 10⁶ cells/g |
| 4 | Soil Formation | 100 kPa | 5°C | 10⁸ cells/g |
| 5 | Breathable | 101 kPa | 15°C | 10¹⁰ cells/g |

Based on Stork et al. (2025), McKay et al. (1991), Zubrin & McKay (1997).

## Technical Stack

- **ML:** Transformers, PEFT (QLoRA), TRL, bitsandbytes
- **RAG:** FAISS, sentence-transformers
- **API:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Data:** NumPy, Matplotlib, SciPy
- **Hardware:** NVIDIA A100 (Jetstream2 HPC)

## Contributors

- Jyotsna Singh
- Ash Black
- Jeff Larsen

## License

MIT License — see LICENSE file.

## Acknowledgements

- Jetstream2 HPC (NSF ACCESS)
- HuggingFace open model ecosystem
- arXiv, Semantic Scholar, PubMed Central for open-access papers
- Anthropic Claude
- OpenAI
