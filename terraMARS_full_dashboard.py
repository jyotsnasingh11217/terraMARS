"""
TerraMARS Phase 6 — Full Interactive Dashboard
Combines LLM, RAG, Monte Carlo, and Intervention Analysis
"""

import streamlit as st
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://149.165.175.237:8000"
MONTECARLO_FILE = "/home/exouser/jyotsna/terra_mars/montecarlo_results.json"
INTERVENTION_FILE = "/home/exouser/jyotsna/terra_mars/intervention_results.json"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TerraMARS Dashboard",
    page_icon="🔴",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🔴")
with col2:
    st.title("TerraMARS")
    st.markdown("**Domain-Specific AI for Mars Terraforming Research**")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔴 TerraMARS")
    st.markdown("*Version 1.0 — Prototype*")
    st.divider()
    
    st.markdown("**Pipeline Components:**")
    st.markdown("- ✅ LLM Fine-tuning (Gemma 3 1B)")
    st.markdown("- ✅ RAG (FAISS + MiniLM)")
    st.markdown("- ✅ Monte Carlo (1,000 sims)")
    st.markdown("- ✅ Intervention Engine")
    st.divider()
    
    st.markdown("**Evaluation Scores:**")
    st.metric("JSON Extraction", "83.3%")
    st.metric("Keyword Accuracy", "62.3%")
    st.metric("Overall", "52.7%")
    st.divider()
    
    st.caption("Built on Jetstream2 HPC")
    st.caption("Gemma 3 1B + QLoRA + RAG")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Mars Mind Chat",
    "📊 Extract Constraints",
    "📈 Monte Carlo Simulation",
    "⚙️ Intervention Analysis",
    "📑 About"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🤖 Mars Mind")
    st.markdown("Chat with our AI assistant backed by Mars science literature.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", 
             "content": "Hello! I am Mars Mind 🔴. Ask me anything about Mars terraforming, pioneer organisms, or habitability conditions!"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🔴" if msg["role"] == "assistant" else None):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask Mars Mind anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🔴"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(f"{API_URL}/ask",
                                      json={"question": prompt}, timeout=300)
                    answer = r.json().get("answer", "No answer")
                    st.write(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [{"role": "assistant",
            "content": "Hello! Ask me anything!"}]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Extract
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Extract Scientific Constraints")
    st.markdown("Paste any Mars science paper text to extract structured JSON constraints.")

    examples = [
        "Custom input",
        "UV fluence at Mars surface exceeds 50 W/m² under clear sky, attenuation length 1.5 cm in basaltic regolith.",
        "Water activity below 0.60 is uninhabitable. Mars surface temperature averages -63°C with atmospheric pressure of 0.6 kPa.",
        "Mars surface dose rate 233 μGy/day measured by RAD instrument. Dust storms reduce UV by 75%.",
    ]

    selected = st.selectbox("Choose example or enter custom:", examples)
    default = "" if selected == "Custom input" else selected
    text = st.text_area("Paper text:", value=default, height=150)

    if st.button("🔍 Extract", type="primary"):
        if text.strip():
            with st.spinner("Extracting..."):
                try:
                    r = requests.post(f"{API_URL}/extract",
                                      json={"text": text}, timeout=300)
                    result = r.json()
                    st.success("✅ Extraction complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Structured JSON:**")
                        if result.get("constraints"):
                            st.json(result["constraints"])
                        else:
                            st.info("No structured JSON found")
                    with col2:
                        st.markdown("**Raw Model Output:**")
                        st.text(result.get("raw", "")[:500])
                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📈 Monte Carlo Stage Transition Simulation")
    st.markdown("Probabilistic prediction of Mars terraforming stages over time.")

    if os.path.exists(MONTECARLO_FILE):
        with open(MONTECARLO_FILE) as f:
            mc_data = json.load(f)

        st.info(f"Based on {mc_data['n_simulations']} Monte Carlo trajectories from {mc_data['start_year']} to {mc_data['end_year']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Simulations", mc_data['n_simulations'])
        col2.metric("Time Range", f"{mc_data['start_year']}-{mc_data['end_year']}")
        col3.metric("Time Step", f"{mc_data['time_step']} years")

        years = mc_data["years"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        stage_names = ["Stage 0: Current Mars", "Stage 1: Pressure", 
                       "Stage 2: Warming", "Stage 3: Biology",
                       "Stage 4: Soil", "Stage 5: Breathable"]
        for s in range(6):
            probs = [p*100 for p in mc_data["stage_probabilities"][f"stage_{s}"]]
            ax.plot(years, probs, label=stage_names[s], linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Probability (%)")
        ax.set_title("Terraforming Stage Probability Over Time")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(years, mc_data["mean_trajectory"]["P"], color="blue", linewidth=2)
            ax.axhline(y=25, color="orange", linestyle="--", label="Stage 1")
            ax.axhline(y=101, color="green", linestyle="--", label="Earth-like")
            ax.set_xlabel("Year")
            ax.set_ylabel("Pressure (kPa)")
            ax.set_title("Atmospheric Pressure")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(years, mc_data["mean_trajectory"]["T"], color="red", linewidth=2)
            ax.axhline(y=0, color="green", linestyle="--", label="Freezing")
            ax.set_xlabel("Year")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Surface Temperature")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Run terraMARS_montecarlo.py first to generate simulation data")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Interventions
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("⚙️ Intervention Strategy Comparison")
    st.markdown("Compare different terraforming strategies.")

    if os.path.exists(INTERVENTION_FILE):
        with open(INTERVENTION_FILE) as f:
            intv_data = json.load(f)

        st.success(f"🏆 **Optimal Strategy:** {intv_data['best_scenario']}")
        st.metric("Stage 5 reached by year", intv_data['best_stage5_year'])

        st.markdown("### Strategy Comparison Table")
        scenarios = intv_data["scenarios"]
        table_data = []
        for name, timings in scenarios.items():
            table_data.append({
                "Strategy": name,
                "Stage 1": timings["stage_1_year"] or "Not achieved",
                "Stage 2": timings["stage_2_year"] or "Not achieved",
                "Stage 3": timings["stage_3_year"] or "Not achieved",
                "Stage 4": timings["stage_4_year"] or "Not achieved",
                "Stage 5": timings["stage_5_year"] or "Not achieved",
            })
        st.dataframe(table_data, use_container_width=True)

        st.markdown("### Scientific Findings")
        st.info("""
        **Key Insights from 2,500 simulations:**
        
        1. Physical interventions alone cannot reach Stage 5
        2. Biology alone is too slow without warm atmosphere
        3. ONLY combined physical + biological approach reaches Stage 5
        4. Earliest breathable Mars: ~180 years from project start
        5. All 6 intervention types required for complete terraforming
        """)
    else:
        st.warning("⚠️ Run terraMARS_intervention.py first to generate comparison data")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: About
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📑 About TerraMARS")
    
    st.markdown("""
    **TerraMARS** is a domain-specific AI pipeline for Mars terraforming research.
    
    ### System Architecture
    
    1. **Data Collection** — 700 papers from arXiv, Semantic Scholar, PubMed Central
    2. **Synthetic Data Generation** — 1,540 training examples via Llama 3.2 + Ollama
    3. **Fine-tuning** — Gemma 3 1B with QLoRA (r=16, α=32)
    4. **RAG** — FAISS vector search with MiniLM embeddings
    5. **Monte Carlo** — 1,000 trajectory simulations for stage prediction
    6. **Intervention Engine** — Strategy comparison across 5 scenarios
    
    ### Technical Stack
    
    - **Framework:** HuggingFace Transformers + TRL + PEFT
    - **Hardware:** NVIDIA A100 on Jetstream2 HPC
    - **API:** FastAPI + Uvicorn
    - **Frontend:** Streamlit
    - **Numerical:** NumPy + Matplotlib
    """)

