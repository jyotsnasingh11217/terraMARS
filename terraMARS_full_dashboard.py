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
    "🤖 TerraMARS Mind Chat",
    "📊 Extract Constraints",
    "📈 Monte Carlo Simulation",
    "⚙️ Intervention Analysis",
    "📑 About"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🤖 TerraMARS Mind")
    st.markdown("Chat with our AI assistant backed by Mars science literature.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", 
             "content": "Hello! I am TerraMARS Mind 🔴. Ask me anything about Mars terraforming, pioneer organisms, or habitability conditions!"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🔴" if msg["role"] == "assistant" else None):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask TerraMARS Mind..."):
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
            "content": "Hello! Ask me!"}]
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
# TAB 3: INTERACTIVE Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📈 Interactive Monte Carlo Simulation")
    st.markdown("Adjust parameters and run your own terraforming simulations in real-time.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_sims = st.slider("Number of simulations", 100, 2000, 500, 100)
        start_yr = st.slider("Start year", 2030, 2100, 2050, 10)
    
    with col2:
        end_yr = st.slider("End year", 2200, 2800, 2500, 50)
        time_step = st.slider("Time step (years)", 5, 50, 10, 5)
    
    with col3:
        co2_rate = st.slider("CO₂ release rate (kPa/yr)", 0.1, 1.0, 0.5, 0.1)
        warming_rate = st.slider("Warming rate (°C/yr)", 0.05, 0.5, 0.2, 0.05)

    st.markdown("**Select interventions to include:**")
    icol1, icol2, icol3 = st.columns(3)
    with icol1:
        use_co2     = st.checkbox("CO₂ Release", True)
        use_mirrors = st.checkbox("Orbital Mirrors", True)
    with icol2:
        use_green   = st.checkbox("Greenhouse Gases", True)
        use_cyano   = st.checkbox("Cyanobacteria", True)
    with icol3:
        use_eng_bio = st.checkbox("Engineered Bio", True)
        use_plants  = st.checkbox("Genetic Plants", True)

    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner(f"Running {n_sims} simulations..."):
            import numpy as np
            
            # Build interventions
            interventions = {}
            if use_co2:
                interventions["co2"] = {"start": start_yr, "P_rate": co2_rate, "T_rate": warming_rate*0.5}
            if use_mirrors:
                interventions["mirrors"] = {"start": start_yr+30, "T_rate": warming_rate*0.75}
            if use_green:
                interventions["greenhouse"] = {"start": start_yr+50, "T_rate": warming_rate*0.5}
            if use_cyano:
                interventions["cyano"] = {"start": start_yr+100, "bio_rate": 0.4}
            if use_eng_bio:
                interventions["eng_bio"] = {"start": start_yr+150, "bio_rate": 0.6}
            if use_plants:
                interventions["plants"] = {"start": start_yr+250, "bio_rate": 0.5}
            
            STAGES = {
                0: {"P": 0.6, "T": -63, "bio": 0},
                1: {"P": 25, "T": -40, "bio": 0},
                2: {"P": 50, "T": -10, "bio": 0},
                3: {"P": 80, "T": 0, "bio": 1e6},
                4: {"P": 100, "T": 5, "bio": 1e8},
                5: {"P": 101, "T": 15, "bio": 1e10},
            }
            
            years = list(range(start_yr, end_yr + 1, time_step))
            all_trajectories = []
            
            for sim in range(n_sims):
                np.random.seed(sim)
                P, T, bio = 0.6, -63.0, 0.0
                stages_over_time = []
                P_over_time, T_over_time, bio_over_time = [], [], []
                
                for year in years:
                    for name, params in interventions.items():
                        if year >= params["start"]:
                            if "P_rate" in params:
                                P += params["P_rate"] * time_step * max(0, np.random.normal(1.0, 0.3))
                            if "T_rate" in params:
                                T += params["T_rate"] * time_step * max(0, np.random.normal(1.0, 0.3))
                            if "bio_rate" in params and T > -20:
                                bio = max(bio, 1.0) * (1 + params["bio_rate"] * time_step * np.random.normal(1.0, 0.4))
                    P = min(P, 101); T = min(T, 20); bio = min(bio, 1e11)
                    stage = 0
                    for s in range(5, -1, -1):
                        if P >= STAGES[s]["P"]*0.9 and T >= STAGES[s]["T"]-5 and bio >= STAGES[s]["bio"]*0.5:
                            stage = s
                            break
                    stages_over_time.append(stage)
                    P_over_time.append(P)
                    T_over_time.append(T)
                    bio_over_time.append(bio)
                
                all_trajectories.append({
                    "stages": stages_over_time,
                    "P": P_over_time,
                    "T": T_over_time,
                    "bio": bio_over_time
                })
            
            # Calculate stage probabilities
            n_years = len(years)
            stage_probs = np.zeros((6, n_years))
            for traj in all_trajectories:
                for t, s in enumerate(traj["stages"]):
                    stage_probs[s, t] += 1
            stage_probs = stage_probs / n_sims * 100
            
            # Stage achievement
            st.success("✅ Simulation complete!")
            st.subheader("🎯 Stage Achievement (50% probability year)")
            result_cols = st.columns(5)
            for target in range(1, 6):
                prob_ge = np.zeros(n_years)
                for t in range(n_years):
                    prob_ge[t] = sum(1 for traj in all_trajectories if traj["stages"][t] >= target) / n_sims
                year_50 = None
                for t, yr in enumerate(years):
                    if prob_ge[t] > 0.5:
                        year_50 = yr
                        break
                result_cols[target-1].metric(f"Stage {target}", str(year_50) if year_50 else "Not achieved")
            
            # Plots
            fig, ax = plt.subplots(figsize=(12, 5))
            stage_names = ["Current Mars", "Pressure", "Warming", "Biology", "Soil", "Breathable"]
            for s in range(6):
                ax.plot(years, stage_probs[s], label=f"Stage {s}: {stage_names[s]}", linewidth=2)
            ax.set_xlabel("Year")
            ax.set_ylabel("Probability (%)")
            ax.set_title(f"Stage Probability Over Time ({n_sims} simulations)")
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                P_mean = np.mean([t["P"] for t in all_trajectories], axis=0)
                ax.plot(years, P_mean, color="blue", linewidth=2)
                ax.axhline(y=25, color="orange", linestyle="--", label="Stage 1")
                ax.axhline(y=101, color="green", linestyle="--", label="Earth-like")
                ax.set_xlabel("Year"); ax.set_ylabel("Pressure (kPa)")
                ax.set_title("Atmospheric Pressure")
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                T_mean = np.mean([t["T"] for t in all_trajectories], axis=0)
                ax.plot(years, T_mean, color="red", linewidth=2)
                ax.axhline(y=0, color="green", linestyle="--", label="Freezing")
                ax.set_xlabel("Year"); ax.set_ylabel("Temperature (°C)")
                ax.set_title("Surface Temperature")
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: INTERACTIVE Interventions
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("⚙️ Interactive Intervention Strategy Builder")
    st.markdown("Design your own terraforming strategy and compare with defaults.")
    
    st.markdown("### Build Your Custom Strategy")
    
    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown("**Physical Interventions**")
        custom_co2     = st.slider("CO₂ release start year", 2050, 2200, 2050, 10, key="c1")
        custom_co2_rate = st.slider("CO₂ release rate", 0.0, 1.0, 0.5, 0.1, key="c2")
        custom_mirrors  = st.slider("Orbital mirrors start", 2050, 2200, 2080, 10, key="c3")
        custom_green    = st.slider("Greenhouse gas start", 2050, 2200, 2100, 10, key="c4")
    
    with scol2:
        st.markdown("**Biological Interventions**")
        custom_cyano    = st.slider("Cyanobacteria start", 2100, 2300, 2150, 10, key="c5")
        custom_cyano_rate = st.slider("Cyanobacteria rate", 0.0, 1.0, 0.4, 0.1, key="c6")
        custom_eng_bio  = st.slider("Engineered bio start", 2100, 2300, 2200, 10, key="c7")
        custom_plants   = st.slider("Genetic plants start", 2100, 2400, 2250, 10, key="c8")

    if st.button("⚙️ Run Custom Strategy", type="primary"):
        with st.spinner("Comparing your strategy against defaults..."):
            import numpy as np
            
            STAGES = {
                0: {"P": 0.6, "T": -63, "bio": 0}, 1: {"P": 25, "T": -40, "bio": 0},
                2: {"P": 50, "T": -10, "bio": 0}, 3: {"P": 80, "T": 0, "bio": 1e6},
                4: {"P": 100, "T": 5, "bio": 1e8}, 5: {"P": 101, "T": 15, "bio": 1e10},
            }
            
            scenarios = {
                "Conservative (CO2 only)": {"co2": {"start": 2050, "P_rate": 0.3, "T_rate": 0.15}},
                "Full Terraforming (default)": {
                    "co2": {"start": 2050, "P_rate": 0.5, "T_rate": 0.2},
                    "mirrors": {"start": 2070, "T_rate": 0.2},
                    "green": {"start": 2080, "T_rate": 0.15},
                    "cyano": {"start": 2120, "bio_rate": 0.4},
                    "eng_bio": {"start": 2180, "bio_rate": 0.6},
                    "plants": {"start": 2250, "bio_rate": 0.5},
                },
                "YOUR CUSTOM STRATEGY": {
                    "co2": {"start": custom_co2, "P_rate": custom_co2_rate, "T_rate": custom_co2_rate*0.5},
                    "mirrors": {"start": custom_mirrors, "T_rate": 0.2},
                    "green": {"start": custom_green, "T_rate": 0.15},
                    "cyano": {"start": custom_cyano, "bio_rate": custom_cyano_rate},
                    "eng_bio": {"start": custom_eng_bio, "bio_rate": 0.6},
                    "plants": {"start": custom_plants, "bio_rate": 0.5},
                },
            }
            
            years = list(range(2050, 2501, 10))
            results = {}
            
            for sc_name, interventions in scenarios.items():
                all_traj = []
                for sim in range(300):
                    np.random.seed(sim)
                    P, T, bio = 0.6, -63.0, 0.0
                    stages_ot = []
                    for year in years:
                        for name, params in interventions.items():
                            if year >= params["start"]:
                                if "P_rate" in params:
                                    P += params["P_rate"] * 10 * max(0, np.random.normal(1.0, 0.3))
                                if "T_rate" in params:
                                    T += params["T_rate"] * 10 * max(0, np.random.normal(1.0, 0.3))
                                if "bio_rate" in params and T > -20:
                                    bio = max(bio, 1.0) * (1 + params["bio_rate"] * 10 * np.random.normal(1.0, 0.4))
                        P = min(P, 101); T = min(T, 20); bio = min(bio, 1e11)
                        stage = 0
                        for s in range(5, -1, -1):
                            if P >= STAGES[s]["P"]*0.9 and T >= STAGES[s]["T"]-5 and bio >= STAGES[s]["bio"]*0.5:
                                stage = s
                                break
                        stages_ot.append(stage)
                    all_traj.append(stages_ot)
                results[sc_name] = all_traj
            
            # Calculate stage years
            st.subheader("🏆 Strategy Comparison")
            table_data = []
            for sc_name, traj_list in results.items():
                row = {"Strategy": sc_name}
                for target in range(1, 6):
                    year_50 = None
                    for t, yr in enumerate(years):
                        count = sum(1 for t_list in traj_list if t_list[t] >= target)
                        if count / 300 > 0.5:
                            year_50 = yr
                            break
                    row[f"Stage {target}"] = str(year_50) if year_50 else "Not achieved"
                table_data.append(row)
            st.dataframe(table_data, use_container_width=True)
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["#1f77b4", "#ff7f0e", "#d62728"]
            for i, (sc_name, traj_list) in enumerate(results.items()):
                prob_5 = []
                for t in range(len(years)):
                    count = sum(1 for t_list in traj_list if t_list[t] >= 5)
                    prob_5.append(count / 300 * 100)
                ax.plot(years, prob_5, label=sc_name, linewidth=2, color=colors[i])
            ax.set_xlabel("Year")
            ax.set_ylabel("P(Stage 5) %")
            ax.set_title("Probability of Reaching Breathable Atmosphere")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

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
    5. **Monte Carlo** — Real-time user-driven trajectory simulations
    6. **Intervention Engine** —  Custom strategy builder with live comparison
    
    ### Technical Stack
    
    - **Framework:** HuggingFace Transformers + TRL + PEFT
    - **Hardware:** NVIDIA A100 on Jetstream2 HPC
    - **API:** FastAPI + Uvicorn
    - **Frontend:** Streamlit
    - **Numerical:** NumPy + Matplotlib

    ### Contributors
    - **Jyotsna Singh**
    - **Ash Black**
    - **Jeff Larson**
    """)

   
