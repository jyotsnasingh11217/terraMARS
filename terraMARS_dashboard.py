import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://149.165.175.237:8000"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TerraMARS",
    page_icon="🔴",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔴 TerraMARS")
st.markdown("**Quantitative Constraint Extraction from Mars Science Literature**")
st.divider()

# ── Check API status ──────────────────────────────────────────────────────────
try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    if r.status_code == 200:
        st.success("✅ TerraMARS API is running")
    else:
        st.error("❌ API not responding")
except:
    st.error("❌ Cannot connect to API — make sure server is running")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Extract Constraints",
    "💬 Ask Question",
    "🪐 Identify Stage"
])

# ── Tab 1: Extract ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Extract Quantitative Constraints")
    st.markdown("Paste any Mars science paper text to extract structured JSON constraints.")

    example_texts = [
        "UV fluence at Mars surface exceeds 50 W/m² under clear sky, attenuation length 1.5 cm in basaltic regolith.",
        "Water activity below 0.60 is considered uninhabitable. Mars surface temperature averages -63°C with atmospheric pressure of 0.6 kPa.",
        "Mars surface dose rate 233 μGy/day measured by RAD instrument. Dust storms reduce UV by 75%.",
    ]

    selected = st.selectbox("Or choose an example:", ["Custom input"] + example_texts)

    if selected == "Custom input":
        text_input = st.text_area("Paste paper text here:", height=150, placeholder="Enter Mars science text...")
    else:
        text_input = st.text_area("Paste paper text here:", value=selected, height=150)

    if st.button("🔍 Extract Constraints", type="primary"):
        if text_input.strip():
            with st.spinner("Extracting constraints..."):
                try:
                    r = requests.post(
                        f"{API_URL}/extract",
                        json={"text": text_input},
                        timeout=300
                    )
                    result = r.json()

                    st.subheader("Results")
                    if result.get("constraints"):
                        st.json(result["constraints"])
                    else:
                        st.text(result.get("raw", "No result"))

                    with st.expander("Raw model output"):
                        st.text(result.get("raw", ""))
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text first!")

# ── Tab 2: Ask ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Ask a Mars Terraforming Question")
    st.markdown("Ask anything about Mars conditions, pioneer organisms, or terraforming constraints.")

    example_questions = [
        "What depth of regolith protects microbes from UV on Mars?",
        "Which pioneer organism is best suited for Mars terraforming Stage 3?",
        "What is the minimum water activity for microbial survival on Mars?",
        "What is the UV flux at Mars surface under clear sky conditions?",
        "How does perchlorate affect microbial survival on Mars?",
    ]

    selected_q = st.selectbox("Or choose an example:", ["Custom question"] + example_questions)

    if selected_q == "Custom question":
        question = st.text_input("Your question:", placeholder="Ask about Mars terraforming...")
    else:
        question = st.text_input("Your question:", value=selected_q)

    if st.button("💬 Ask", type="primary"):
        if question.strip():
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        timeout=300
                    )
                    result = r.json()
                    st.subheader("Answer")
                    st.write(result.get("answer", "No answer"))
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question!")

# ── Tab 3: Stage ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Identify Terraforming Stage")
    st.markdown("Paste text describing Mars conditions to identify the terraforming stage (0-5).")

    st.info("""
    **Terraforming Stages:**
    - Stage 0: Current Mars (P=0.6 kPa, T=-63°C)
    - Stage 1: Pressure Buildup (target: 20-30 kPa)
    - Stage 2: Warming Phase (target: T > -10°C)
    - Stage 3: Pioneer Biology (target: 10⁶ cells/g)
    - Stage 4: Soil Formation (target: 1-3% organic carbon)
    - Stage 5: Breathable Atmosphere (O₂ > 20 kPa)
    """)

    stage_text = st.text_area(
        "Describe Mars conditions:",
        height=150,
        placeholder="e.g. Atmospheric pressure has reached 20 kPa and cyanobacteria colonies are establishing..."
    )

    if st.button("🪐 Identify Stage", type="primary"):
        if stage_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    r = requests.post(
                        f"{API_URL}/stage",
                        json={"text": stage_text},
                        timeout=300
                    )
                    result = r.json()
                    st.subheader("Stage Analysis")
                    st.write(result.get("analysis", "No result"))
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text!")
# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
TerraMARS v1 · Gemma 3 1B + QLoRA · 1,540 training examples · 
JSON extraction accuracy: 83.3%
</div>
""", unsafe_allow_html=True)
