"""
pages/chatbot.py

Streamlit page: AI Chatbot
Interactive RAG chatbot grounded in the CarValue AI dataset.
Maintains conversation history across turns.
"""

import streamlit as st
import pandas as pd
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot_engine import chat
from src.embeddings import build_index, load_index

st.set_page_config(page_title="AI Chatbot · CarValue AI", page_icon="💬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .page-title { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #f7fafc; }
  .mode-badge {
    display: inline-block;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px;
    border-radius: 6px;
    margin-left: 8px;
  }
  .mode-semantic  { background: #1a3a2d; border: 1px solid #68d391; color: #68d391; }
  .mode-aggregate { background: #2d2a00; border: 1px solid #ecc94b; color: #ecc94b; }
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='page-title'>💬 AI Chatbot</p>", unsafe_allow_html=True)
st.caption(
    "Ask anything about the used car inventory. "
    "The chatbot retrieves relevant data from the dataset before answering — "
    "answers are grounded, not hallucinated."
)

# ── load data + index ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading inventory data…")
def get_data_and_index():
    df = pd.read_csv("data/used_car_sales.csv")
    for c in ["Manufactured Year", "Mileage-KM", "Engine Power-HP",
              "Number of Seats", "Number of Doors", "Price-$", "Sold Price-$"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    CURRENT_YEAR = 2026
    df_unsold = df[(df["Sold Price-$"].isna()) | (df["Sold Price-$"] <= 0)].copy()
    req = ["Manufacturer Name", "Car Name", "Car Type", "Color", "Gearbox", "Energy",
           "Location", "Manufactured Year", "Mileage-KM", "Engine Power-HP",
           "Number of Seats", "Number of Doors", "Price-$"]
    df_unsold = df_unsold.dropna(subset=req).copy()
    df_unsold["Car Age"] = (CURRENT_YEAR - df_unsold["Manufactured Year"]).clip(lower=1)
    df_unsold["Mileage per Year"] = df_unsold["Mileage-KM"] / df_unsold["Car Age"]

    FEATURES = ["Manufacturer Name", "Car Name", "Car Type", "Color", "Gearbox", "Energy",
                "Location", "Car Age", "Mileage-KM", "Mileage per Year",
                "Engine Power-HP", "Number of Seats", "Number of Doors"]

    model = joblib.load("model/model.pkl")
    df_unsold["Predicted Price"] = model.predict(df_unsold[FEATURES])
    df_unsold["Gap ($)"]         = df_unsold["Price-$"] - df_unsold["Predicted Price"]
    MAE = 749

    def deal_label(g):
        if g <= 1.5 * MAE: return "Good Deal"
        if g <= 2.0 * MAE: return "Potential Deal"
        return "Risky / Overpaying"

    df_unsold["Deal Label"] = df_unsold["Gap ($)"].apply(deal_label)
    df_unsold["Deal Score"] = ((1 - df_unsold["Gap ($)"] / (2 * MAE)) * 100).clip(0, 100)

    # Try loading FAISS index
    result = load_index()
    if result is not None:
        faiss_index, _, _ = result
    else:
        _, df_unsold, _ = build_index(df_unsold, save=True)
        result = load_index()
        faiss_index = result[0] if result else None

    return df_unsold, faiss_index


df_unsold, faiss_index = get_data_and_index()

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []   # parallel list for retrieved context

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Chat Settings")
    k_results = st.slider("Cars retrieved per query", 3, 15, 6)
    show_context = st.toggle("Show retrieved context", value=False)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.contexts = []
        st.rerun()

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What's the average price of a Toyota SUV?",
        "Find me a good deal on an electric car",
        "How many diesel cars are in Good Deal category?",
        "Compare Mahindra and Maruti average prices",
        "Show me cars similar to a 2022 Hyundai i20",
        "Why might a car be marked as Risky?",
        "Which location has the best deals on average?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state._prefill = ex
            st.rerun()

# ── render chat history ───────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show retrieved context under assistant messages if toggled
        if show_context and msg["role"] == "assistant" and i // 2 < len(st.session_state.contexts):
            ctx_data = st.session_state.contexts[i // 2]
            with st.expander(f"📎 Retrieved context ({ctx_data['mode']} retrieval)"):
                st.code(ctx_data["context"], language="text")

# ── input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", "")
user_input = st.chat_input("Ask about the car inventory…") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                answer = (
                    "⚠️ No API key found. Please set the `ANTHROPIC_API_KEY` "
                    "environment variable in your Streamlit Cloud secrets."
                )
                context_str = ""
                mode = "none"
            else:
                try:
                    answer, context_str, mode = chat(
                        question=user_input,
                        history=st.session_state.messages[:-1],  # exclude current user msg
                        df=df_unsold,
                        index=faiss_index,
                        k=k_results,
                    )
                except Exception as e:
                    answer = f"❌ Error calling the AI: {str(e)}"
                    context_str = ""
                    mode = "error"

        st.markdown(answer)

        if show_context and context_str:
            with st.expander(f"📎 Retrieved context ({mode} retrieval)"):
                st.code(context_str, language="text")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.contexts.append({"context": context_str, "mode": mode})

# ── empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding: 60px 0; color: #718096;'>
      <div style='font-size: 48px; margin-bottom: 16px'>💬</div>
      <div style='font-size: 16px; color: #a0aec0'>Ask me anything about the car inventory</div>
      <div style='font-size: 13px; margin-top: 8px'>
          Answers are grounded in real data retrieved from the dataset — not hallucinated
      </div>
    </div>
    """, unsafe_allow_html=True)
