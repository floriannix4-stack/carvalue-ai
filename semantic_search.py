"""
pages/semantic_search.py

Streamlit page: Semantic Car Search
Lets users describe a car in plain English and returns the most similar
listings from the inventory using sentence-transformer embeddings + FAISS.
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import build_index, load_index, semantic_search

st.set_page_config(page_title="Semantic Search · CarValue AI", page_icon="🔍", layout="wide")

# ── shared CSS (same dark theme as main app) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .page-title { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #f7fafc; }
  .sim-badge {
    display: inline-block;
    background: #1a3a2d;
    border: 1px solid #68d391;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 12px;
    color: #68d391;
    font-family: 'Space Mono', monospace;
  }
  .car-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .car-title { font-weight: 600; font-size: 15px; color: #e2e8f0; }
  .car-detail { font-size: 13px; color: #a0aec0; margin-top: 4px; }
  .verdict-good     { color: #68d391; font-weight: 600; }
  .verdict-potential{ color: #ecc94b; font-weight: 600; }
  .verdict-risky    { color: #fc8181; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='page-title'>🔍 Semantic Car Search</p>", unsafe_allow_html=True)
st.caption(
    "Describe the car you're looking for in plain English. "
    "The AI uses sentence embeddings to find the most semantically similar listings — "
    "no need to match exact filter values."
)

# ── load data + index ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building embedding index…")
def get_index():
    """Try to load saved index; rebuild from scratch if not found."""
    result = load_index()
    if result is not None:
        return result
    # Rebuild — load df the same way as app.py does
    import pandas as pd, joblib, numpy as np

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

    return build_index(df_unsold, save=True)


index, df_unsold, embeddings = get_index()

# ── search UI ─────────────────────────────────────────────────────────────────
col_q, col_k = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Describe what you're looking for",
        placeholder="e.g. cheap electric car with low mileage, or family SUV automatic diesel",
        label_visibility="collapsed",
    )
with col_k:
    k = st.selectbox("Results", [5, 10, 15, 20], index=1, label_visibility="collapsed")

example_queries = [
    "affordable hatchback with low mileage",
    "powerful SUV automatic diesel",
    "cheap electric car good range",
    "recent Toyota or Hyundai sedan",
    "old truck low price Texas",
]
st.caption("Try: " + "  ·  ".join(f"`{q}`" for q in example_queries))

if query.strip():
    with st.spinner("Searching…"):
        results = semantic_search(query.strip(), index, df_unsold, k=k)

    st.markdown(f"**{len(results)} results** for *\"{query}\"*")
    st.markdown("---")

    for i, row in results.iterrows():
        sim_pct = row["Similarity"] * 100
        verdict = row["Deal Label"]
        if "Good" in verdict:
            v_cls = "verdict-good"
        elif "Potential" in verdict:
            v_cls = "verdict-potential"
        else:
            v_cls = "verdict-risky"

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"""
            <div class='car-card'>
              <div class='car-title'>
                #{i+1} &nbsp; {int(row['Manufactured Year'])} {row['Manufacturer Name']} {row['Car Name']}
                &nbsp;<span class='sim-badge'>similarity {sim_pct:.0f}%</span>
              </div>
              <div class='car-detail'>
                {row['Car Type']} · {row['Energy']} · {row['Gearbox']} · {row['Color']} ·
                {int(row['Mileage-KM']):,} km · {int(row['Engine Power-HP'])} HP ·
                {int(row.get('Number of Seats', 0))} seats · 📍 {row['Location']}
              </div>
              <div class='car-detail' style='margin-top:8px'>
                <strong style='color:#e2e8f0'>Asking: ${int(row['Price-$']):,}</strong> &nbsp;|&nbsp;
                AI value: ${int(row['Predicted Price']):,} &nbsp;|&nbsp;
                Gap: <strong>${row['Gap ($)']:+,.0f}</strong> &nbsp;|&nbsp;
                Score: <strong>{row['Deal Score']:.0f}/100</strong> &nbsp;|&nbsp;
                Verdict: <span class='{v_cls}'>{verdict}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.metric("Deal Score", f"{row['Deal Score']:.0f}/100")
            st.metric("Asking", f"${int(row['Price-$']):,}")

    # Offer to open chatbot with these results
    st.markdown("---")
    st.info("💬 Want to ask questions about these results? Head to the **AI Chatbot** page in the sidebar.")

else:
    st.markdown("""
    <div style='text-align:center; padding: 60px 0; color: #718096;'>
        <div style='font-size: 48px; margin-bottom: 12px'>🔍</div>
        <div style='font-size: 16px'>Type a description above to search the inventory</div>
        <div style='font-size: 13px; margin-top: 8px'>
            Uses sentence-transformer embeddings — no exact keyword matching required
        </div>
    </div>
    """, unsafe_allow_html=True)
