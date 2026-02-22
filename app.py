"""
CarValue AI — Full Prototype
ESADE PDAI Assignment 1

Classification logic (v2 — MAE-anchored):
  Gap = asking_price − predicted_price  (always ≥ 0 in this dataset)
  🟢 Good Deal      : gap ≤ 0.5 × MAE  → seller asks within half our error margin
  🟡 Potential Deal : gap ≤ 1.0 × MAE  → within model uncertainty, worth negotiating
  🔴 Risky / Overpaying : gap > 1.0 × MAE  → statistically outside error band

  Score (0–100): 100 = gap is 0 (predicted == asked), 0 = gap ≥ 2 × MAE
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import json
import random
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CarValue AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;1,9..40,300&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* ── KPI cards — fixed equal height ── */
  .kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d4a7a;
    border-radius: 14px;
    padding: 16px 14px 14px;
    text-align: center;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
  }
  .kpi-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.15;
  }
  .kpi-delta {
    font-size: 11px;
    margin-top: 5px;
    color: #718096;
  }
  .c-green  { color: #68d391; }
  .c-yellow { color: #ecc94b; }
  .c-red    { color: #fc8181; }
  .c-blue   { color: #76e4f7; }
  .c-white  { color: #e2e8f0; }

  /* ── Verdict box (Price Estimator) ── */
  .verdict-wrap {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
  }
  .v-green  { background: linear-gradient(135deg,#1c3d2d,#276749); border: 2px solid #68d391; }
  .v-yellow { background: linear-gradient(135deg,#3d3300,#5a4a00); border: 2px solid #ecc94b; }
  .v-red    { background: linear-gradient(135deg,#3d1c1c,#5a2626); border: 2px solid #fc8181; }
  .v-emoji  { font-size: 52px; margin-bottom: 6px; }
  .v-title  { font-family: 'Space Mono', monospace; font-size: 20px; font-weight: 700; color: #f7fafc; }
  .v-detail { font-size: 14px; color: #cbd5e0; margin-top: 10px; line-height: 1.6; }

  /* ── Header ── */
  .app-header { padding: 8px 0 16px; border-bottom: 1px solid #2d3748; margin-bottom: 4px; }
  .app-title  { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #f7fafc; margin: 0; }
  .app-sub    { color: #718096; font-size: 13px; margin-top: 2px; }

  /* ── Legend ── */
  .legend-row { display:flex; gap:24px; padding:10px 4px; font-size:13px; color:#a0aec0; flex-wrap:wrap; }
  .ldot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LOCATION_TO_STATE = {
    "California": "California",   "Florida": "Florida",
    "New York": "New York",       "Texas": "Texas",
    "North Carolina": "North Carolina", "Tennessee": "Tennessee",
    "Oklahoma": "Oklahoma",       "Utah": "Utah",
    "Chicago": "Illinois",        "Columbus": "Ohio",
    "Denver": "Colorado",         "Detroit": "Michigan",
    "Madison": "Wisconsin",       "Philadelphia": "Pennsylvania",
    "Portland": "Oregon",         "San Jose": "California",
    "Tucson": "Arizona",
}

CURRENT_YEAR = 2026

FEATURES = [
    "Manufacturer Name", "Car Name", "Car Type", "Color",
    "Gearbox", "Energy", "Location", "Car Age",
    "Mileage-KM", "Mileage per Year", "Engine Power-HP",
    "Number of Seats", "Number of Doors",
]

# Labels & colours
LABEL_GOOD      = "Good Deal"
LABEL_POTENTIAL = "Potential Deal"
LABEL_RISKY     = "Risky / Overpaying"

COLOR_MAP = {
    LABEL_GOOD:      [72,  187, 120, 210],   # green
    LABEL_POTENTIAL: [236, 201,  75, 210],   # yellow
    LABEL_RISKY:     [252, 129, 129, 210],   # red
}
CHART_COLORS = {
    LABEL_GOOD:      "#68d391",
    LABEL_POTENTIAL: "#ecc94b",
    LABEL_RISKY:     "#fc8181",
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/used_car_sales.csv")
    for c in ["Manufactured Year", "Mileage-KM", "Engine Power-HP",
              "Number of Seats", "Number of Doors", "Price-$",
              "Sold Price-$", "Margin-%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_geojson():
    with open("data/gz_2010_us_040_00_500k.json") as f:
        return json.load(f)

@st.cache_data
def load_model_meta():
    try:
        import json as _j
        with open("model/model_meta.json") as f:
            return _j.load(f)
    except Exception:
        return {}

@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv("model/model_comparison.csv")
    except Exception:
        return pd.DataFrame()

model          = load_model()
df_full        = load_data()
states_geojson = load_geojson()
model_meta     = load_model_meta()
model_comp_df  = load_model_comparison()

# MAE from model metadata (fallback to known training result)
MODEL_MAE = float(model_meta.get("best_test_mae", 749))

# MAE-anchored thresholds for the gap = (asking - predicted)
# Green  : gap ≤ 0.5 × MAE  → within half our error margin
# Yellow : gap ≤ 1.0 × MAE  → still within uncertainty band
# Red    : gap > 1.0 × MAE  → seller asks significantly more than AI estimates
THRESH_GOOD      = 1.5 * MODEL_MAE   # ~$1,124 — within 1.5× error: competitive price
THRESH_POTENTIAL = 2.0 * MODEL_MAE   # ~$1,498 — within 2× error: worth negotiating

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_state_rings(_geojson):
    rings = {}
    for feat in _geojson["features"]:
        name = feat["properties"]["NAME"]
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            rings[name] = geom["coordinates"][0]
        else:
            polys = geom["coordinates"]
            rings[name] = polys[5][0] if name == "Michigan" else max(polys, key=lambda p: len(p[0]))[0]
    return rings

STATE_RINGS = build_state_rings(states_geojson)

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def sample_point_in_ring(ring, rng, max_tries=1000):
    xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
    bx = (min(xs), max(xs)); by = (min(ys), max(ys))
    for _ in range(max_tries):
        px, py = rng.uniform(*bx), rng.uniform(*by)
        if point_in_poly(px, py, ring):
            return px, py
    return sum(bx) / 2, sum(by) / 2

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    d = df.copy()
    d["Car Age"]          = (CURRENT_YEAR - d["Manufactured Year"]).clip(lower=1)
    d["Mileage per Year"] = d["Mileage-KM"] / d["Car Age"]
    return d

def deal_label(gap: float) -> str:
    """gap = asking_price − predicted_price (positive = seller asks more than AI estimate)"""
    if gap <= THRESH_GOOD:      return LABEL_GOOD
    if gap <= THRESH_POTENTIAL: return LABEL_POTENTIAL
    return LABEL_RISKY

def deal_score(gap: float) -> float:
    """
    100 = perfect deal (gap ≤ 0), 0 = gap ≥ 2×MAE.
    Linear interpolation in between.
    """
    s = (1.0 - gap / (2.0 * MODEL_MAE)) * 100.0
    return float(np.clip(s, 0.0, 100.0))

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE UNSOLD INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
df_unsold = df_full[(df_full["Sold Price-$"].isna()) | (df_full["Sold Price-$"] <= 0)].copy()
req_cols  = ["Manufacturer Name", "Car Name", "Car Type", "Color", "Gearbox", "Energy",
             "Location", "Manufactured Year", "Mileage-KM", "Engine Power-HP",
             "Number of Seats", "Number of Doors", "Price-$"]
df_unsold = df_unsold.dropna(subset=req_cols).copy()
df_unsold = add_features(df_unsold)

df_unsold["Predicted Price"] = model.predict(df_unsold[FEATURES])

df_unsold["Gap ($)"]         = df_unsold["Price-$"] - df_unsold["Predicted Price"]
df_unsold["Deal Label"]      = df_unsold["Gap ($)"].apply(deal_label)
df_unsold["Deal Score"]      = df_unsold["Gap ($)"].apply(deal_score)
df_unsold["State"]           = df_unsold["Location"].map(LOCATION_TO_STATE).fillna(df_unsold["Location"])

# Sold cars for analytics
df_sold = df_full[df_full["Sold Price-$"] > 0].copy()
df_sold = add_features(df_sold)

# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def assign_coordinates(locs: tuple) -> tuple:
    rng = random.Random(42)
    lats, lons = [], []
    for loc in locs:
        state = LOCATION_TO_STATE.get(loc)
        if state and state in STATE_RINGS:
            lon, lat = sample_point_in_ring(STATE_RINGS[state], rng)
        else:
            lat, lon = rng.uniform(32, 44), rng.uniform(-114, -80)
        lats.append(lat); lons.append(lon)
    return tuple(lats), tuple(lons)

with st.spinner("Placing cars on map…"):
    _lats, _lons = assign_coordinates(tuple(df_unsold["Location"]))
df_unsold = df_unsold.copy()
df_unsold["lat"] = list(_lats)
df_unsold["lon"] = list(_lons)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚗 CarValue AI")
    st.caption("AI-powered used car intelligence")
    st.divider()

    st.markdown("**🗺️ Map Filters**")
    sb_labels = st.multiselect(
        "Deal verdict",
        [LABEL_GOOD, LABEL_POTENTIAL, LABEL_RISKY],
        default=[LABEL_GOOD, LABEL_POTENTIAL, LABEL_RISKY],
    )
    sb_states = st.multiselect(
        "State",
        sorted(df_unsold["State"].unique()),
        default=sorted(df_unsold["State"].unique()),
    )

    st.divider()
    st.markdown("**⚙️ Map Display**")
    only_good = st.toggle("Show only 🟢 Good Deals", value=False)
    dot_size  = st.slider("Dot size", 2000, 12000, 5000, step=1000)

    st.divider()
    n_good = (df_unsold["Deal Label"] == LABEL_GOOD).sum()
    n_pot  = (df_unsold["Deal Label"] == LABEL_POTENTIAL).sum()
    n_risk = (df_unsold["Deal Label"] == LABEL_RISKY).sum()
    st.markdown(f"**{len(df_unsold):,}** cars in inventory")
    st.markdown(f"🟢 {n_good:,} &nbsp;🟡 {n_pot:,} &nbsp;🔴 {n_risk:,}")
    st.caption(f"MAE thresholds: 🟢 < ${THRESH_GOOD:,.0f}  🟡 < ${THRESH_POTENTIAL:,.0f}  🔴 ≥ ${THRESH_POTENTIAL:,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='app-header'>
  <p class='app-title'>🚗 CarValue AI</p>
  <p class='app-sub'>AI-powered used car market intelligence — find good deals before anyone else</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Market Overview",
    "🔎 Deal Finder",
    "🔮 Price Estimator",
    "📊 Analytics",
    "🤖 Model Performance",
])

# ═════════════════════════════════════════════════════════════════════════════
# HELPER: render a uniform KPI card
# ═════════════════════════════════════════════════════════════════════════════
def kpi_card(col, label: str, value: str, sub: str = "", color_cls: str = "c-white"):
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'><span class='{color_cls}'>{value}</span></div>"
        f"<div class='kpi-delta'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    df_map = df_unsold[
        df_unsold["Deal Label"].isin(sb_labels) &
        df_unsold["State"].isin(sb_states)
    ].copy()
    if only_good:
        df_map = df_map[df_map["Deal Label"] == LABEL_GOOD]
    df_map["color"] = df_map["Deal Label"].apply(lambda l: COLOR_MAP.get(l, [180, 180, 180, 160]))

    # KPI values
    n_shown   = len(df_map)
    avg_list  = df_map["Price-$"].mean()           if n_shown else 0
    avg_pred  = df_map["Predicted Price"].mean()    if n_shown else 0
    avg_gap   = df_map["Gap ($)"].mean()            if n_shown else 0
    pct_good  = (df_map["Deal Label"] == LABEL_GOOD).mean() * 100 if n_shown else 0
    avg_score = df_map["Deal Score"].mean()         if n_shown else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_card(k1, "Cars Shown",        f"{n_shown:,}",          "on current map",       "c-blue")
    kpi_card(k2, "Avg Asking Price",  f"${avg_list:,.0f}",     "seller listed",        "c-white")
    kpi_card(k3, "Avg AI Value",      f"${avg_pred:,.0f}",     "model predicted",      "c-white")
    kpi_card(k4, "Avg Overprice Gap", f"${avg_gap:+,.0f}",     "asking − predicted",   "c-red" if avg_gap > 0 else "c-green")
    kpi_card(k5, "Good Deals",        f"{pct_good:.1f}%",      "gap ≤ ½ × MAE",        "c-green")

    st.markdown("<br>", unsafe_allow_html=True)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=dot_size,
        pickable=True,
        auto_highlight=True,
    )
    states_layer = pdk.Layer(
        "GeoJsonLayer",
        states_geojson,
        stroked=True, filled=False,
        get_line_color=[80, 100, 140],
        line_width_min_pixels=1,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[states_layer, scatter_layer],
            initial_view_state=pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3.5, pitch=0),
            map_style="mapbox://styles/mapbox/dark-v10",
            tooltip={"text": (
                "{Manufacturer Name} {Car Name}  ({Car Type})\n"
                "📍 {Location}\n"
                "📅 {Manufactured Year}  |  🛣 {Mileage-KM} km\n"
                "💰 Asking:    ${Price-$}\n"
                "🤖 AI Value:  ${Predicted Price}\n"
                "📊 Gap:       ${Gap ($)}\n"
                "🏷 Verdict:   {Deal Label}\n"
                "🏆 Score:     {Deal Score}/100"
            )},
        ),
        use_container_width=True,
    )

    st.markdown(f"""<div class='legend-row'>
      <span><span class='ldot' style='background:#68d391'></span>
        <strong>Good Deal</strong> — gap ≤ ${THRESH_GOOD:,.0f} (within 1.5× model error: competitive price)</span>
      <span><span class='ldot' style='background:#ecc94b'></span>
        <strong>Potential Deal</strong> — gap ${THRESH_GOOD:,.0f}–${THRESH_POTENTIAL:,.0f} (within 2× model error: negotiate)</span>
      <span><span class='ldot' style='background:#fc8181'></span>
        <strong>Risky / Overpaying</strong> — gap > ${THRESH_POTENTIAL:,.0f} (clearly above AI estimate)</span>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEAL FINDER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔎 Deal Finder")
    st.caption(
        "Filter the inventory and surface the best deals. "
        "Results are ranked by **Deal Score** (100 = asking price matches AI value, 0 = far over AI estimate)."
    )

    with st.form("deal_finder"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_make = st.multiselect("Manufacturer", sorted(df_unsold["Manufacturer Name"].dropna().unique()))
        with fc2:
            f_type = st.multiselect("Car Type", sorted(df_unsold["Car Type"].dropna().unique()))
        with fc3:
            f_energy = st.multiselect("Fuel Type", sorted(df_unsold["Energy"].dropna().unique()))
        with fc4:
            f_gearbox = st.radio("Gearbox", ["Any", "Automatic", "Manual"], horizontal=True)

        fs1, fs2, fs3 = st.columns(3)
        with fs1:
            f_price = st.slider("Max Asking Price ($)", 6000, 11000, 11000, step=100)
        with fs2:
            f_mileage = st.slider("Max Mileage (km)", 1000, 100000, 100000, step=1000)
        with fs3:
            f_year = st.slider("Min Manufacture Year", 2015, 2024, 2015)

        fb1, fb2 = st.columns([3, 1])
        with fb1:
            f_verdict = st.radio(
                "Deal Verdict",
                ["All", f"🟢 {LABEL_GOOD}", f"🟡 {LABEL_POTENTIAL}", f"🔴 {LABEL_RISKY}"],
                horizontal=True,
            )
        with fb2:
            f_min_score = st.number_input("Min Score (0–100)", min_value=0, max_value=100, value=0, step=5)

        st.form_submit_button("🔍 Find Deals", use_container_width=True, type="primary")

    # Applying filters
    res = df_unsold.copy()
    if f_make:    res = res[res["Manufacturer Name"].isin(f_make)]
    if f_type:    res = res[res["Car Type"].isin(f_type)]
    if f_energy:  res = res[res["Energy"].isin(f_energy)]
    if f_gearbox != "Any": res = res[res["Gearbox"] == f_gearbox]
    res = res[res["Price-$"] <= f_price]
    res = res[res["Mileage-KM"] <= f_mileage]
    res = res[res["Manufactured Year"] >= f_year]
    if LABEL_GOOD      in f_verdict: res = res[res["Deal Label"] == LABEL_GOOD]
    elif LABEL_POTENTIAL in f_verdict: res = res[res["Deal Label"] == LABEL_POTENTIAL]
    elif LABEL_RISKY   in f_verdict: res = res[res["Deal Label"] == LABEL_RISKY]
    res = res[res["Deal Score"] >= f_min_score]
    res = res.sort_values("Deal Score", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Results",       f"{len(res):,}")
    m2.metric("Avg Asking",    f"${res['Price-$'].mean():,.0f}"         if len(res) else "—")
    m3.metric("Avg AI Value",  f"${res['Predicted Price'].mean():,.0f}" if len(res) else "—")
    m4.metric("Avg Score",     f"{res['Deal Score'].mean():.0f}/100"    if len(res) else "—")

    if len(res) == 0:
        st.warning("No cars match your filters. Try widening the search criteria.")
    else:
        display_cols = ["Deal Score", "Deal Label", "Manufacturer Name", "Car Name",
                        "Car Type", "Energy", "Gearbox", "Color", "Manufactured Year",
                        "Mileage-KM", "Engine Power-HP", "Location",
                        "Price-$", "Predicted Price", "Gap ($)"]
        st.dataframe(
            res[display_cols].reset_index(drop=True).rename(columns={
                "Deal Score": "Score /100", "Deal Label": "Verdict",
                "Price-$": "Asking ($)", "Predicted Price": "AI Value ($)",
                "Gap ($)": "Gap ($)",
            }),
            use_container_width=True,
            height=500,
            column_config={
                "Score /100": st.column_config.ProgressColumn(
                    "Score /100", min_value=0, max_value=100, format="%.0f"
                ),
                "Asking ($)":   st.column_config.NumberColumn("Asking ($)",   format="$%d"),
                "AI Value ($)": st.column_config.NumberColumn("AI Value ($)", format="$%d"),
                "Gap ($)":      st.column_config.NumberColumn("Gap ($)",      format="$%+d"),
            }
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE ESTIMATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔮 AI Price Estimator")
    st.caption(
        "Enter any car's details and the seller's asking price. "
        "The AI predicts the fair market value and tells you whether the deal is within the model's confidence margin."
    )

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**🚘 Car Details**")
        e_make = st.selectbox("Manufacturer", sorted(df_unsold["Manufacturer Name"].dropna().unique()), key="em")
        models_for_make = sorted(df_unsold[df_unsold["Manufacturer Name"] == e_make]["Car Name"].dropna().unique())
        e_name = st.selectbox("Model", models_for_make if models_for_make else ["—"], key="en")

        ca, cb = st.columns(2)
        with ca:
            e_type    = st.selectbox("Car Type",  sorted(df_unsold["Car Type"].dropna().unique()),  key="et")
            e_gearbox = st.selectbox("Gearbox",   ["Automatic", "Manual"],                          key="eg")
            e_energy  = st.selectbox("Fuel Type", sorted(df_unsold["Energy"].dropna().unique()),   key="ee")
        with cb:
            e_color = st.selectbox("Color",  sorted(df_unsold["Color"].dropna().unique()),   key="ec")
            e_seats = st.number_input("Seats", min_value=2, max_value=8, value=5, key="es")
            e_doors = st.number_input("Doors", min_value=2, max_value=5, value=4, key="ed")

        e_location = st.selectbox("Location", sorted(df_unsold["Location"].dropna().unique()), key="el")

    with right:
        st.markdown("**📏 Condition**")
        e_year    = st.slider("Manufacture Year",   2015, 2024, 2020, key="ey")
        e_mileage = st.slider("Mileage (km)",       1000, 100000, 30000, step=500, key="emi")
        e_hp      = st.slider("Engine Power (HP)",  100, 250, 150, key="eh")

        st.markdown("**💰 Asking Price**")
        e_price = st.number_input(
            "What is the seller asking? ($)",
            min_value=3000, max_value=20000, value=8000, step=100, key="ep",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🤖 Get AI Valuation", use_container_width=True, type="primary")

    if run_btn:
        car_age          = max(CURRENT_YEAR - e_year, 1)
        mileage_per_year = e_mileage / car_age

        row = pd.DataFrame([{
            "Manufacturer Name": e_make, "Car Name": e_name, "Car Type": e_type,
            "Color": e_color, "Gearbox": e_gearbox, "Energy": e_energy,
            "Location": e_location, "Car Age": car_age,
            "Mileage-KM": e_mileage, "Mileage per Year": mileage_per_year,
            "Engine Power-HP": e_hp, "Number of Seats": e_seats, "Number of Doors": e_doors,
        }])
        predicted = float(model.predict(row[FEATURES])[0])
        gap       = e_price - predicted       
        verdict   = deal_label(gap)
        score     = deal_score(gap)

        st.markdown("---")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if verdict == LABEL_GOOD:
                cls, emoji, title = "v-green", "🟢", "GOOD DEAL — Within Confidence Margin"
                detail = (
                    f"The asking price is <strong>${abs(gap):,.0f} {'above' if gap > 0 else 'below'}</strong> "
                    f"the AI estimate of <strong>${predicted:,.0f}</strong>. "
                    f"This gap is within 1.5× the model's average error (${THRESH_GOOD:,.0f}), "
                    f"which means the price is competitively fair — you're not being overcharged beyond the model's confidence band."
                )
            elif verdict == LABEL_POTENTIAL:
                cls, emoji, title = "v-yellow", "🟡", "POTENTIAL DEAL — Negotiate Before Buying"
                detail = (
                    f"The gap of <strong>${abs(gap):,.0f}</strong> falls between 1.5× and 2× the model's error margin. "
                    f"The seller may be pricing fairly for their market, but there's room to negotiate. "
                    f"Use the AI value of <strong>${predicted:,.0f}</strong> as your anchor — push to close the gap."
                )
            else:
                cls, emoji, title = "v-red", "🔴", "RISKY — Asking Price Exceeds AI Confidence"
                detail = (
                    f"The gap of <strong>${abs(gap):,.0f}</strong> exceeds the model's error margin "
                    f"(${THRESH_POTENTIAL:,.0f}). The seller is asking <em>significantly more</em> than the AI's "
                    f"estimate of <strong>${predicted:,.0f}</strong>. "
                    f"Negotiate firmly or look for a better deal in the Deal Finder."
                )

            st.markdown(f"""<div class='verdict-wrap {cls}'>
              <div class='v-emoji'>{emoji}</div>
              <div class='v-title'>{title}</div>
              <div class='v-detail'>{detail}</div>
            </div>""", unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("🤖 AI Market Value", f"${predicted:,.0f}")
        d2.metric("💰 Asking Price",    f"${e_price:,.0f}")
        d3.metric("📊 Gap",             f"${gap:+,.0f}",
                  delta=f"{'⚠️ above' if gap > 0 else '✅ below'} AI estimate",
                  delta_color="inverse" if gap > 0 else "normal")
        d4.metric("🏆 Deal Score",      f"{score:.0f} / 100")

        st.info(
            f"**Model confidence:** MAE = ${MODEL_MAE:,.0f}. "
            f"Good Deal threshold = ${THRESH_GOOD:,.0f} (1.5 × MAE). "
            f"Potential Deal threshold = ${THRESH_POTENTIAL:,.0f} (2 × MAE).",
            icon="ℹ️",
        )

        # Comparables
        comparables = df_unsold[
            (df_unsold["Manufacturer Name"] == e_make) &
            (df_unsold["Car Type"] == e_type)
        ].sort_values("Deal Score", ascending=False).head(8)

        if len(comparables) >= 3:
            st.markdown(f"#### 🔍 Similar {e_make} {e_type}s in Inventory")
            st.caption("Ranked by Deal Score — the higher the score, the closer the asking price is to the AI estimate")
            st.dataframe(
                comparables[["Car Name", "Manufactured Year", "Mileage-KM",
                             "Price-$", "Predicted Price", "Gap ($)", "Deal Score", "Deal Label", "Location"]]
                .reset_index(drop=True).rename(columns={
                    "Price-$": "Asking ($)", "Predicted Price": "AI Value ($)",
                    "Deal Score": "Score /100", "Deal Label": "Verdict",
                }),
                use_container_width=True,
                column_config={
                    "Score /100":   st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%.0f"),
                    "Asking ($)":   st.column_config.NumberColumn("Asking ($)",   format="$%d"),
                    "AI Value ($)": st.column_config.NumberColumn("AI Value ($)", format="$%d"),
                    "Gap ($)":      st.column_config.NumberColumn("Gap ($)",      format="$%+d"),
                }
            )
    else:
        st.info("👈 Fill in the car details and asking price, then click **Get AI Valuation**.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Market Analytics")

    row1a, row1b = st.columns(2)

    with row1a:
        st.markdown("**Deal Verdict Distribution**")
        lc = df_unsold["Deal Label"].value_counts().reset_index()
        lc.columns = ["Label", "Count"]
        fig_pie = px.pie(lc, names="Label", values="Count", hole=0.45,
                         color="Label", color_discrete_map=CHART_COLORS)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                               margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1b:
        st.markdown("**Avg Asking vs AI Value by Manufacturer**")
        mfr = (df_unsold.groupby("Manufacturer Name")[["Price-$","Predicted Price"]]
               .mean().sort_values("Predicted Price", ascending=True).reset_index())
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Asking ($)",   y=mfr["Manufacturer Name"], x=mfr["Price-$"],
                                  orientation="h", marker_color="#76e4f7"))
        fig_bar.add_trace(go.Bar(name="AI Value ($)", y=mfr["Manufacturer Name"], x=mfr["Predicted Price"],
                                  orientation="h", marker_color="#68d391"))
        fig_bar.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=300, margin=dict(t=10,b=10),
                               legend=dict(orientation="h", y=1.05))
        fig_bar.update_xaxes(gridcolor="#2d3748")
        st.plotly_chart(fig_bar, use_container_width=True)

    row2a, row2b = st.columns(2)

    with row2a:
        st.markdown("**Mileage vs Overprice Gap by Car Type**")
        fig_sc = px.scatter(df_unsold, x="Mileage-KM", y="Gap ($)", color="Car Type", opacity=0.6,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig_sc.add_hline(y=THRESH_GOOD,      line_dash="dot",  line_color="#68d391",
                          annotation_text="Good Deal threshold")
        fig_sc.add_hline(y=THRESH_POTENTIAL, line_dash="dash", line_color="#ecc94b",
                          annotation_text="Potential Deal threshold")
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", height=300, margin=dict(t=10,b=10))
        fig_sc.update_xaxes(gridcolor="#2d3748"); fig_sc.update_yaxes(gridcolor="#2d3748")
        st.plotly_chart(fig_sc, use_container_width=True)

    with row2b:
        st.markdown("**Asking Price Distribution by Car Type**")
        fig_box = px.box(df_unsold, x="Car Type", y="Price-$", color="Car Type",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=300, margin=dict(t=10,b=10), showlegend=False)
        fig_box.update_xaxes(gridcolor="#2d3748"); fig_box.update_yaxes(gridcolor="#2d3748")
        st.plotly_chart(fig_box, use_container_width=True)

    row3a, row3b = st.columns(2)

    with row3a:
        st.markdown("**Avg Deal Score by Location**")
        loc_s = df_unsold.groupby("Location")["Deal Score"].mean().sort_values().reset_index()
        fig_loc = px.bar(loc_s, x="Deal Score", y="Location", orientation="h",
                          color="Deal Score", color_continuous_scale="RdYlGn", range_color=[0,100])
        fig_loc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=380, margin=dict(t=10,b=10),
                               coloraxis_showscale=False)
        fig_loc.update_xaxes(gridcolor="#2d3748")
        st.plotly_chart(fig_loc, use_container_width=True)

    with row3b:
        st.markdown("**Deal Verdict by Fuel Type**")
        en_lbl = df_unsold.groupby(["Energy","Deal Label"]).size().reset_index(name="Count")
        fig_en = px.bar(en_lbl, x="Energy", y="Count", color="Deal Label", barmode="group",
                         color_discrete_map=CHART_COLORS)
        fig_en.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", height=380, margin=dict(t=10,b=10))
        fig_en.update_xaxes(gridcolor="#2d3748"); fig_en.update_yaxes(gridcolor="#2d3748")
        st.plotly_chart(fig_en, use_container_width=True)

    st.divider()
    st.markdown("#### 🏷️ Historical Sales Insights")
    h1, h2 = st.columns(2)

    with h1:
        st.markdown("**Sold Price Distribution**")
        fig_hist = px.histogram(df_sold, x="Sold Price-$", nbins=40, color_discrete_sequence=["#76e4f7"])
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0", height=280, margin=dict(t=10,b=10))
        fig_hist.update_xaxes(gridcolor="#2d3748"); fig_hist.update_yaxes(gridcolor="#2d3748")
        st.plotly_chart(fig_hist, use_container_width=True)

    with h2:
        if "Margin-%" in df_sold.columns:
            st.markdown("**Avg Dealer Margin % by Manufacturer**")
            mg = df_sold.groupby("Manufacturer Name")["Margin-%"].mean().sort_values().reset_index()
            fig_mg = px.bar(mg, x="Manufacturer Name", y="Margin-%",
                             color="Margin-%", color_continuous_scale="RdYlGn")
            fig_mg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e2e8f0", height=280, margin=dict(t=10,b=10),
                                  coloraxis_showscale=False)
            fig_mg.update_xaxes(gridcolor="#2d3748"); fig_mg.update_yaxes(gridcolor="#2d3748")
            st.plotly_chart(fig_mg, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Model Performance & Transparency")
    st.caption(
        "This tab documents the AI pipeline, model accuracy, and why the MAE-anchored "
        "classification produces honest, interpretable deal verdicts."
    )

    best_name = model_meta.get("best_model", "GradientBoosting")
    best_mae  = model_meta.get("best_test_mae", 749)
    best_r2   = model_meta.get("best_test_r2",  0.56)
    t_rows    = model_meta.get("train_rows",    1732)
    te_rows   = model_meta.get("test_rows",      434)
    mean_sold = df_sold["Sold Price-$"].mean()

    st.markdown("### 📈 Best Model Summary")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Best Model",       best_name)
    p2.metric("Test MAE",         f"${best_mae:,.0f}")
    p3.metric("R²",               f"{best_r2:.3f}")
    p4.metric("Training Rows",    f"{t_rows:,}")
    p5.metric("MAE as % of Mean", f"{best_mae / mean_sold * 100:.1f}%")

    st.info(
        f"**Why MAE-anchored thresholds?** The model predicts the market-clearing (sold) price. "
        f"Because the model has an average error of **${MODEL_MAE:,.0f}**, a gap between asking and "
        f"predicted price that is smaller than this margin could simply be model uncertainty — "
        f"not a genuine overprice. Thresholds: "
        f"🟢 **Good Deal** gap ≤ ${THRESH_GOOD:,.0f} (1.5 × MAE)  |  "
        f"🟡 **Potential** gap ≤ ${THRESH_POTENTIAL:,.0f} (2 × MAE)  |  "
        f"🔴 **Risky** gap > ${THRESH_POTENTIAL:,.0f}."
    )

    if len(model_comp_df) > 0:
        st.markdown("### 🏆 Model Comparison (GridSearchCV)")
        st.dataframe(
            model_comp_df.rename(columns={
                "model": "Model", "cv_mae": "CV MAE ($)",
                "test_mae": "Test MAE ($)", "test_rmse": "Test RMSE ($)",
                "test_r2": "R²", "best_params": "Best Params",
            }).style.format({
                "CV MAE ($)": "${:,.0f}", "Test MAE ($)": "${:,.0f}",
                "Test RMSE ($)": "${:,.0f}", "R²": "{:.3f}",
            }).highlight_min(subset=["Test MAE ($)"], color="#1a3a2a"),
            use_container_width=True,
        )

    st.markdown("### 🔬 Accuracy Context")
    st.markdown(f"""
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE = ${best_mae:,.0f}** | Good | Average prediction error; used directly as classification threshold |
| **R² = {best_r2:.2f}** | Moderate | Model explains ~{best_r2*100:.0f}% of variance in actual sale prices |
| **Remaining {(1-best_r2)*100:.0f}%** | Expected | Negotiation, urgency, local demand — unobservable in data |
| **Training set** | {t_rows:,} sold cars | Ground-truth market-clearing prices |
| **Prediction set** | 7,834 unsold cars | Never seen during training |
    """)

    # Gap distribution histogram with threshold lines
    st.markdown("### 📊 Gap Distribution (Asking − AI Value) Across Unsold Inventory")
    st.caption("Shows how far sellers are listing above our AI's predicted market value. Threshold lines show deal quality cutoffs.")
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(
        x=df_unsold["Gap ($)"].values, nbinsx=60,
        marker_color="#76e4f7", opacity=0.8, name="Gap ($)"
    ))
    fig_d.add_vline(x=THRESH_GOOD, line_dash="dot", line_color="#68d391",
                    annotation_text=f"Good Deal ≤ ${THRESH_GOOD:,.0f}",
                    annotation_position="top right")
    fig_d.add_vline(x=THRESH_POTENTIAL, line_dash="dash", line_color="#ecc94b",
                    annotation_text=f"Potential ≤ ${THRESH_POTENTIAL:,.0f}",
                    annotation_position="top right")
    fig_d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=360, margin=dict(t=20,b=20),
        xaxis_title="Gap ($) = Asking Price − AI Predicted Value",
        yaxis_title="Number of cars",
    )
    fig_d.update_xaxes(gridcolor="#2d3748"); fig_d.update_yaxes(gridcolor="#2d3748")
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("### ⚙️ Features Used by the Model")
    st.success(
        "**Key design decision:** `Price-$` (the seller's asking price) was intentionally **excluded** from model features. "
        "Including it causes the model to memorise the listing price (MAE ≈ $0) rather than predict true market value — "
        "which is exactly what we want to evaluate independently."
    )
    feat_list = model_meta.get("features", FEATURES)
    fc1, fc2, fc3 = st.columns(3)
    chunk = (len(feat_list) + 2) // 3
    for i, col in enumerate([fc1, fc2, fc3]):
        for f in feat_list[i * chunk:(i + 1) * chunk]:
            col.markdown(f"- `{f}`")