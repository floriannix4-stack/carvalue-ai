"""
src/retrieval.py

Retrieval layer for the chatbot.
Given a user question, returns structured context as a string
that is safe to pass into an LLM prompt.

Two retrieval modes
-------------------
1. Semantic  — find the k cars most similar to the question (via FAISS)
2. Aggregate — compute summary statistics for a filtered slice of the dataset

The router picks the right mode based on simple keyword detection,
with semantic search as the fallback.
"""

import re
import pandas as pd
import numpy as np

from src.embeddings import semantic_search


# ── keyword patterns that suggest aggregation queries ─────────────────────────
_AGG_PATTERNS = [
    r"\baverage\b", r"\bavg\b", r"\bmean\b", r"\bhow many\b",
    r"\bcount\b", r"\btotal\b", r"\bmost (expensive|common|popular|affordable)\b",
    r"\bcheapest\b", r"\bpriciest\b", r"\bstatistic\b",
    r"\bcompare\b", r"\bbreakdown\b", r"\bdistribution\b",
]
_AGG_RE = re.compile("|".join(_AGG_PATTERNS), re.IGNORECASE)

# ── keyword patterns that suggest "show me the best deals" sorted queries ──────
_BEST_PATTERNS = [
    r"\bbest deal\b", r"\btop deal\b", r"\bhighest score\b", r"\bbest score\b",
    r"\bmost undervalued\b", r"\bbiggest discount\b", r"\bbest value\b",
    r"\babsolute best\b", r"\btop ([\d]+)? ?(deal|value|score)\b",
]
_BEST_RE = re.compile("|".join(_BEST_PATTERNS), re.IGNORECASE)


def _detect_mode(question: str) -> str:
    if _BEST_RE.search(question):
        return "best_deals"
    if _AGG_RE.search(question):
        return "aggregate"
    return "semantic"


# ── aggregate retrieval ───────────────────────────────────────────────────────
def _aggregate_context(question: str, df: pd.DataFrame) -> str:
    """
    Compute useful statistics from the dataset and format as a readable string.
    Tries to filter by manufacturer / car type / energy if mentioned.
    """
    q = question.lower()
    filtered = df.copy()

    # Try to narrow by manufacturer
    for mfr in df["Manufacturer Name"].dropna().unique():
        if mfr.lower() in q:
            filtered = filtered[filtered["Manufacturer Name"].str.lower() == mfr.lower()]
            break

    # Try to narrow by car type
    for ct in df["Car Type"].dropna().unique():
        if ct.lower() in q:
            filtered = filtered[filtered["Car Type"].str.lower() == ct.lower()]
            break

    # Try to narrow by energy / fuel
    for en in df["Energy"].dropna().unique():
        if en.lower() in q:
            filtered = filtered[filtered["Energy"].str.lower() == en.lower()]
            break

    if filtered.empty:
        filtered = df

    lines = [
        f"Dataset slice: {len(filtered):,} cars",
        f"Avg asking price:      ${filtered['Price-$'].mean():,.0f}",
        f"Avg predicted value:   ${filtered['Predicted Price'].mean():,.0f}",
        f"Avg mileage:           {filtered['Mileage-KM'].mean():,.0f} km",
        f"Avg engine power:      {filtered['Engine Power-HP'].mean():.0f} HP",
        f"Avg car age:           {filtered['Car Age'].mean():.1f} years",
        "",
        "Deal breakdown:",
    ]

    for label, count in filtered["Deal Label"].value_counts().items():
        pct = count / len(filtered) * 100
        lines.append(f"  {label}: {count:,} ({pct:.1f}%)")

    lines += [
        "",
        "Top 5 by Deal Score:",
    ]
    top5 = filtered.nlargest(5, "Deal Score")[
        ["Manufacturer Name", "Car Name", "Car Type", "Energy",
         "Manufactured Year", "Mileage-KM", "Price-$", "Predicted Price", "Deal Score", "Deal Label"]
    ]
    for _, r in top5.iterrows():
        lines.append(
            f"  {r['Manufactured Year']} {r['Manufacturer Name']} {r['Car Name']} "
            f"({r['Car Type']}, {r['Energy']}) — "
            f"asking ${r['Price-$']:,.0f}, AI value ${r['Predicted Price']:,.0f}, "
            f"score {r['Deal Score']:.0f}/100, verdict: {r['Deal Label']}"
        )

    return "\n".join(lines)


# ── best deals retrieval ──────────────────────────────────────────────────────
def _best_deals_context(question: str, df: pd.DataFrame, k: int = 10) -> str:
    """Sort by Deal Score descending and return the top k — with optional filters."""
    q = question.lower()
    filtered = df.copy()

    # Optional filters by manufacturer / type / energy / location
    for mfr in df["Manufacturer Name"].dropna().unique():
        if mfr.lower() in q:
            filtered = filtered[filtered["Manufacturer Name"].str.lower() == mfr.lower()]
            break
    for ct in df["Car Type"].dropna().unique():
        if ct.lower() in q:
            filtered = filtered[filtered["Car Type"].str.lower() == ct.lower()]
            break
    for en in df["Energy"].dropna().unique():
        if en.lower() in q:
            filtered = filtered[filtered["Energy"].str.lower() == en.lower()]
            break
    for loc in df["Location"].dropna().unique():
        if loc.lower() in q:
            filtered = filtered[filtered["Location"].str.lower() == loc.lower()]
            break

    if filtered.empty:
        filtered = df

    top = filtered.nlargest(k, "Deal Score")
    lines = [f"Top {len(top)} best deals by Deal Score (sorted highest first):", ""]
    for i, r in top.iterrows():
        lines.append(
            f"{len(lines)-1}. {int(r['Manufactured Year'])} {r['Manufacturer Name']} {r['Car Name']} "
            f"({r['Car Type']}, {r['Energy']}, {r['Gearbox']})\n"
            f"   {int(r['Mileage-KM']):,} km | {int(r['Engine Power-HP'])} HP | {r['Location']}\n"
            f"   Asking: ${r['Price-$']:,.0f} | AI value: ${r['Predicted Price']:,.0f} | "
            f"Gap: ${r['Gap ($)']:+,.0f} | Score: {r['Deal Score']:.0f}/100 | Verdict: {r['Deal Label']}"
        )
    return "\n".join(lines)



def _semantic_context(question: str, index, df: pd.DataFrame, k: int = 6) -> str:
    """Run FAISS search and format results as readable context."""
    results = semantic_search(question, index, df, k=k)
    lines = [f"Top {len(results)} semantically similar cars for: '{question}'", ""]
    for i, r in results.iterrows():
        lines.append(
            f"{i+1}. {r['Manufactured Year']} {r['Manufacturer Name']} {r['Car Name']} "
            f"({r['Car Type']}, {r['Energy']}, {r['Gearbox']})\n"
            f"   Colour: {r['Color']} | {int(r['Mileage-KM']):,} km | "
            f"{int(r['Engine Power-HP'])} HP | {r['Location']}\n"
            f"   Asking: ${r['Price-$']:,.0f} | AI value: ${r['Predicted Price']:,.0f} | "
            f"Gap: ${r['Gap ($)']:+,.0f} | Verdict: {r['Deal Label']} | "
            f"Score: {r['Deal Score']:.0f}/100\n"
            f"   Similarity: {r['Similarity']:.3f}"
        )
    return "\n".join(lines)


# ── public API ────────────────────────────────────────────────────────────────
def retrieve_context(
    question: str,
    df: pd.DataFrame,
    index=None,
    k: int = 6,
    force_mode: str | None = None,
) -> tuple[str, str]:
    """
    Returns (context_string, mode_used).
    mode_used is 'semantic' or 'aggregate'.
    """
    mode = force_mode or _detect_mode(question)

    if mode == "best_deals":
        return _best_deals_context(question, df, k=k), "best_deals"
    if mode == "aggregate" or index is None:
        return _aggregate_context(question, df), "aggregate"

    return _semantic_context(question, index, df, k=k), "semantic"