"""
src/embeddings.py

Builds a FAISS vector index over the unsold car inventory.
Each car is converted into a natural-language sentence, then embedded
with sentence-transformers/all-MiniLM-L6-v2.

Usage
-----
    from src.embeddings import build_index, load_index
    index, df = build_index(df_unsold)          # first run
    index, df = load_index()                    # subsequent runs
"""

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── paths ─────────────────────────────────────────────────────────────────────
INDEX_PATH = "model/faiss_index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── singleton encoder (cached at module level) ─────────────────────────────────
_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


# ── car → sentence ─────────────────────────────────────────────────────────────
def car_to_sentence(row: pd.Series) -> str:
    """Convert a car row into a descriptive natural-language string."""
    age = row.get("Car Age", "unknown")
    return (
        f"{row.get('Manufactured Year', '')} {row.get('Manufacturer Name', '')} "
        f"{row.get('Car Name', '')} {row.get('Car Type', '')}, "
        f"{row.get('Energy', '')} engine, {row.get('Gearbox', '')} gearbox, "
        f"{row.get('Color', '')} colour, "
        f"{int(row.get('Mileage-KM', 0)):,} km, "
        f"{int(row.get('Engine Power-HP', 0))} HP, "
        f"{int(row.get('Number of Seats', 0))} seats, "
        f"located in {row.get('Location', '')}, "
        f"asking ${int(row.get('Price-$', 0)):,}"
    )


# ── build index ───────────────────────────────────────────────────────────────
def build_index(df: pd.DataFrame, save: bool = True) -> tuple:
    """
    Encode all cars and build a FAISS flat L2 index.
    Returns (index, df_with_sentences).
    """
    import faiss  # lazy import — only needed here

    encoder = _get_encoder()
    df = df.copy().reset_index(drop=True)
    df["_sentence"] = df.apply(car_to_sentence, axis=1)

    sentences = df["_sentence"].tolist()
    embeddings = encoder.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on normalised vecs == cosine similarity
    index.add(embeddings)

    if save:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"index": index, "df": df, "embeddings": embeddings}, f)

    return index, df, embeddings


# ── load cached index ─────────────────────────────────────────────────────────
def load_index() -> tuple | None:
    """Load a previously saved index. Returns (index, df, embeddings) or None."""
    if not os.path.exists(INDEX_PATH):
        return None
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["df"], data["embeddings"]


# ── attribute extractor ───────────────────────────────────────────────────────
def _extract_filters(query: str, df: pd.DataFrame) -> dict:
    """
    Scan the query for explicitly mentioned structured attributes.
    Returns a dict of {column: value} for any that are found.
    Only detects attributes whose values actually exist in the dataset.
    """
    q = query.lower()
    filters = {}

    # Color
    for val in df["Color"].dropna().unique():
        if val.lower() in q:
            filters["Color"] = val
            break

    # Car Type
    for val in df["Car Type"].dropna().unique():
        if val.lower() in q:
            filters["Car Type"] = val
            break

    # Energy / fuel
    for val in df["Energy"].dropna().unique():
        if val.lower() in q:
            filters["Energy"] = val
            break

    # Gearbox
    if "automatic" in q:
        filters["Gearbox"] = "Automatic"
    elif "manual" in q:
        filters["Gearbox"] = "Manual"

    # Manufacturer
    for val in df["Manufacturer Name"].dropna().unique():
        if val.lower() in q:
            filters["Manufacturer Name"] = val
            break

    # Location / state
    for val in df["Location"].dropna().unique():
        if val.lower() in q:
            filters["Location"] = val
            break

    # Number of doors
    import re
    door_match = re.search(r"(\d)\s*door", q)
    if door_match:
        filters["Number of Doors"] = int(door_match.group(1))

    # Number of seats
    seat_match = re.search(r"(\d)\s*seat", q)
    if seat_match:
        filters["Number of Seats"] = int(seat_match.group(1))

    return filters


# ── search ────────────────────────────────────────────────────────────────────
def semantic_search(query: str, index, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Embed a free-text query and return the top-k most similar cars.
    If the query explicitly mentions structured attributes (color, car type,
    gearbox, fuel, manufacturer, location, doors, seats), results are filtered
    to exact matches on those attributes only — other attributes remain open.
    """
    encoder = _get_encoder()
    q_vec = encoder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    # Detect any explicitly mentioned filters
    filters = _extract_filters(query, df)

    if filters:
        # Apply hard filters to get a candidate pool
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in filters.items():
            if col in df.columns:
                mask &= df[col] == val
        df_pool = df[mask].copy()

        if len(df_pool) == 0:
            # No exact matches — fall back to unfiltered search
            df_pool = df.copy()
            filters = {}
    else:
        df_pool = df.copy()

    # Search within the pool using the original FAISS indices
    # Map pool rows back to their FAISS index positions
    pool_indices = df_pool.index.tolist()

    # Fetch a large candidate set from FAISS then intersect with pool
    fetch_k = min(len(df), max(k * 20, 500))
    scores, indices = index.search(q_vec, fetch_k)

    # Keep only results that are in the filtered pool
    pool_set = set(pool_indices)
    matched = [(idx, score) for idx, score in zip(indices[0], scores[0]) if idx in pool_set]

    # Take top k
    matched = matched[:k]

    if not matched:
        # Fallback: just return top-k from pool by order
        results = df_pool.head(k).copy()
        results["Similarity"] = 0.0
        results["_filters_applied"] = str(filters)
        return results.reset_index(drop=True)

    result_indices = [m[0] for m in matched]
    result_scores  = [m[1] for m in matched]

    results = df.iloc[result_indices].copy()
    results["Similarity"] = result_scores
    results["_filters_applied"] = str(filters) if filters else ""
    return results.reset_index(drop=True)