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


# ── search ────────────────────────────────────────────────────────────────────
def semantic_search(query: str, index, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Embed a free-text query and return the top-k most similar cars.
    """
    encoder = _get_encoder()
    q_vec = encoder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    scores, indices = index.search(q_vec, k)
    results = df.iloc[indices[0]].copy()
    results["Similarity"] = scores[0]
    return results.reset_index(drop=True)
