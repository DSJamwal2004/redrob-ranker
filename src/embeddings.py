"""
embeddings.py — sentence-transformer encoding + FAISS index build/load.

All functions are CPU-only and deterministic given fixed model weights.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .constants import (
    CAND_EMBEDDINGS_PATH,
    CAND_IDS_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    JD_EMBEDDING_PATH,
    JD_TEXT,
)

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once per process
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode_texts(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of strings → float32 ndarray (N, DIM).
    Batched; no GPU; deterministic.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine
    )
    return embeddings.astype(np.float32)


def encode_jd() -> np.ndarray:
    """Return the JD embedding (shape: [1, DIM])."""
    return encode_texts([JD_TEXT], show_progress=False)


# ─── FAISS helpers ────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat inner-product index (cosine via L2-normalised vectors).
    FlatIP is exact, deterministic, and fast enough for 100K × 384.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors", index.ntotal)
    return index


def save_artifacts(
    index: faiss.Index,
    jd_emb: np.ndarray,
    cand_embs: np.ndarray,
    cand_ids: List[str],
) -> None:
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(str(JD_EMBEDDING_PATH), jd_emb)
    np.save(str(CAND_EMBEDDINGS_PATH), cand_embs)
    # store IDs as object array so numpy preserves strings
    np.save(str(CAND_IDS_PATH), np.array(cand_ids, dtype=object))
    logger.info("Artifacts saved to %s", FAISS_INDEX_PATH.parent)


def load_artifacts() -> Tuple[faiss.Index, np.ndarray, np.ndarray, List[str]]:
    """Load precomputed index + embeddings from disk."""
    for p in [FAISS_INDEX_PATH, JD_EMBEDDING_PATH, CAND_EMBEDDINGS_PATH, CAND_IDS_PATH]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}\n"
                "Run:  python scripts/precompute.py --candidates <path>"
            )
    index    = faiss.read_index(str(FAISS_INDEX_PATH))
    jd_emb   = np.load(str(JD_EMBEDDING_PATH))
    cand_embs = np.load(str(CAND_EMBEDDINGS_PATH))
    cand_ids  = list(np.load(str(CAND_IDS_PATH), allow_pickle=True))
    logger.info("Artifacts loaded: %d candidates", index.ntotal)
    return index, jd_emb, cand_embs, cand_ids


def retrieve_top_k(
    index: faiss.Index,
    cand_ids: List[str],
    jd_emb: np.ndarray,
    k: int,
) -> Tuple[List[str], np.ndarray]:
    """
    FAISS retrieval: return (top_k_ids, cosine_scores) sorted by similarity desc.
    """
    k = min(k, index.ntotal)
    scores, idxs = index.search(jd_emb, k)
    scores = scores[0]
    idxs   = idxs[0]
    # cosine ∈ [-1, 1] → scale to [0, 1]
    cosine_scaled = ((scores + 1.0) / 2.0).clip(0.0, 1.0)
    top_ids = [cand_ids[i] for i in idxs]
    return top_ids, cosine_scaled
