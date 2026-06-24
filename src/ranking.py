"""
ranking.py — orchestrates the full ranking pipeline.

Pipeline:
  1. Load precomputed artifacts (FAISS index, embeddings, IDs)
  2. Retrieve top-K candidates by JD cosine similarity
  3. Re-rank with hybrid structured scores
  4. Apply consistency penalty
  5. Sort deterministically (score desc, candidate_id asc for ties)
  6. Generate reasoning for top 100
  7. Return DataFrame-ready list of dicts
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    CAND_FEATURES_PATH,
    FAISS_RETRIEVAL_K,
    FINAL_TOP_N,
    SCORE_PRECISION,
)
from .data_loader import load_all_candidates
from .embeddings import load_artifacts, retrieve_top_k
from .reasoning import generate_reasoning
from .scoring import compute_all_scores

logger = logging.getLogger(__name__)


def load_features() -> Optional[Dict[str, dict]]:
    """Load precomputed structured features if available."""
    if Path(CAND_FEATURES_PATH).exists():
        with open(CAND_FEATURES_PATH, "rb") as fh:
            return pickle.load(fh)
    return None


def rank_candidates(candidates_path: str) -> List[dict]:
    """
    Full ranking pipeline. Returns list of dicts ready for CSV write.
    Each dict: {candidate_id, rank, score, reasoning, _scores (debug)}
    """
    # ── Step 1: load data ──────────────────────────────────────────────────────
    logger.info("Loading candidates from %s …", candidates_path)
    candidates = load_all_candidates(candidates_path)
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    logger.info("Loaded %d candidates", len(candidates))

    # ── Step 2: load artifacts ─────────────────────────────────────────────────
    logger.info("Loading FAISS artifacts …")
    index, jd_emb, cand_embs, cand_ids = load_artifacts()

    # Validate ID alignment
    faiss_id_set = set(cand_ids)
    dataset_id_set = set(cand_by_id.keys())
    missing_in_data = faiss_id_set - dataset_id_set
    if missing_in_data:
        logger.warning(
            "%d candidate IDs in FAISS not found in dataset (stale artifacts?)",
            len(missing_in_data),
        )

    # ── Step 3: FAISS retrieval ────────────────────────────────────────────────
    k = min(FAISS_RETRIEVAL_K, len(cand_ids))
    logger.info("FAISS retrieval: top %d candidates …", k)
    top_ids, cosine_scores = retrieve_top_k(index, cand_ids, jd_emb, k)

    # Build id → cosine_score lookup
    sem_scores = dict(zip(top_ids, cosine_scores))

    # ── Step 4: structured scoring + penalty ──────────────────────────────────
    logger.info("Scoring %d candidates …", len(top_ids))
    scored = []
    for cid in top_ids:
        c = cand_by_id.get(cid)
        if c is None:
            continue
        sem = float(sem_scores.get(cid, 0.0))
        score_dict = compute_all_scores(c, sem)
        scored.append((cid, score_dict))

    # ── Step 5: deterministic sort ─────────────────────────────────────────────
    # Primary: score desc; Tie-break: candidate_id asc (matches validator rule)
    def sort_key(item):
        cid, s = item
        return (-round(s["final"], SCORE_PRECISION), cid)

    scored.sort(key=sort_key)

    # ── Step 6: take top 100 + generate reasoning ──────────────────────────────
    top100 = scored[:FINAL_TOP_N]
    if len(top100) < FINAL_TOP_N:
        logger.error(
            "Only %d candidates available; need %d", len(top100), FINAL_TOP_N
        )

    results = []
    prev_score = None
    for rank_idx, (cid, s) in enumerate(top100, start=1):
        c = cand_by_id[cid]
        score_val = round(s["final"], SCORE_PRECISION)

        # Ensure non-increasing (safety clamp for floating-point edge cases)
        if prev_score is not None and score_val > prev_score:
            score_val = prev_score
        prev_score = score_val

        reasoning = generate_reasoning(c, s)

        results.append(
            {
                "candidate_id": cid,
                "rank": rank_idx,
                "score": score_val,
                "reasoning": reasoning,
                "_scores": s,    # kept for debug; stripped before CSV write
            }
        )

    logger.info("Ranking complete. Top score: %.4f  Bottom score: %.4f",
                results[0]["score"] if results else 0,
                results[-1]["score"] if results else 0)
    return results


def write_submission_csv(results: List[dict], out_path: str) -> None:
    """Write the final submission CSV in the required format."""
    import csv

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in results:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                row["score"],
                row["reasoning"],
            ])

    logger.info("Submission written to %s (%d rows)", out, len(results))
