#!/usr/bin/env python3
"""
retrieval_diagnostics.py — inspect FAISS retrieval quality.

Shows:
  - JD embedding stats
  - Score distribution of retrieved candidates
  - Top-20 retrieved candidates by semantic similarity
  - Coverage: how many of top-K have relevant titles

Usage:
    python scripts/retrieval_diagnostics.py --candidates path/to/candidates.jsonl
    python scripts/retrieval_diagnostics.py --candidates path/to/candidates.jsonl --k 200
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DOMAIN_TITLE_HINTS = [
    "ml", "machine learning", "nlp", "ranking", "retrieval", "search",
    "recommendation", "embedding", "data science", "ai", "artificial intelligence",
]


def _has_domain_title(profile: dict) -> bool:
    title = (profile.get("current_title") or "").lower()
    return any(h in title for h in DOMAIN_TITLE_HINTS)


def parse_args():
    p = argparse.ArgumentParser(description="FAISS retrieval diagnostics")
    p.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    p.add_argument("--k", type=int, default=None, help="Override retrieval_k for this run")
    p.add_argument("--top", type=int, default=20, help="How many top candidates to display (default: 20)")
    return p.parse_args()


def main():
    args = parse_args()

    from src.constants import FAISS_RETRIEVAL_K
    from src.data_loader import load_all_candidates
    from src.embeddings import load_artifacts, retrieve_top_k

    k = args.k or FAISS_RETRIEVAL_K

    logger.info("Loading FAISS artifacts …")
    index, jd_emb, cand_embs, cand_ids = load_artifacts()
    logger.info("Index size: %d vectors  JD embedding shape: %s", index.ntotal, jd_emb.shape)

    logger.info("JD embedding norm: %.4f  (should be ~1.0 for L2-normalised)", float((jd_emb**2).sum()**0.5))
    logger.info("JD embedding min/max: %.4f / %.4f", float(jd_emb.min()), float(jd_emb.max()))

    logger.info("Retrieving top %d candidates …", k)
    top_ids, cosine_scores = retrieve_top_k(index, cand_ids, jd_emb, k)

    logger.info("Loading candidate data for lookup …")
    candidates = load_all_candidates(args.candidates)
    cand_by_id = {c["candidate_id"]: c for c in candidates}

    # ── Score distribution ─────────────────────────────────────────────────────
    import numpy as np
    scores_arr = np.array(cosine_scores)
    print("\n─── Cosine Similarity Distribution (top-%d retrieved) ───" % k)
    print(f"  min:    {scores_arr.min():.4f}")
    print(f"  p25:    {float(np.percentile(scores_arr, 25)):.4f}")
    print(f"  median: {float(np.percentile(scores_arr, 50)):.4f}")
    print(f"  p75:    {float(np.percentile(scores_arr, 75)):.4f}")
    print(f"  p90:    {float(np.percentile(scores_arr, 90)):.4f}")
    print(f"  max:    {scores_arr.max():.4f}")
    print(f"  mean:   {scores_arr.mean():.4f}  std: {scores_arr.std():.4f}")

    # ── Coverage check ─────────────────────────────────────────────────────────
    domain_count = 0
    for cid in top_ids:
        c = cand_by_id.get(cid)
        if c and _has_domain_title(c.get("profile", {})):
            domain_count += 1
    coverage_pct = 100.0 * domain_count / len(top_ids)
    print(f"\n─── Domain coverage in top-{k}: {domain_count}/{len(top_ids)} ({coverage_pct:.1f}%) have domain-relevant titles")
    if coverage_pct < 20:
        print("  ⚠️  Low coverage — consider adjusting JD text or retrieval_k")

    # ── Top-N display ──────────────────────────────────────────────────────────
    n = min(args.top, len(top_ids))
    print(f"\n─── Top {n} by Semantic Similarity ───")
    print(f"  {'Rank':<5} {'CandID':<15} {'Score':<7} {'Title':<35} {'Company':<25}")
    print(f"  {'-'*5} {'-'*15} {'-'*7} {'-'*35} {'-'*25}")
    for i, (cid, sc) in enumerate(zip(top_ids[:n], cosine_scores[:n]), 1):
        c = cand_by_id.get(cid, {})
        prof = c.get("profile", {})
        title   = (prof.get("current_title",   "") or "")[:35]
        company = (prof.get("current_company", "") or "")[:25]
        print(f"  {i:<5} {cid:<15} {sc:.4f}  {title:<35} {company:<25}")

    # ── Score cutoff sensitivity ───────────────────────────────────────────────
    for cutoff in [0.4, 0.5, 0.6, 0.7]:
        n_above = (scores_arr >= cutoff).sum()
        print(f"  Candidates with similarity >= {cutoff}: {n_above}")

    print("\n─── Diagnostics complete ───\n")


if __name__ == "__main__":
    main()
