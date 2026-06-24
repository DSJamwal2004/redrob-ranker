#!/usr/bin/env python3
"""
precompute.py — one-time offline precomputation (run before rank.py).

What it does:
  1. Loads all candidates from candidates.jsonl
  2. Builds a text representation of each candidate
  3. Encodes all texts with sentence-transformers (batched)
  4. Encodes the JD
  5. Builds a FAISS FlatIP index
  6. Saves index, embeddings, and IDs to artifacts/

Runtime: ~10–25 minutes for 100K candidates on CPU (one-time cost).

Usage:
    python scripts/precompute.py --candidates path/to/candidates.jsonl
    python scripts/precompute.py --candidates path/to/candidates.jsonl --sample 500
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

# ensure src is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidate_text import build_candidate_text
from src.constants import CAND_FEATURES_PATH, JD_TEXT
from src.data_loader import load_all_candidates
from src.embeddings import (
    build_faiss_index,
    encode_jd,
    encode_texts,
    get_model,
    save_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Precompute embeddings and FAISS index")
    p.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl (or sample_candidates.json)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, only process first N candidates (for fast testing)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    t0 = time.time()
    logger.info("Loading candidates from %s …", args.candidates)
    candidates = load_all_candidates(args.candidates)

    if args.sample:
        candidates = candidates[: args.sample]
        logger.info("SAMPLE mode: using first %d candidates", len(candidates))

    logger.info("Total candidates: %d", len(candidates))

    # ── Build candidate texts ───────────────────────────────────────────────────
    logger.info("Building candidate texts …")
    texts   = []
    ids     = []
    for c in candidates:
        texts.append(build_candidate_text(c))
        ids.append(c["candidate_id"])

    # ── Encode JD ──────────────────────────────────────────────────────────────
    from src.constants import REPO_ROOT
    _jd_src = "jd.txt" if (REPO_ROOT / "jd.txt").exists() else "embedded fallback"
    logger.info("Encoding JD (source: %s) …", _jd_src)
    # warm up model
    get_model()
    jd_emb = encode_jd()
    logger.info("JD embedding shape: %s", jd_emb.shape)

    # ── Encode candidates ──────────────────────────────────────────────────────
    logger.info("Encoding %d candidate texts (this is the slow step) …", len(texts))
    t_enc = time.time()
    cand_embs = encode_texts(texts, show_progress=True)
    logger.info("Encoding done in %.1f s  shape: %s", time.time() - t_enc, cand_embs.shape)

    # ── Build FAISS index ──────────────────────────────────────────────────────
    logger.info("Building FAISS index …")
    index = build_faiss_index(cand_embs)

    # ── Save artifacts ──────────────────────────────────────────────────────────
    logger.info("Saving artifacts …")
    save_artifacts(index, jd_emb, cand_embs, ids)

    logger.info("Precomputation complete in %.1f s", time.time() - t0)
    logger.info("Next step: python scripts/rank.py --candidates %s", args.candidates)


if __name__ == "__main__":
    main()
