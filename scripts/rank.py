#!/usr/bin/env python3
"""
rank.py — main ranking script. Must run in < 5 minutes on CPU.

Usage:
    python scripts/rank.py --candidates path/to/candidates.jsonl
    python scripts/rank.py --candidates path/to/candidates.jsonl --out submission/my_team.csv

Requires precomputed artifacts in artifacts/ (run precompute.py first).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import SUBMISSION_DIR
from src.data_loader import get_candidate_ids, load_all_candidates
from src.ranking import rank_candidates, write_submission_csv
from src.validation import print_report, validate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Rank candidates for the Redrob challenge")
    p.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: submission/submission.csv)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation on output before finishing (default: True)",
    )
    p.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip post-run validation",
    )
    return p.parse_args()


def main():
    args = parse_args()

    out_path = args.out or str(SUBMISSION_DIR / "submission.csv")

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Redrob ranking pipeline starting …")
    logger.info("Candidates: %s", args.candidates)
    logger.info("Output:     %s", out_path)
    logger.info("=" * 60)

    # ── Rank ───────────────────────────────────────────────────────────────────
    results = rank_candidates(args.candidates)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    write_submission_csv(results, out_path)

    elapsed = time.time() - t0
    logger.info("Total runtime: %.1f s (%.2f min)", elapsed, elapsed / 60)

    if elapsed > 280:
        logger.warning(
            "Runtime %.1f s is close to the 5-minute limit! "
            "Reduce retrieval.retrieval_k in config.yaml if needed.",
            elapsed,
        )

    # ── Validate ───────────────────────────────────────────────────────────────
    if args.validate:
        logger.info("Running validation …")
        dataset_ids = get_candidate_ids(load_all_candidates(args.candidates))
        errors = validate_submission(out_path, dataset_ids=dataset_ids)
        ok = print_report(errors, label="Submission validation")
        if not ok:
            sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Top 5 candidates:")
    for row in results[:5]:
        s = row["_scores"]
        logger.info(
            "  Rank %d  %s  score=%.4f  career=%.2f  sem=%.2f  pnl=%.2f  '%s'",
            row["rank"],
            row["candidate_id"],
            row["score"],
            s["career"],
            s["semantic"],
            s["penalty"],
            row["reasoning"][:80],
        )
    logger.info("-" * 60)
    logger.info("Done. Submission: %s", out_path)


if __name__ == "__main__":
    main()
