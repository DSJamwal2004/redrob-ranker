#!/usr/bin/env python3
"""
top100_summary.py — sanity check and summary of the top-100 submission.

Prints:
  - Score spread and distribution
  - Career fit, skill, semantic, behavior stats
  - India / non-India breakdown
  - Notice period distribution
  - Consulting vs product breakdown
  - Reasoning length / diversity stats
  - Red flags for manual review

Usage:
    python scripts/top100_summary.py --submission submission/submission.csv \
                                     --candidates path/to/candidates.jsonl
"""

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Top-100 submission sanity summary")
    p.add_argument("--submission", required=True, help="Path to submission CSV")
    p.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    return p.parse_args()


def _pct(n, total):
    return 100.0 * n / total if total else 0.0


def main():
    args = parse_args()

    from src.data_loader import load_all_candidates, safe_career, safe_profile, safe_signals, safe_skills
    from src.scoring import compute_all_scores
    from src.constants import CONSULTING_FIRMS

    # ── Load submission ────────────────────────────────────────────────────────
    rows = []
    with open(args.submission, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    print(f"\n{'='*64}")
    print(f"  Top-100 Submission Summary: {args.submission}")
    print(f"  {len(rows)} rows loaded")
    print(f"{'='*64}")

    scores = [float(r["score"]) for r in rows]
    print(f"\n─── Score Distribution ───────────────────────────────────")
    print(f"  Rank  1 score : {scores[0]:.4f}")
    print(f"  Rank 10 score : {scores[9]:.4f}" if len(scores) >= 10 else "")
    print(f"  Rank 25 score : {scores[24]:.4f}" if len(scores) >= 25 else "")
    print(f"  Rank 50 score : {scores[49]:.4f}" if len(scores) >= 50 else "")
    print(f"  Rank100 score : {scores[-1]:.4f}")
    spread = scores[0] - scores[-1]
    print(f"  Score spread  : {spread:.4f}")
    if spread < 0.05:
        print("  ⚠️  Very tight score spread — may indicate weak differentiation")

    # Check non-increasing
    violations = sum(1 for i in range(len(scores)-1) if scores[i] < scores[i+1])
    if violations:
        print(f"  ❌ Non-increasing violations: {violations}")
    else:
        print("  ✅ Scores are non-increasing")

    # ── Load candidates + re-score ─────────────────────────────────────────────
    print(f"\n─── Per-candidate breakdown ───────────────────────────────")
    candidates = load_all_candidates(args.candidates)
    cand_by_id = {c["candidate_id"]: c for c in candidates}

    sub_ids = [r["candidate_id"] for r in rows]
    missing = [cid for cid in sub_ids if cid not in cand_by_id]
    if missing:
        print(f"  ⚠️  {len(missing)} submission IDs not found in dataset: {missing[:5]}")

    # Aggregate stats
    career_scores, skill_scores, sem_scores, beh_scores = [], [], [], []
    n_india, n_preferred, n_consulting_heavy = 0, 0, 0
    notice_buckets = Counter()
    reasoning_lengths = [len(r["reasoning"]) for r in rows]

    for row in rows:
        cid = row["candidate_id"]
        c = cand_by_id.get(cid)
        if not c:
            continue

        # Use stored semantic from submission score — approximate via re-scoring with 0 semantic
        # (We don't have the original FAISS scores here, so we just show structured scores)
        s = compute_all_scores(c, semantic_sim=0.0)  # semantic excluded for breakdown
        career_scores.append(s["career"])
        skill_scores.append(s["skills"])
        beh_scores.append(s["behavior"])

        prof = safe_profile(c)
        sig  = safe_signals(c)
        career = safe_career(c)

        if (prof.get("country") or "").lower() == "india":
            n_india += 1
            loc = (prof.get("location") or "").lower()
            if any(city in loc for city in ["pune", "noida"]):
                n_preferred += 1

        notice = int(sig.get("notice_period_days") or 60)
        if notice <= 30:
            notice_buckets["≤30d"] += 1
        elif notice <= 60:
            notice_buckets["31-60d"] += 1
        elif notice <= 90:
            notice_buckets["61-90d"] += 1
        else:
            notice_buckets[">90d"] += 1

        # Consulting heavy = >50% career at known consulting firms
        months_consulting = sum(
            max(int(r.get("duration_months") or 1), 1)
            for r in career
            if any(firm in (r.get("company") or "").lower() for firm in CONSULTING_FIRMS)
        )
        total_months = sum(max(int(r.get("duration_months") or 1), 1) for r in career)
        if total_months > 0 and months_consulting / total_months > 0.5:
            n_consulting_heavy += 1

    n = len(rows)
    def _avg(lst):
        return sum(lst)/len(lst) if lst else 0.0

    print(f"\n─── Score Component Averages (top 100) ─────────────────────")
    print(f"  Career (w=0.50): avg={_avg(career_scores):.3f}  min={min(career_scores):.3f}  max={max(career_scores):.3f}")
    print(f"  Skills (w=0.20): avg={_avg(skill_scores):.3f}  min={min(skill_scores):.3f}  max={max(skill_scores):.3f}")
    print(f"  Behav  (w=0.10): avg={_avg(beh_scores):.3f}  min={min(beh_scores):.3f}  max={max(beh_scores):.3f}")
    print(f"  (semantic not available without FAISS artifacts at display time)")

    print(f"\n─── Geography ──────────────────────────────────────────────")
    print(f"  India:          {n_india}/{n}  ({_pct(n_india, n):.0f}%)")
    print(f"  Pune/Noida:     {n_preferred}/{n}  ({_pct(n_preferred, n):.0f}%)")
    if n_india < 70:
        print("  ⚠️  Less than 70% India — check location scoring")

    print(f"\n─── Notice Period ──────────────────────────────────────────")
    for bucket, cnt in sorted(notice_buckets.items()):
        print(f"  {bucket:<10}: {cnt}")

    print(f"\n─── Company Background ─────────────────────────────────────")
    print(f"  Consulting-heavy (>50% tenure): {n_consulting_heavy}/{n}")
    if n_consulting_heavy > 20:
        print("  ⚠️  High consulting concentration — check company scoring")

    print(f"\n─── Reasoning Quality ──────────────────────────────────────")
    print(f"  Avg length: {_avg(reasoning_lengths):.0f} chars  (limit: 200)")
    print(f"  Max length: {max(reasoning_lengths)}  Min: {min(reasoning_lengths)}")
    n_too_long = sum(1 for l in reasoning_lengths if l > 200)
    if n_too_long:
        print(f"  ❌ {n_too_long} reasoning strings exceed 200 chars")
    else:
        print(f"  ✅ All reasoning within limit")
    unique_r = len(set(r["reasoning"] for r in rows))
    print(f"  Unique reasoning: {unique_r}/{n}  ({_pct(unique_r, n):.0f}%)")
    if unique_r < n * 0.85:
        print("  ⚠️  Low reasoning diversity — possible template collision")

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
