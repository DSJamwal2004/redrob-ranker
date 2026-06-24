#!/usr/bin/env python3
"""
config_sanity.py — display the active configuration and run sanity checks.

Shows all effective tuning values (from config.yaml or defaults),
validates them, and reports potential issues before a real run.

Usage:
    python scripts/config_sanity.py
    python scripts/config_sanity.py --verbose
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Show active config and run sanity checks")
    p.add_argument("--verbose", action="store_true", help="Show all values including derived")
    args = p.parse_args()

    # Import after sys.path is set
    from src import constants as C

    issues = []
    warnings = []

    repo = C.REPO_ROOT
    config_path = repo / "config.yaml"
    jd_path     = repo / C._get("paths.jd_text_file", "jd.txt")

    print(f"\n{'='*64}")
    print(f"  Redrob Ranker — Active Configuration")
    print(f"{'='*64}")

    # ── Config file ────────────────────────────────────────────────────────────
    print(f"\n─── Config File ────────────────────────────────────────────")
    if config_path.exists():
        print(f"  ✅ config.yaml found: {config_path}")
    else:
        warnings.append("config.yaml not found — all values are built-in defaults")
        print(f"  ⚠️  config.yaml NOT found ({config_path})")
        print(f"       All values are built-in defaults. Create config.yaml to tune.")

    if jd_path.exists():
        jd_len = len(jd_path.read_text(encoding="utf-8").strip())
        print(f"  ✅ jd.txt found: {jd_path}  ({jd_len} chars)")
    else:
        warnings.append(f"jd.txt not found ({jd_path}) — using embedded JD fallback")
        print(f"  ⚠️  jd.txt NOT found — using embedded JD fallback")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\n─── Model ─────────────────────────────────────────────────")
    print(f"  embedding_model:      {C.EMBEDDING_MODEL}")
    print(f"  embedding_batch_size: {C.EMBEDDING_BATCH_SIZE}")
    print(f"  embedding_dim:        {C.EMBEDDING_DIM}")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    print(f"\n─── Retrieval ──────────────────────────────────────────────")
    print(f"  retrieval_k:  {C.FAISS_RETRIEVAL_K}")
    print(f"  final_top_n:  {C.FINAL_TOP_N}")
    if C.FAISS_RETRIEVAL_K < C.FINAL_TOP_N:
        issues.append(f"retrieval_k ({C.FAISS_RETRIEVAL_K}) < final_top_n ({C.FINAL_TOP_N}) — cannot produce {C.FINAL_TOP_N} results")
    if C.FAISS_RETRIEVAL_K < 100:
        warnings.append(f"retrieval_k={C.FAISS_RETRIEVAL_K} is very low — may miss strong candidates")

    # ── Scoring weights ────────────────────────────────────────────────────────
    print(f"\n─── Scoring Weights ────────────────────────────────────────")
    wsum = C.W_CAREER + C.W_SKILLS + C.W_SEMANTIC + C.W_LOCATION + C.W_BEHAVIOR
    print(f"  career   (target 0.50): {C.W_CAREER:.4f}")
    print(f"  skills   (target 0.20): {C.W_SKILLS:.4f}")
    print(f"  semantic (target 0.15): {C.W_SEMANTIC:.4f}")
    print(f"  location (target 0.05): {C.W_LOCATION:.4f}")
    print(f"  behavior (target 0.10): {C.W_BEHAVIOR:.4f}")
    print(f"  ──────────────────────────────")
    print(f"  SUM:                    {wsum:.4f}  {'✅' if abs(wsum-1.0)<1e-4 else '❌'}")
    if abs(wsum - 1.0) > 1e-4:
        issues.append(f"Weights sum to {wsum:.6f}, not 1.0")
    if C.W_SEMANTIC > 0.25:
        warnings.append(f"semantic weight {C.W_SEMANTIC} is high — it should be a support signal, not dominant")
    if C.W_CAREER < 0.40:
        warnings.append(f"career weight {C.W_CAREER} is low — career evidence should dominate")

    # ── Company heuristic ──────────────────────────────────────────────────────
    print(f"\n─── Company Heuristic ──────────────────────────────────────")
    print(f"  consulting_score: {C.COMPANY_CONSULTING_SCORE:.3f}")
    print(f"  product_score:    {C.COMPANY_PRODUCT_SCORE:.3f}")
    ratio = C.COMPANY_PRODUCT_SCORE / C.COMPANY_CONSULTING_SCORE if C.COMPANY_CONSULTING_SCORE > 0 else 99
    print(f"  ratio (product/consulting): {ratio:.2f}x")
    if ratio > 3.0:
        warnings.append(f"Company score ratio {ratio:.1f}x is large — may act as a prestige filter")

    # ── Honeypot thresholds ────────────────────────────────────────────────────
    print(f"\n─── Honeypot Thresholds ────────────────────────────────────")
    print(f"  yoe_mismatch_threshold_years: {C.HP_YOE_MISMATCH_THRESHOLD}")
    print(f"  yoe_mismatch_penalty:         {C.HP_YOE_MISMATCH_PENALTY}")
    print(f"  expert_zero_duration_limit:   {C.HP_EXPERT_ZERO_DUR_LIMIT}")
    print(f"  expert_zero_duration_penalty: {C.HP_EXPERT_ZERO_DUR_PENALTY}")
    print(f"  expert_zero_endorse_limit:    {C.HP_EXPERT_ZERO_END_LIMIT}")
    print(f"  expert_zero_endorse_penalty:  {C.HP_EXPERT_ZERO_END_PENALTY}")
    print(f"  current_role_yoe_slack_years: {C.HP_CURRENT_ROLE_SLACK}")
    print(f"  current_role_yoe_penalty:     {C.HP_CURRENT_ROLE_PENALTY}")
    print(f"  min_penalty_multiplier:       {C.HP_MIN_MULTIPLIER}")
    if C.HP_MIN_MULTIPLIER < 0.15:
        warnings.append("min_penalty_multiplier is very low — aggressive false positive risk")

    # ── Behavioral thresholds ──────────────────────────────────────────────────
    print(f"\n─── Behavioral Thresholds ──────────────────────────────────")
    print(f"  active_days:           {C.BEH_ACTIVE_DAYS}d")
    print(f"  recent_days:           {C.BEH_RECENT_DAYS}d")
    print(f"  moderate_days:         {C.BEH_MODERATE_DAYS}d")
    print(f"  stale_days:            {C.BEH_STALE_DAYS}d")
    print(f"  inactivity_concern:    {C.BEH_INACTIVITY_CONCERN}d")

    # ── Locations ──────────────────────────────────────────────────────────────
    print(f"\n─── Preferred Locations ────────────────────────────────────")
    print(f"  preferred: {sorted(C.PREFERRED_CITIES)}")
    print(f"  ok:        {sorted(C.OK_CITIES)}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    print(f"\n─── Artifact Paths ─────────────────────────────────────────")
    print(f"  artifacts_dir:  {C.ARTIFACTS_DIR}")
    print(f"  submission_dir: {C.SUBMISSION_DIR}")
    for name, path in [
        ("faiss_index", C.FAISS_INDEX_PATH),
        ("jd_embedding", C.JD_EMBEDDING_PATH),
        ("cand_embeddings", C.CAND_EMBEDDINGS_PATH),
        ("cand_ids", C.CAND_IDS_PATH),
    ]:
        exists = "✅" if path.exists() else "⬜ (not built yet)"
        print(f"  {name:<18}: {exists}  {path}")

    # ── Score precision ────────────────────────────────────────────────────────
    print(f"\n─── Output ─────────────────────────────────────────────────")
    print(f"  score_precision: {C.SCORE_PRECISION} decimal places")
    print(f"  final_top_n:     {C.FINAL_TOP_N}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    if issues:
        print(f"  ❌ ISSUES ({len(issues)}) — must fix before submission:")
        for iss in issues:
            print(f"     • {iss}")
    if warnings:
        print(f"  ⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"     • {w}")
    if not issues and not warnings:
        print(f"  ✅ All checks passed — configuration looks healthy")
    print(f"{'='*64}\n")

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
