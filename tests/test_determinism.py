"""
test_determinism.py — verify that two ranking runs produce identical output.

Run with:  pytest tests/test_determinism.py -v
Or:        python tests/test_determinism.py (small sample, fast)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Helpers ─────────────────────────────────────────────────────────────────

SAMPLE_CANDIDATES_PATH = (
    Path(__file__).parent.parent
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "sample_candidates.json"
)
# Fallback: user may have placed it differently
_FALLBACKS = [
    Path("sample_candidates.json"),
    Path("data/sample_candidates.json"),
]


def _find_sample_file():
    if SAMPLE_CANDIDATES_PATH.exists():
        return SAMPLE_CANDIDATES_PATH
    for f in _FALLBACKS:
        if f.exists():
            return f
    return None


def _make_minimal_candidates(n=10) -> list:
    """Generate minimal valid candidates for unit tests (no real file needed)."""
    from datetime import date
    cands = []
    for i in range(1, n + 1):
        cands.append({
            "candidate_id": f"CAND_{i:07d}",
            "profile": {
                "anonymized_name": f"Test Candidate {i}",
                "headline": "ML Engineer | NLP, Ranking, Embeddings",
                "summary": f"7 years building ranking and retrieval systems. shipped search {i}.",
                "location": "Bangalore",
                "country": "India",
                "years_of_experience": 6.0 + i * 0.1,
                "current_title": "Senior ML Engineer",
                "current_company": "ProductCo",
                "current_company_size": "201-500",
                "current_industry": "Technology",
            },
            "career_history": [
                {
                    "company": "ProductCo",
                    "title": "Senior ML Engineer",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "duration_months": 28,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "201-500",
                    "description": "Built and shipped ranking system, deployed retrieval pipeline, "
                                   "implemented embedding-based search for recommendation engine.",
                }
            ],
            "education": [
                {
                    "institution": "IIT Bombay",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2012,
                    "end_year": 2016,
                    "grade": "8.5",
                    "tier": "tier_1",
                }
            ],
            "skills": [
                {"name": "NLP", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
                {"name": "Ranking", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
                {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 72},
            ],
            "certifications": [],
            "languages": [{"language": "English", "proficiency": "native"}],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2020-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 50,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.75,
                "avg_response_time_hours": 12.0,
                "skill_assessment_scores": {"Python": 88},
                "connection_count": 300,
                "endorsements_received": 90,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 70,
                "search_appearance_30d": 10,
                "saved_by_recruiters_30d": 3,
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": 0.9,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        })
    return cands


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_scoring_determinism():
    """Same candidate → same score on two calls."""
    from src.scoring import compute_all_scores

    cands = _make_minimal_candidates(3)
    c = cands[0]
    sem = 0.65

    s1 = compute_all_scores(c, sem)
    s2 = compute_all_scores(c, sem)

    assert s1 == s2, f"Scores differ:\n  run1={s1}\n  run2={s2}"
    print("✅  test_scoring_determinism passed")


def test_reasoning_determinism():
    """Same candidate → same reasoning string on two calls."""
    from src.reasoning import generate_reasoning
    from src.scoring import compute_all_scores

    cands = _make_minimal_candidates(3)
    c = cands[0]
    scores = compute_all_scores(c, 0.65)

    r1 = generate_reasoning(c, scores)
    r2 = generate_reasoning(c, scores)

    assert r1 == r2, f"Reasoning differs:\n  r1={r1!r}\n  r2={r2!r}"
    print(f"✅  test_reasoning_determinism passed: {r1!r}")


def test_score_bounds():
    """All score components must be in [0, 1]."""
    from src.scoring import compute_all_scores

    cands = _make_minimal_candidates(5)
    for c in cands:
        s = compute_all_scores(c, 0.5)
        for k, v in s.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of bounds for {c['candidate_id']}"
    print("✅  test_score_bounds passed")


def test_reasoning_length():
    """Reasoning must be ≤ 200 chars."""
    from src.reasoning import generate_reasoning
    from src.scoring import compute_all_scores

    cands = _make_minimal_candidates(5)
    for c in cands:
        scores = compute_all_scores(c, 0.5)
        r = generate_reasoning(c, scores)
        assert len(r) <= 200, f"Reasoning too long ({len(r)} chars): {r!r}"
    print("✅  test_reasoning_length passed")


def test_consistency_penalty_catches_honeypot():
    """A candidate with impossible profile should get a penalty < 0.7."""
    from src.scoring import consistency_penalty

    honeypot = _make_minimal_candidates(1)[0]
    # Make YoE mismatch: claim 2 years but career started 10 years ago
    honeypot["profile"]["years_of_experience"] = 2.0
    honeypot["career_history"][0]["start_date"] = "2014-01-01"
    # Add expert skills with zero duration
    honeypot["skills"] = [
        {"name": "Ranking", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ]

    p = consistency_penalty(honeypot)
    assert p < 0.7, f"Honeypot penalty should be < 0.7, got {p}"
    print(f"✅  test_consistency_penalty_catches_honeypot passed  penalty={p:.2f}")


def test_candidate_text_non_empty():
    """build_candidate_text must return a non-empty string."""
    from src.candidate_text import build_candidate_text

    cands = _make_minimal_candidates(5)
    for c in cands:
        t = build_candidate_text(c)
        assert t and len(t) > 10, f"Empty text for {c['candidate_id']}"
    print("✅  test_candidate_text_non_empty passed")


def test_full_ranking_pipeline_small():
    """
    End-to-end ranking on a small synthetic dataset.
    Skips if artifacts are not built (CI-safe).
    """
    import os
    import tempfile
    import json
    from pathlib import Path

    from src.constants import FAISS_INDEX_PATH
    if not FAISS_INDEX_PATH.exists():
        print("⚠️   test_full_ranking_pipeline_small SKIPPED (no artifacts; run precompute.py first)")
        return

    from src.ranking import rank_candidates, write_submission_csv
    from src.validation import validate_submission

    cands = _make_minimal_candidates(50)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(cands, fh)
        cand_path = fh.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        out_path = fh.name

    try:
        # This test works only if artifacts cover at least 50 candidates
        results = rank_candidates(cand_path)
        assert len(results) == min(100, 50)
        write_submission_csv(results, out_path)
        errs = validate_submission(out_path)
        assert not errs, f"Validation errors: {errs}"
        print("✅  test_full_ranking_pipeline_small passed")
    finally:
        os.unlink(cand_path)
        os.unlink(out_path)


if __name__ == "__main__":
    test_scoring_determinism()
    test_reasoning_determinism()
    test_score_bounds()
    test_reasoning_length()
    test_consistency_penalty_catches_honeypot()
    test_candidate_text_non_empty()
    test_full_ranking_pipeline_small()
    print("\n✅  All determinism tests passed")
