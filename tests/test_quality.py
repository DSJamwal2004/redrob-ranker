"""
test_quality.py — quality sanity checks on scoring logic.

These tests don't require a real dataset; they verify that the scoring
functions behave sensibly on hand-crafted examples.

Run:  pytest tests/test_quality.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_candidate(
    yoe=7.0,
    title="Senior ML Engineer",
    company="ProductCo",
    industry="Technology",
    company_size="201-500",
    description="Built and shipped ranking system. Deployed embedding-based retrieval for recommendation.",
    skills=None,
    open_to_work=True,
    country="India",
    location="Bangalore",
    notice_days=30,
    last_active="2026-06-01",
    rr=0.8,
    career_start="2022-01-01",
    verified_email=True,
    verified_phone=True,
):
    if skills is None:
        skills = [
            {"name": "NLP", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
            {"name": "Ranking", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
        ]
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Senior ML Engineer | NLP, Ranking, Search",
            "summary": "7 years building ranking and retrieval systems.",
            "location": location,
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": company_size,
            "current_industry": industry,
        },
        "career_history": [
            {
                "company": company,
                "title": title,
                "start_date": career_start,
                "end_date": None,
                "duration_months": 28,
                "is_current": True,
                "industry": industry,
                "company_size": company_size,
                "description": description,
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
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": "2020-01-01",
            "last_active_date": last_active,
            "open_to_work_flag": open_to_work,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": rr,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 300,
            "endorsements_received": 90,
            "notice_period_days": notice_days,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.9,
            "verified_email": verified_email,
            "verified_phone": verified_phone,
            "linkedin_connected": True,
        },
    }


# ─── career_fit_score tests ───────────────────────────────────────────────────

def test_strong_career_scores_high():
    from src.scoring import career_fit_score
    c = _make_candidate(
        description="Shipped ranking system for 5M users. Deployed embedding retrieval. "
                    "Built recommendation engine. Implemented NLP pipeline for search.",
        yoe=7.0,
    )
    score = career_fit_score(c)
    assert score > 0.55, f"Strong candidate should score > 0.55, got {score:.3f}"
    print(f"✅  strong career: {score:.3f}")


def test_pure_consulting_scores_lower():
    from src.scoring import career_fit_score
    c = _make_candidate(
        company="Infosys",
        industry="IT Services",
        description="Worked on client projects. Managed stakeholders. Delivered reports.",
    )
    c2 = _make_candidate(
        company="ProductStartup",
        industry="Technology",
        description="Shipped ranking system. Deployed retrieval. Built search pipeline.",
    )
    s_consulting = career_fit_score(c)
    s_product    = career_fit_score(c2)
    assert s_consulting < s_product, (
        f"Consulting ({s_consulting:.3f}) should score < Product ({s_product:.3f})"
    )
    print(f"✅  consulting vs product: {s_consulting:.3f} < {s_product:.3f}")


def test_wrong_yoe_penalised():
    from src.scoring import career_fit_score
    c_sweet = _make_candidate(yoe=7.0)
    c_junior = _make_candidate(yoe=1.5)
    c_senior = _make_candidate(yoe=20.0)
    s_sweet  = career_fit_score(c_sweet)
    s_junior = career_fit_score(c_junior)
    s_senior = career_fit_score(c_senior)
    assert s_sweet > s_junior, f"Sweet-spot YoE should beat junior: {s_sweet:.3f} vs {s_junior:.3f}"
    print(f"✅  YoE seniority: sweet={s_sweet:.3f}  junior={s_junior:.3f}  senior={s_senior:.3f}")


# ─── skill_relevance_score tests ──────────────────────────────────────────────

def test_deep_skills_beat_shallow():
    from src.scoring import skill_relevance_score

    deep = _make_candidate(skills=[
        {"name": "NLP", "proficiency": "expert", "endorsements": 45, "duration_months": 60},
        {"name": "Ranking", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
    ])
    shallow = _make_candidate(skills=[
        {"name": "NLP", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
        {"name": "Ranking", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
        {"name": "LLM", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
        {"name": "Embedding", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
        {"name": "RAG", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
    ])
    s_deep    = skill_relevance_score(deep)
    s_shallow = skill_relevance_score(shallow)
    assert s_deep > s_shallow, f"Deep skills ({s_deep:.3f}) should beat shallow ({s_shallow:.3f})"
    print(f"✅  skill depth: deep={s_deep:.3f}  shallow={s_shallow:.3f}")


def test_keyword_stuffer_penalised():
    from src.scoring import skill_relevance_score
    # 30 "expert" AI skills with no endorsements and no duration
    stuffer = _make_candidate(skills=[
        {"name": kw, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for kw in ["NLP", "ML", "LLM", "RAG", "Embedding", "Ranking", "Retrieval",
                   "Transformer", "BERT", "GPT", "Vector", "Faiss", "Pinecone",
                   "Weaviate", "Qdrant", "Elasticsearch", "OpenSearch", "NDCG",
                   "MRR", "Fine-tuning", "Sentence-Transformer", "HuggingFace",
                   "PyTorch", "Evaluation", "Information Retrieval", "Search",
                   "Recommendation", "Recommender", "Reranker", "Re-ranking"]
    ])
    legit = _make_candidate(skills=[
        {"name": "NLP", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
        {"name": "Ranking", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
    ])
    s_stuffer = skill_relevance_score(stuffer)
    s_legit   = skill_relevance_score(legit)
    assert s_stuffer < s_legit, (
        f"Keyword stuffer ({s_stuffer:.3f}) should score < legitimate ({s_legit:.3f})"
    )
    print(f"✅  keyword stuffer: stuffer={s_stuffer:.3f}  legit={s_legit:.3f}")


# ─── location_score tests ──────────────────────────────────────────────────────

def test_india_preferred_over_abroad():
    from src.scoring import location_score
    india = _make_candidate(country="India", location="Pune", notice_days=30)
    abroad = _make_candidate(country="USA", location="San Francisco", notice_days=30)
    abroad["redrob_signals"]["willing_to_relocate"] = False
    s_india  = location_score(india)
    s_abroad = location_score(abroad)
    assert s_india > s_abroad, f"India ({s_india:.3f}) should score > abroad no-relocate ({s_abroad:.3f})"
    print(f"✅  location India vs abroad: {s_india:.3f} > {s_abroad:.3f}")


# ─── behavioral_score tests ────────────────────────────────────────────────────

def test_active_open_beats_inactive_closed():
    from src.scoring import behavioral_score
    active = _make_candidate(open_to_work=True, last_active="2026-06-01", rr=0.9)
    inactive = _make_candidate(open_to_work=False, last_active="2024-01-01", rr=0.1)
    s_active   = behavioral_score(active)
    s_inactive = behavioral_score(inactive)
    assert s_active > s_inactive, (
        f"Active ({s_active:.3f}) should beat inactive ({s_inactive:.3f})"
    )
    print(f"✅  behavioral: active={s_active:.3f}  inactive={s_inactive:.3f}")


# ─── consistency_penalty tests ────────────────────────────────────────────────

def test_clean_profile_no_penalty():
    from src.scoring import consistency_penalty
    c = _make_candidate(yoe=7.0, career_start="2019-01-01")
    p = consistency_penalty(c)
    assert p > 0.85, f"Clean profile should have penalty > 0.85, got {p:.3f}"
    print(f"✅  clean profile penalty={p:.3f}")


def test_yoe_mismatch_penalised():
    from src.scoring import consistency_penalty
    c = _make_candidate(yoe=2.0, career_start="2012-01-01")  # claims 2yr, but 14yr career
    p = consistency_penalty(c)
    assert p < 0.7, f"YoE mismatch should give penalty < 0.7, got {p:.3f}"
    print(f"✅  YoE mismatch penalty={p:.3f}")


if __name__ == "__main__":
    test_strong_career_scores_high()
    test_pure_consulting_scores_lower()
    test_wrong_yoe_penalised()
    test_deep_skills_beat_shallow()
    test_keyword_stuffer_penalised()
    test_india_preferred_over_abroad()
    test_active_open_beats_inactive_closed()
    test_clean_profile_no_penalty()
    test_yoe_mismatch_penalised()
    print("\n✅  All quality tests passed")
