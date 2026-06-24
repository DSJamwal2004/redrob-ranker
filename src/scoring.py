"""
scoring.py — all structured scoring components.

Each function returns a float in [0, 1].
final_score() assembles the weighted composite.

Weights (from config.yaml → constants.py):
  career   0.50  — dominant signal
  skills   0.20
  semantic 0.15  — retrieval support, not dominant
  behavior 0.10  — availability matters materially
  location 0.05
  × consistency penalty (HP_MIN_MULTIPLIER – 1.0)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from .constants import (
    AI_SKILL_KEYWORDS,
    BEH_ACTIVE_DAYS,
    BEH_INACTIVITY_CONCERN,
    BEH_MODERATE_DAYS,
    BEH_RECENT_DAYS,
    BEH_STALE_DAYS,
    COMPANY_CONSULTING_SCORE,
    COMPANY_PRODUCT_SCORE,
    CONSULTING_FIRMS,
    DOMAIN_TERMS,
    HP_CURRENT_ROLE_PENALTY,
    HP_CURRENT_ROLE_SLACK,
    HP_EXPERT_ZERO_DUR_LIMIT,
    HP_EXPERT_ZERO_DUR_PENALTY,
    HP_EXPERT_ZERO_END_LIMIT,
    HP_EXPERT_ZERO_END_PENALTY,
    HP_MIN_MULTIPLIER,
    HP_YOE_MISMATCH_PENALTY,
    HP_YOE_MISMATCH_THRESHOLD,
    OK_CITIES,
    PREFERRED_CITIES,
    SHIPPING_VERBS,
    W_BEHAVIOR,
    W_CAREER,
    W_LOCATION,
    W_SEMANTIC,
    W_SKILLS,
)
from .data_loader import safe_career, safe_profile, safe_signals, safe_skills

logger = logging.getLogger(__name__)

NOW = datetime.now(timezone.utc)


def _days_ago(date_str: Optional[str]) -> Optional[float]:
    """Return days since date_str (ISO 8601), or None on failure."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((NOW - dt).days, 0.0)
    except (ValueError, TypeError):
        return None


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CAREER FIT  (weight W_CAREER = 0.50)
# ═══════════════════════════════════════════════════════════════════════════════

def career_fit_score(candidate: dict) -> float:
    """
    Sub-components:
      - shipping_evidence (0–1): action verbs + domain terms in role descriptions
      - domain_density    (0–1): saturation of ranking/retrieval terminology
      - recency           (0–1): how recently was relevant work done
      - seniority         (0–1): 5–9 yrs = sweet spot
      - company_type      (0–1): product > mixed > consulting (light heuristic)
    """
    profile = safe_profile(candidate)
    career  = safe_career(candidate)

    if not career:
        return 0.1

    yoe = float(profile.get("years_of_experience", 0) or 0)

    # ── Shipping evidence ──────────────────────────────────────────────────────
    shipping_score = 0.0
    domain_score   = 0.0
    best_recency   = 0.0
    relevant_role_found = False

    for role in career:
        desc  = (role.get("description") or "").lower()
        title = (role.get("title")       or "").lower()

        verb_hits   = sum(desc.count(v) for v in SHIPPING_VERBS)
        domain_hits = sum(1 for t in DOMAIN_TERMS if t in desc or t in title)

        # saturate: 3 verb hits = full, 3 domain hits = full
        role_ship   = _clip(verb_hits   / 3.0)
        role_domain = _clip(domain_hits / 3.0)

        if role_domain > 0.2:       # role is at least partially relevant
            relevant_role_found = True
            shipping_score = max(shipping_score, role_ship)
            domain_score   = max(domain_score,   role_domain)

            # recency: how long ago did this role start?
            days = _days_ago(role.get("start_date"))
            if days is not None:
                if days <= 548:     # ~18 months
                    recency_val = 1.0
                elif days <= 1095:  # ~3 years
                    recency_val = 0.7
                elif days <= 1825:  # ~5 years
                    recency_val = 0.5
                else:
                    recency_val = 0.3
                best_recency = max(best_recency, recency_val)

    if not relevant_role_found:
        # Weakly score adjacent/transition candidates
        for role in career:
            desc = (role.get("description") or "").lower()
            verb_hits = sum(desc.count(v) for v in SHIPPING_VERBS)
            shipping_score = max(shipping_score, _clip(verb_hits / 4.0) * 0.5)
        best_recency = 0.3

    # ── Seniority alignment ────────────────────────────────────────────────────
    if 5.0 <= yoe <= 9.0:
        seniority = 1.0
    elif yoe < 5.0:
        seniority = _clip(0.5 + (yoe / 5.0) * 0.5)
    else:
        seniority = _clip(1.0 - (yoe - 9.0) * 0.05)   # mild penalty past 9

    # ── Company type (light heuristic, not prestige filter) ───────────────────
    company_type = _company_type_score(career)

    # ── Weighted sub-component sum ─────────────────────────────────────────────
    score = (
        shipping_score * 0.35
        + domain_score   * 0.25
        + best_recency   * 0.20
        + seniority      * 0.12
        + company_type   * 0.08
    )
    return _clip(score)


def _company_type_score(career: list) -> float:
    """
    Classify each company as consulting or product using configurable scores.
    Final = duration-weighted average. Consulting is a soft discount, not a hard filter.
    """
    total_months   = 0
    weighted_score = 0.0

    for role in career:
        company  = (role.get("company")  or "").lower().strip()
        industry = (role.get("industry") or "").lower()
        months   = max(int(role.get("duration_months") or 1), 1)

        is_consulting  = any(firm in company for firm in CONSULTING_FIRMS)
        is_it_services = "it service" in industry or "consulting" in industry

        role_score = COMPANY_CONSULTING_SCORE if (is_consulting or is_it_services) \
                     else COMPANY_PRODUCT_SCORE

        weighted_score += role_score * months
        total_months   += months

    if total_months == 0:
        return 0.5
    return _clip(weighted_score / total_months)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SKILL RELEVANCE  (weight W_SKILLS = 0.20)
# ═══════════════════════════════════════════════════════════════════════════════

def skill_relevance_score(candidate: dict) -> float:
    """
    Prioritises depth (proficiency × endorsements × duration) over raw count.
    Penalises keyword stuffing.
    """
    skills = safe_skills(candidate)
    if not skills:
        return 0.05

    PROF_MAP = {"beginner": 0.20, "intermediate": 0.55, "advanced": 0.80, "expert": 1.0}

    depth_scores = []
    n_expert_zero_duration = 0
    total_skills = len(skills)

    for skill in skills:
        name = (skill.get("name") or "").lower()
        if not any(kw in name for kw in AI_SKILL_KEYWORDS):
            continue

        prof_raw = (skill.get("proficiency") or "beginner").lower()
        prof     = PROF_MAP.get(prof_raw, 0.3)

        endorsements    = float(skill.get("endorsements")    or 0)
        duration_months = float(skill.get("duration_months") or 0)

        endorse_boost  = _clip(endorsements    / 15.0)   # 15 endorsements saturates
        duration_boost = _clip(duration_months / 24.0)   # 2 years saturates

        depth = prof * 0.55 + endorse_boost * 0.25 + duration_boost * 0.20
        depth_scores.append(_clip(depth))

        if prof_raw == "expert" and duration_months == 0:
            n_expert_zero_duration += 1

    if not depth_scores:
        return 0.10

    avg_depth = sum(depth_scores) / len(depth_scores)

    # Keyword stuffing penalty
    penalty = 1.0
    if total_skills > 25:
        penalty *= 0.7
    if n_expert_zero_duration > 2:
        penalty *= 0.6
    if len(depth_scores) > 10 and avg_depth < 0.4:
        penalty *= 0.7

    return _clip(avg_depth * penalty)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BEHAVIORAL SIGNALS  (weight W_BEHAVIOR = 0.10)
# ═══════════════════════════════════════════════════════════════════════════════

def behavioral_score(candidate: dict) -> float:
    """
    Availability + engagement + credibility composite.
    At 0.10 weight this materially affects final rank for close candidates.
    """
    sig = safe_signals(candidate)
    if not sig:
        return 0.3

    # ── Availability ───────────────────────────────────────────────────────────
    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.4

    days_inactive = _days_ago(sig.get("last_active_date"))
    if days_inactive is None:
        recency = 0.5
    elif days_inactive <= BEH_ACTIVE_DAYS:
        recency = 1.0
    elif days_inactive <= BEH_RECENT_DAYS:
        recency = 0.85
    elif days_inactive <= BEH_MODERATE_DAYS:
        recency = 0.65
    elif days_inactive <= BEH_STALE_DAYS:
        recency = 0.40
    else:
        recency = 0.15

    availability = open_to_work * 0.50 + recency * 0.50

    # ── Engagement ─────────────────────────────────────────────────────────────
    rr = _clip(float(sig.get("recruiter_response_rate") or 0.0))

    interview_cr = sig.get("interview_completion_rate", -1)
    icr = _clip(float(interview_cr)) if interview_cr != -1 else 0.5

    engagement = rr * 0.70 + icr * 0.30

    # ── Credibility ────────────────────────────────────────────────────────────
    verified_email = 1.0 if sig.get("verified_email") else 0.5
    verified_phone = 1.0 if sig.get("verified_phone") else 0.5
    connections    = _clip(float(sig.get("connection_count") or 0) / 150.0)
    github         = sig.get("github_activity_score", -1)
    github_score   = _clip(float(github) / 100.0) if github != -1 else 0.0

    credibility = (
        verified_email * 0.30
        + verified_phone * 0.20
        + connections    * 0.25
        + github_score   * 0.25
    )

    score = availability * 0.50 + engagement * 0.30 + credibility * 0.20
    return _clip(score)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOCATION & LOGISTICS  (weight W_LOCATION = 0.05)
# ═══════════════════════════════════════════════════════════════════════════════

def location_score(candidate: dict) -> float:
    """
    Prefers India (Pune/Noida best), then other India,
    then relocating international, then non-India no-relocate.
    """
    profile = safe_profile(candidate)
    sig     = safe_signals(candidate)

    country  = (profile.get("country")  or "").lower().strip()
    location = (profile.get("location") or "").lower().strip()
    relocate = bool(sig.get("willing_to_relocate"))

    loc_tokens = set(location.replace(",", " ").split())

    if country == "india":
        if loc_tokens & PREFERRED_CITIES:
            loc_score = 1.0
        elif loc_tokens & OK_CITIES:
            loc_score = 0.90
        else:
            loc_score = 0.80
    elif relocate:
        loc_score = 0.65
    else:
        loc_score = 0.30

    # Notice period
    notice = int(sig.get("notice_period_days") or 60)
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.95
    elif notice <= 60:
        notice_score = 0.80
    elif notice <= 90:
        notice_score = 0.60
    elif notice <= 120:
        notice_score = 0.40
    else:
        notice_score = 0.20

    return _clip(loc_score * 0.65 + notice_score * 0.35)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONSISTENCY PENALTY  (honeypot detection, multiplicative HP_MIN_MULTIPLIER–1.0)
# ═══════════════════════════════════════════════════════════════════════════════

def consistency_penalty(candidate: dict) -> float:
    """
    Detects generic inconsistencies signalling honeypot / junk profiles.
    Returns a multiplier in [HP_MIN_MULTIPLIER, 1.0]. Applied to raw score.

    Checks (thresholds from config.yaml → constants.py):
      A) YoE vs. inferred career timeline mismatch
      B) Current role start > total claimed YoE (impossible tenure)
      C) Expert skills with 0 duration (> limit instances)
      D) Expert skills with 0 endorsements (> limit instances)
    """
    profile = safe_profile(candidate)
    career  = safe_career(candidate)
    skills  = safe_skills(candidate)

    penalty = 1.0

    # ── A: YoE vs. career timeline ─────────────────────────────────────────────
    claimed_yoe = float(profile.get("years_of_experience") or 0)
    if career:
        oldest_start = None
        for role in career:
            d = _days_ago(role.get("start_date"))
            if d is not None:
                oldest_start = d if oldest_start is None else max(oldest_start, d)
        if oldest_start is not None:
            inferred_yoe = oldest_start / 365.25
            if abs(inferred_yoe - claimed_yoe) > HP_YOE_MISMATCH_THRESHOLD:
                penalty *= HP_YOE_MISMATCH_PENALTY

    # ── B: Current role tenure > total claimed YoE ─────────────────────────────
    for role in career:
        if role.get("is_current"):
            days = _days_ago(role.get("start_date"))
            if days is not None:
                current_yoe = days / 365.25
                if current_yoe > claimed_yoe + HP_CURRENT_ROLE_SLACK:
                    penalty *= HP_CURRENT_ROLE_PENALTY
            break

    # ── C: Expert skills with zero duration ────────────────────────────────────
    expert_zero_dur = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and (s.get("duration_months") or 0) == 0
    )
    if expert_zero_dur > HP_EXPERT_ZERO_DUR_LIMIT:
        penalty *= HP_EXPERT_ZERO_DUR_PENALTY
    elif expert_zero_dur > 0:
        penalty *= 0.85

    # ── D: Expert skills with zero endorsements ────────────────────────────────
    expert_zero_end = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and (s.get("endorsements") or 0) == 0
    )
    if expert_zero_end > HP_EXPERT_ZERO_END_LIMIT:
        penalty *= HP_EXPERT_ZERO_END_PENALTY

    # Floor to avoid completely erasing legit-but-unusual profiles
    return max(HP_MIN_MULTIPLIER, _clip(penalty))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FINAL COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_scores(candidate: dict, semantic_sim: float) -> Dict[str, float]:
    """
    Compute all score components + final weighted score.
    semantic_sim is the pre-computed cosine similarity from FAISS embedding.
    """
    career   = career_fit_score(candidate)
    skills   = skill_relevance_score(candidate)
    location = location_score(candidate)
    behavior = behavioral_score(candidate)
    penalty  = consistency_penalty(candidate)

    raw = (
        W_CAREER   * career
        + W_SKILLS   * skills
        + W_SEMANTIC * semantic_sim
        + W_LOCATION * location
        + W_BEHAVIOR * behavior
    )

    final = _clip(raw * penalty)

    return {
        "career":   career,
        "skills":   skills,
        "semantic": semantic_sim,
        "location": location,
        "behavior": behavior,
        "penalty":  penalty,
        "final":    final,
    }
