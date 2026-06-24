"""
reasoning.py — generate deterministic, factual, recruiter-style reasoning.

Rules:
- 1–2 sentences max, ≤ 200 characters
- Mention strongest fact first (career evidence > skills > location/availability)
- Mention concern only if meaningful (low engagement, long notice)
- Never speculate, never hallucinate data not in the profile
- Template-based → deterministic for same input
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from .constants import AI_SKILL_KEYWORDS, BEH_INACTIVITY_CONCERN, DOMAIN_TERMS, SHIPPING_VERBS
from .data_loader import safe_career, safe_profile, safe_signals, safe_skills

NOW = datetime.now(timezone.utc)


def _days_ago(date_str: Optional[str]) -> Optional[float]:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((NOW - dt).days, 0.0)
    except (ValueError, TypeError):
        return None


def _best_domain_role(career: list) -> Optional[dict]:
    """Return the career role with the most domain-relevant terms."""
    best_role = None
    best_hits = 0
    for role in career:
        desc = (role.get("description") or "").lower()
        hits = sum(1 for t in DOMAIN_TERMS if t in desc)
        if hits > best_hits:
            best_hits = hits
            best_role = role
    return best_role


def _top_ai_skills(skills: list, n: int = 2) -> list:
    """Return up to n AI-relevant skill names, sorted by endorsements desc."""
    ai_skills = [
        s for s in skills
        if any(kw in (s.get("name") or "").lower() for kw in AI_SKILL_KEYWORDS)
    ]
    ai_skills.sort(key=lambda s: -(s.get("endorsements") or 0))
    return [s["name"] for s in ai_skills[:n]]


def generate_reasoning(candidate: dict, scores: Dict[str, float]) -> str:
    """
    Build a ≤ 200 character recruiter note. Factual and deterministic.
    """
    profile = safe_profile(candidate)
    career  = safe_career(candidate)
    skills  = safe_skills(candidate)
    sig     = safe_signals(candidate)

    yoe     = float(profile.get("years_of_experience") or 0)
    title   = profile.get("current_title", "")
    company = profile.get("current_company", "")
    country = profile.get("country", "")
    loc     = profile.get("location", "")

    # ── Part 1: strongest positive fact ───────────────────────────────────────
    main_parts = []

    if yoe > 0 and title:
        main_parts.append(f"{yoe:.0f}yr {title}")
    elif title:
        main_parts.append(title)

    if company:
        main_parts.append(f"@ {company}")

    best_role = _best_domain_role(career)
    if best_role:
        desc = (best_role.get("description") or "").lower()
        # pick the first strong shipping verb found
        shipped_verb = next(
            (v for v in ["shipped", "built", "deployed", "launched", "implemented"]
             if v in desc),
            None
        )
        # pick the first domain term found
        domain_hit = next(
            (t for t in ["ranking", "retrieval", "search", "recommendation",
                          "embedding", "rag", "nlp", "information retrieval"]
             if t in desc),
            None
        )
        if shipped_verb and domain_hit:
            main_parts.append(f"({shipped_verb} {domain_hit})")
        elif domain_hit:
            main_parts.append(f"({domain_hit} work)")

    top_skills = _top_ai_skills(skills)
    if top_skills:
        main_parts.append(f"[{', '.join(top_skills)}]")

    if country == "India":
        main_parts.append(f"{loc}")
    elif sig.get("willing_to_relocate"):
        main_parts.append(f"{country}, open to relocate")

    sentence1 = "; ".join(p for p in main_parts if p) + "."

    # ── Part 2: concerns (only if meaningful) ─────────────────────────────────
    concerns = []

    rr = float(sig.get("recruiter_response_rate") or 0)
    if rr < 0.25:
        concerns.append("low recruiter engagement")

    notice = int(sig.get("notice_period_days") or 60)
    if notice > 90:
        concerns.append(f"{notice}d notice")

    days_inactive = _days_ago(sig.get("last_active_date"))
    if days_inactive is not None and days_inactive > BEH_INACTIVITY_CONCERN:
        months_inactive = int(days_inactive / 30)
        concerns.append(f"inactive {months_inactive}mo")

    # Consistency issue
    if scores.get("penalty", 1.0) < 0.6:
        concerns.append("profile consistency flag")

    sentence2 = ""
    if concerns:
        sentence2 = " Note: " + ", ".join(concerns) + "."

    full = sentence1 + sentence2

    # Truncate to 200 chars at a word boundary
    if len(full) > 200:
        full = full[:197] + "..."

    return full
