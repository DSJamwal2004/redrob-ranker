"""
candidate_text.py — construct the text representation of a candidate for embedding.

Strategy:
- Headline + summary first (semantic overview)
- Career descriptions (most signal-rich)
- Skills (name only, not proficiency — keeps text dense, not noisy)
- Education field

Keep it under ~512 tokens to fit MiniLM context window cleanly.
"""

from .data_loader import safe_career, safe_education, safe_profile, safe_skills


def build_candidate_text(candidate: dict) -> str:
    """
    Return a single text string for FAISS embedding.
    Deterministic for the same input.
    """
    profile = safe_profile(candidate)
    career  = safe_career(candidate)
    skills  = safe_skills(candidate)
    edu     = safe_education(candidate)

    parts = []

    # 1. Headline + summary (high signal for JD similarity)
    headline = profile.get("headline", "").strip()
    summary  = profile.get("summary", "").strip()
    if headline:
        parts.append(headline)
    if summary:
        # Truncate to ~300 chars — enough for the key sentence
        parts.append(summary[:400])

    # 2. Current title + company
    title   = profile.get("current_title", "")
    company = profile.get("current_company", "")
    if title:
        parts.append(f"Current role: {title} at {company}")

    # 3. Career descriptions — most informative for shipping evidence
    for role in career[:4]:                     # top 4 roles max
        desc = role.get("description", "").strip()
        rtitle = role.get("title", "")
        rco    = role.get("company", "")
        if desc:
            parts.append(f"{rtitle} at {rco}: {desc[:300]}")

    # 4. Skills (names only — avoids embedding noise from proficiency labels)
    skill_names = [s.get("name", "") for s in skills if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names[:20]))

    # 5. Education field
    for e in edu[:2]:
        field = e.get("field_of_study", "")
        degree = e.get("degree", "")
        inst   = e.get("institution", "")
        if field:
            parts.append(f"{degree} {field} {inst}")

    text = " | ".join(p for p in parts if p)
    return text if text else "unknown candidate"
