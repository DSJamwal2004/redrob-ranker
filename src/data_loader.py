"""
data_loader.py — load and parse candidate JSONL / JSON data.

All public functions return plain Python dicts; no custom classes needed.
"""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """
    Yield one candidate dict per line from a JSONL file.
    Skips blank lines and logs parse errors without crashing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        # Sample file is a JSON array, not JSONL
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return

    # JSONL
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)


def load_all_candidates(path: str | Path) -> List[dict]:
    """Load entire file into memory. Fine for 100K × ~5KB = ~500MB."""
    return list(iter_candidates(path))


def get_candidate_ids(candidates: List[dict]) -> set:
    return {c["candidate_id"] for c in candidates}


# ─── Safe field accessors ─────────────────────────────────────────────────────

def safe_profile(c: dict) -> dict:
    return c.get("profile", {})

def safe_career(c: dict) -> list:
    return c.get("career_history", [])

def safe_skills(c: dict) -> list:
    return c.get("skills", [])

def safe_education(c: dict) -> list:
    return c.get("education", [])

def safe_signals(c: dict) -> dict:
    return c.get("redrob_signals", {})

def safe_certifications(c: dict) -> list:
    return c.get("certifications", [])
