"""
constants.py — central configuration loader.

Priority order for values:
  1. config.yaml (repo root) — edit this for tuning
  2. Hardcoded defaults below (fallback if config.yaml absent)

JD text is loaded from jd.txt (repo root) if it exists;
falls back to the embedded string below.

Do not move tuning numbers into other source files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── Load config.yaml (optional — graceful fallback) ─────────────────────────
_cfg: Dict[str, Any] = {}

_config_path = REPO_ROOT / "config.yaml"
if _config_path.exists():
    try:
        import yaml  # pyyaml
        with open(_config_path, "r", encoding="utf-8") as _fh:
            _cfg = yaml.safe_load(_fh) or {}
        logger.debug("Loaded config from %s", _config_path)
    except ImportError:
        logger.warning(
            "pyyaml not installed; falling back to built-in defaults. "
            "Run: pip install pyyaml"
        )
    except Exception as exc:
        logger.warning("Failed to parse config.yaml: %s — using defaults", exc)


def _get(keys: str, default: Any) -> Any:
    """Dot-path getter for nested config dict, e.g. 'scoring.weights.career'."""
    node = _cfg
    for k in keys.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ─── Paths (configurable) ─────────────────────────────────────────────────────
_artifacts_rel = _get("paths.artifacts_dir", "artifacts")
_submission_rel = _get("paths.submission_dir", "submission")

ARTIFACTS_DIR  = Path(_artifacts_rel) if Path(_artifacts_rel).is_absolute() else REPO_ROOT / _artifacts_rel
SUBMISSION_DIR = Path(_submission_rel) if Path(_submission_rel).is_absolute() else REPO_ROOT / _submission_rel

ARTIFACTS_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH     = ARTIFACTS_DIR / "faiss_index.bin"
JD_EMBEDDING_PATH    = ARTIFACTS_DIR / "jd_embedding.npy"
CAND_EMBEDDINGS_PATH = ARTIFACTS_DIR / "candidate_embeddings.npy"
CAND_IDS_PATH        = ARTIFACTS_DIR / "candidate_ids.npy"
CAND_FEATURES_PATH   = ARTIFACTS_DIR / "candidate_features.pkl"

# ─── Model ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = _get("model.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(_get("model.embedding_batch_size", 256))
EMBEDDING_DIM        = int(_get("model.embedding_dim", 384))

# ─── Retrieval ────────────────────────────────────────────────────────────────
FAISS_RETRIEVAL_K = int(_get("retrieval.retrieval_k", 1500))
FINAL_TOP_N       = int(_get("retrieval.final_top_n", 100))

# ─── Scoring weights (must sum to 1.0) ────────────────────────────────────────
# Career evidence dominates; semantic is a retrieval support signal.
W_CAREER   = float(_get("scoring.weights.career",   0.50))
W_SKILLS   = float(_get("scoring.weights.skills",   0.20))
W_SEMANTIC = float(_get("scoring.weights.semantic", 0.15))
W_LOCATION = float(_get("scoring.weights.location", 0.05))
W_BEHAVIOR = float(_get("scoring.weights.behavior", 0.10))
SCORE_PRECISION = int(_get("scoring.score_precision", 4))

_weight_sum = round(W_CAREER + W_SKILLS + W_SEMANTIC + W_LOCATION + W_BEHAVIOR, 6)
if abs(_weight_sum - 1.0) > 1e-4:
    logger.warning("Scoring weights sum to %.6f, not 1.0 — check config.yaml", _weight_sum)

# ─── Company type heuristic (light signal, not prestige filter) ───────────────
COMPANY_CONSULTING_SCORE = float(_get("company.consulting_score", 0.45))
COMPANY_PRODUCT_SCORE    = float(_get("company.product_score",    0.82))

# ─── Honeypot / consistency penalty thresholds ────────────────────────────────
HP_YOE_MISMATCH_THRESHOLD = float(_get("honeypot.yoe_mismatch_threshold_years", 3.0))
HP_YOE_MISMATCH_PENALTY   = float(_get("honeypot.yoe_mismatch_penalty",         0.55))
HP_EXPERT_ZERO_DUR_LIMIT  = int(  _get("honeypot.expert_zero_duration_limit",   2))
HP_EXPERT_ZERO_DUR_PENALTY= float(_get("honeypot.expert_zero_duration_penalty", 0.55))
HP_EXPERT_ZERO_END_LIMIT  = int(  _get("honeypot.expert_zero_endorse_limit",    4))
HP_EXPERT_ZERO_END_PENALTY= float(_get("honeypot.expert_zero_endorse_penalty",  0.70))
HP_CURRENT_ROLE_SLACK     = float(_get("honeypot.current_role_yoe_slack_years", 1.5))
HP_CURRENT_ROLE_PENALTY   = float(_get("honeypot.current_role_yoe_penalty",     0.35))
HP_MIN_MULTIPLIER         = float(_get("honeypot.min_penalty_multiplier",        0.30))

# ─── Behavioral thresholds ────────────────────────────────────────────────────
BEH_ACTIVE_DAYS        = int(_get("behavioral.active_days",            30))
BEH_RECENT_DAYS        = int(_get("behavioral.recent_days",            90))
BEH_MODERATE_DAYS      = int(_get("behavioral.moderate_days",         180))
BEH_STALE_DAYS         = int(_get("behavioral.stale_days",            365))
BEH_INACTIVITY_CONCERN = int(_get("behavioral.inactivity_concern_days", 180))

# ─── Location preferences ─────────────────────────────────────────────────────
_pref = _get("locations.preferred_cities", ["pune", "noida"])
_ok   = _get("locations.ok_cities", [
    "bangalore", "bengaluru", "hyderabad", "mumbai",
    "gurgaon", "gurugram", "delhi", "new delhi", "chennai", "kolkata",
])
PREFERRED_CITIES = set(_pref) if isinstance(_pref, list) else set(_pref.values())
OK_CITIES        = set(_ok)   if isinstance(_ok,   list) else set(_ok.values())

# ─── JD text (loaded from jd.txt if present, else embedded fallback) ──────────
_JD_FALLBACK = """
Senior AI Engineer — Founding Team (Series A, India)

We are building the next-generation intelligent search and recommendation
platform. We are a small, fast-moving team and you would be one of the first
senior AI hires. You will own the ranking and retrieval stack end-to-end.

Responsibilities:
- Design, build, and ship production-grade ranking and retrieval systems
  (embedding-based retrieval, re-rankers, hybrid search) to real users.
- Own the offline and online evaluation pipeline: NDCG, MRR, MAP, A/B tests.
- Work hands-on with vector databases (Faiss, Pinecone, Weaviate, Qdrant,
  Elasticsearch, OpenSearch).
- Fine-tune and evaluate sentence-transformers, BGE, E5, and similar models.
- Collaborate with product and data teams to iterate quickly.
- Write production Python — not notebooks, not PoCs.

Requirements:
- 5–9 years total experience; 4–5 years in applied ML/AI at product companies
  (not consulting firms, not research-only).
- Must have shipped at least one end-to-end ranking, search, or recommendation
  system to real users in the last 18 months.
- Deep hands-on experience with information retrieval, NLP, embeddings,
  LLM integration, and evaluation frameworks.
- Strong Python (PyTorch, HuggingFace, sentence-transformers, scikit-learn).
- Experience with vector databases and hybrid search.
- Not a framework enthusiast — a systems thinker who can justify every design
  decision.

What automatically disqualifies:
- Entire career at consulting firms (TCS, Infosys, Wipro) with no product
  company experience.
- "AI experience" = only recent LangChain + OpenAI wrapper work.
- No production code in the last 18 months.
- Only computer vision or speech, no NLP or information retrieval.
- Pure research background with no deployment history.

Location: Pune or Noida preferred; Bangalore, Hyderabad, Mumbai also fine.
Open to relocation strongly preferred. Sub-30-day notice ideal.

Salary: Competitive. Equity. Series A.

Keywords: ranking, retrieval, recommendation, embeddings, vector search,
NLP, information retrieval, sentence-transformers, FAISS, Pinecone,
evaluation, NDCG, MRR, LLM, fine-tuning, RAG, production ML, Python,
PyTorch, HuggingFace, senior AI engineer.
"""

_jd_file = REPO_ROOT / _get("paths.jd_text_file", "jd.txt")
if _jd_file.exists():
    try:
        JD_TEXT = _jd_file.read_text(encoding="utf-8").strip()
        logger.debug("JD text loaded from %s (%d chars)", _jd_file, len(JD_TEXT))
    except Exception as exc:
        logger.warning("Could not read %s: %s — using embedded fallback", _jd_file, exc)
        JD_TEXT = _JD_FALLBACK.strip()
else:
    JD_TEXT = _JD_FALLBACK.strip()

# ─── Career fit keywords ──────────────────────────────────────────────────────
SHIPPING_VERBS = [
    "shipped", "deployed", "launched", "built", "released", "delivered",
    "implemented", "developed", "created", "designed and shipped",
    "productionized", "rolled out",
]

DOMAIN_TERMS = [
    "ranking", "rank", "retrieval", "search", "recommendation", "recommender",
    "embedding", "embeddings", "vector", "nlp", "information retrieval",
    "re-rank", "rerank", "semantic search", "dense retrieval", "rag",
    "llm", "fine-tun", "fine tuning", "sentence-transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "elasticsearch", "opensearch",
    "ndcg", "mrr", "map@", "a/b test", "offline eval", "evaluation",
]

# ─── Skill relevance keywords ──────────────────────────────────────────────────
AI_SKILL_KEYWORDS = [
    "ml", "machine learning", "nlp", "natural language", "embedding",
    "retrieval", "ranking", "llm", "vector", "rag", "fine-tun", "fine tuning",
    "transformer", "bert", "gpt", "sentence-transformer", "information retrieval",
    "recommendation", "recommender", "search", "faiss", "pinecone", "weaviate",
    "qdrant", "elasticsearch", "opensearch", "pytorch", "huggingface",
    "evaluation", "ndcg", "mrr",
]

# ─── Consulting firm heuristic (soft penalty only) ────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "cognizant", "capgemini",
    "accenture", "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree", "mindtree", "zensar", "niit technologies", "mastech",
    "syntel", "patni", "birlasoft",
}
