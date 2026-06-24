# Redrob Intelligent Candidate Ranking — Solution

Hybrid semantic + structured + behavioral ranking system for the
[Redrob Data & AI Challenge](https://redrob.io).

Ranks 100,000 candidates for a Senior AI Engineer role and produces a
submission CSV with exactly 100 ranked candidates, complete with deterministic
factual reasoning.

---

## Architecture

```
candidates.jsonl
      │
      ▼
[candidate_text.py]  →  headline + summary + career descriptions + skills
      │
      ▼
[embeddings.py]      →  all-MiniLM-L6-v2  (batched, CPU, normalised)
      │
      ▼
[FAISS FlatIP]       →  top-1500 by JD cosine similarity  (retrieval)
      │
      ▼
[scoring.py]         →  structured re-ranking
  ├── career_fit    (40%)  shipping evidence × domain terms × recency × seniority
  ├── skill_depth   (20%)  proficiency × endorsements × duration (anti-stuffing)
  ├── semantic_sim  (30%)  pre-computed cosine
  ├── location      ( 5%)  India + city preference + notice period
  └── behavioral    ( 5%)  availability + engagement + credibility
      │  × consistency_penalty (0.3–1.0)  ← honeypot detection
      ▼
[reasoning.py]       →  1–2 sentence factual recruiter note (≤200 chars)
      │
      ▼
submission.csv       →  100 rows: candidate_id, rank, score, reasoning
```

**Key design principles:**
- Career evidence dominates (0.40 weight). A candidate without shipping history cannot rank in the top 50.
- Semantic similarity supports retrieval, not ranking. It prevents re-ranking from working on a zero-coverage pool.
- Skill depth beats skill count. An "expert NLP, 40 endorsements, 48 months" beats 30 beginner AI keywords.
- Honeypot detection is soft and generic. No brittle blacklists; just three generic consistency checks that naturally catch impossible profiles.
- Fully deterministic. Same input → same output, every time.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run.
To pre-cache it (recommended for offline runs):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### 2. Precompute embeddings + FAISS index (one-time, ~15–25 min on CPU)

```bash
python scripts/precompute.py --candidates /path/to/candidates.jsonl
```

This saves to `artifacts/`:
- `faiss_index.bin` — FAISS FlatIP index (100K vectors)
- `jd_embedding.npy` — JD embedding
- `candidate_embeddings.npy` — all candidate embeddings
- `candidate_ids.npy` — ID alignment array

**Fast test with sample file** (< 60 seconds):
```bash
python scripts/precompute.py --candidates sample_candidates.json --sample 50
```

### 3. Rank (< 5 minutes on CPU after precompute)

```bash
python scripts/rank.py --candidates /path/to/candidates.jsonl
# Output: submission/submission.csv
```

Custom output path:
```bash
python scripts/rank.py --candidates /path/to/candidates.jsonl --out submission/my_team.csv
```

### 4. Validate

```bash
# Quick format check
python scripts/validate.py submission/submission.csv

# Full check including dataset ID existence
python scripts/validate.py submission/submission.csv --candidates /path/to/candidates.jsonl

# Determinism check (run rank.py twice, compare)
python scripts/validate.py submission/run1.csv --compare submission/run2.csv
```

### 5. Run tests

```bash
# Unit tests (no dataset required)
python tests/test_determinism.py
python tests/test_quality.py

# With pytest
pytest tests/ -v
```

---

## Scoring Formula

```
score = (
    0.40 × career_fit(candidate)
  + 0.20 × skill_depth(candidate)
  + 0.30 × cosine_similarity(candidate_embedding, jd_embedding)
  + 0.05 × location_logistics(candidate)
  + 0.05 × behavioral_signals(candidate)
) × consistency_penalty(candidate)

clipped to [0.0, 1.0]
```

### career_fit (40%)
Sub-components weighted as: shipping_evidence×0.35 + domain_density×0.25 + recency×0.20 + seniority×0.12 + company_type×0.08
- **Shipping evidence**: counts action verbs ("shipped", "deployed", "built", "launched") in role descriptions
- **Domain density**: counts ranking/retrieval/search/embedding/NLP terms in descriptions
- **Recency**: role start within 18 months = 1.0, 3 years = 0.7, 5+ years = 0.3
- **Seniority**: 5–9 years = 1.0, soft ramp below/above
- **Company type**: product company = 0.85, IT services/consulting = 0.35, weighted by duration

### skill_depth (20%)
For each AI-relevant skill: `depth = prof×0.55 + endorsements×0.25 + duration×0.20`
Stuffing penalties: >25 skills → ×0.7, >2 expert-zero-duration → ×0.6, many shallow AI skills → ×0.7

### semantic_sim (30%)
Cosine similarity between candidate text embedding and JD embedding, scaled [−1,1] → [0,1].

### location_logistics (5%)
India Pune/Noida = 1.0, other India metros = 0.90, India other = 0.80, international + relocate = 0.65, international no-relocate = 0.30. Notice: ≤15d = 1.0, ≤30d = 0.95, ≤60d = 0.80, ≤90d = 0.60, >120d = 0.20.

### behavioral_signals (5%)
Availability (open_to_work + recency) × 0.50 + engagement (response_rate + interview_cr) × 0.30 + credibility (email + phone + connections + GitHub) × 0.20.

### consistency_penalty (honeypot detection)
Returns a multiplier 0.3–1.0:
- YoE vs. career timeline mismatch > 3 years → ×0.55
- Current role start > total YoE + 1.5 years → ×0.35
- Expert skills with 0 duration_months, >2 instances → ×0.55
- Expert + 0 endorsements, >4 instances → ×0.70

---

## Repo Structure

```
redrob-ranker/
├── README.md
├── requirements.txt
├── .gitignore
├── submission_metadata.yaml
├── src/
│   ├── __init__.py
│   ├── constants.py          # JD text, weights, paths, keyword lists
│   ├── data_loader.py        # JSONL / JSON candidate loading
│   ├── candidate_text.py     # Text representation for embedding
│   ├── embeddings.py         # Encoding + FAISS build/load/retrieve
│   ├── scoring.py            # All 5 scoring components + composite
│   ├── reasoning.py          # Deterministic factual reasoning text
│   ├── ranking.py            # Full pipeline orchestrator
│   └── validation.py         # CSV validation (mirrors official validator)
├── scripts/
│   ├── precompute.py         # One-time offline embedding precomputation
│   ├── rank.py               # Main entry point (≤5 min)
│   └── validate.py           # Standalone validation CLI
├── tests/
│   ├── test_determinism.py   # Reproducibility + bounds checks
│   └── test_quality.py       # Scoring logic quality tests
├── docker/
│   └── Dockerfile
├── artifacts/                # (gitignored) FAISS index + embeddings
└── submission/               # (gitignored) output CSV
```

---

## Docker

```bash
# Build
docker build -f docker/Dockerfile -t redrob-ranker .

# Precompute (mount your data)
docker run --rm \
  -v /path/to/data:/data \
  -v $(pwd)/artifacts:/app/artifacts \
  redrob-ranker \
  python scripts/precompute.py --candidates /data/candidates.jsonl

# Rank
docker run --rm \
  -v /path/to/data:/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/submission:/app/submission \
  redrob-ranker \
  python scripts/rank.py --candidates /data/candidates.jsonl
```

---

## Tuning

To change weights without editing code, modify `src/constants.py`:

```python
W_CAREER   = 0.40   # increase if honeypot rate is high
W_SKILLS   = 0.20   # decrease if keyword stuffers still rank high
W_SEMANTIC = 0.30   # decrease if scores are too flat
W_LOCATION = 0.05
W_BEHAVIOR = 0.05
```

To change retrieval pool size (speed vs. coverage tradeoff):
```python
FAISS_RETRIEVAL_K = 1500   # reduce to 800 if runtime is > 4 min
```

---

## Pre-Submission Checklist

- [ ] `python scripts/precompute.py` ran without errors
- [ ] `python scripts/rank.py` completed in < 5 minutes
- [ ] `python scripts/validate.py submission/submission.csv --candidates candidates.jsonl` → PASSED
- [ ] Ran `rank.py` twice, compared outputs → identical
- [ ] Manually reviewed top-10: all should be engineers with shipping history
- [ ] Top-100 includes candidates from India (should be majority)
- [ ] No obvious honeypots in top-20 (check: no consultant-only careers with 20 AI skills + 0 endorsements)
- [ ] `submission_metadata.yaml` filled with your team details
- [ ] GitHub repo is clean (no secrets, no `.env`, no large binaries)

---

## What to Review Manually Before Submitting

1. **Top 5 candidates**: read their full profiles. Do they look like real senior ML engineers with shipping history? If any are HR managers or pure consultants, your career_fit or penalty weights need adjustment.
2. **Reasoning text**: check 5 random rows. Does the reasoning accurately reflect the profile data? No hallucinations?
3. **Score distribution**: top score should be ~0.7–0.9, bottom (rank 100) should be ~0.4–0.6. Tight clustering around 0.5 = low discrimination.
4. **India representation**: with location weight + JD preference, majority of top-100 should be India-based.
5. **Runtime**: `time python scripts/rank.py --candidates candidates.jsonl` should print < 5 minutes.

---

## Regenerating the Submission Deterministically

```bash
# Clean run from scratch (after precompute)
python scripts/rank.py --candidates candidates.jsonl --out submission/submission_v2.csv

# Verify against previous run
python scripts/validate.py submission/submission_v1.csv --compare submission/submission_v2.csv
# Expected: "✅  Outputs are identical (deterministic)"
```

The ranking is deterministic because:
- Embedding model weights are fixed (pinned version in requirements.txt)
- FAISS FlatIP is exact (no approximation randomness)
- Sorting uses `(-score, candidate_id)` as deterministic tiebreaker
- Scores are rounded to 4 decimal places before sorting

---

## Why This Works

The sample submission (`sample_submission.csv`) shows the naive approach ranks an **HR Manager** at rank 1 and a **Content Writer** at rank 4. These are textbook honeypots — high AI keyword count, zero relevant experience. Our system avoids them because:

1. `career_fit_score` requires shipping evidence in role **descriptions** (not skill labels)
2. `skill_relevance_score` requires endorsements + duration (not just keyword presence)
3. `consistency_penalty` catches impossible profiles automatically

Result: honeypots will typically score 0.05–0.25 in career_fit and get ×0.3–0.6 penalty, landing them well outside the top 100.
