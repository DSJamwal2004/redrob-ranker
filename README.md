# Redrob Intelligent Candidate Ranking — Final Solution

Hybrid semantic + structured + behavioral ranking system for the Redrob Data & AI Challenge.

Ranks 100,000 candidates for a Senior AI Engineer role and produces a submission CSV with exactly 100 ranked candidates, complete with deterministic factual reasoning.

---

## Architecture

```text
candidates.jsonl
      │
      ▼
[candidate_text.py]
      │
      ▼
[embeddings.py]
  → sentence-transformers/all-MiniLM-L6-v2
  → 384-dimensional embeddings
      │
      ▼
[FAISS FlatIP]
  → Top-1500 semantic retrieval
      │
      ▼
[scoring.py]
  ├── Career Fit       (50%)
  ├── Skill Depth      (20%)
  ├── Semantic Match   (15%)
  ├── Behavioral       (10%)
  └── Location         ( 5%)
          ×
   Consistency Penalty
   (0.30–1.00)
      │
      ▼
[reasoning.py]
  → Deterministic factual recruiter note
      │
      ▼
submission.csv
  → candidate_id
  → rank
  → score
  → reasoning
```

### Key Design Principles

* Career evidence dominates ranking.
* Semantic similarity is used for retrieval support, not ranking dominance.
* Skill depth matters more than skill count.
* Behavioral signals prevent unavailable candidates from ranking highly.
* Honeypot detection uses soft consistency penalties rather than hard filters.
* Fully deterministic output.

---

## Final Scoring Formula

```text
score = (
    0.50 × career_fit(candidate)
  + 0.20 × skill_depth(candidate)
  + 0.15 × semantic_similarity(candidate)
  + 0.10 × behavioral_signals(candidate)
  + 0.05 × location_logistics(candidate)
) × consistency_penalty(candidate)
```

Scores are clipped to [0,1].

---

### Career Fit (50%)

Primary ranking signal.

Measures:

* Shipping evidence
* Search / ranking / retrieval relevance
* Domain density
* Career progression
* Company-type heuristic

Examples of strong evidence:

* built ranking systems
* shipped retrieval pipelines
* search infrastructure ownership
* recommendation systems
* embeddings infrastructure

---

### Skill Depth (20%)

Measures quality rather than quantity.

Signals:

* proficiency
* endorsements
* duration
* anti-stuffing checks

A candidate with:

```text
NLP (Expert)
48 months
25 endorsements
```

scores significantly higher than someone listing many shallow AI skills.

---

### Semantic Similarity (15%)

Cosine similarity between:

```text
candidate embedding
vs
job description embedding
```

Used primarily to retrieve relevant candidates efficiently from the 100K pool.

---

### Behavioral Signals (10%)

Measures hiring practicality.

Signals:

* open-to-work
* recruiter response rate
* last active date
* profile verification
* connection count

Inactive candidates are naturally down-weighted.

---

### Location & Logistics (5%)

Measures:

* India preference
* preferred cities
* willingness to relocate
* notice period

Preferred cities:

```text
Pune
Noida
```

Supported metros:

```text
Bangalore
Hyderabad
Mumbai
Delhi NCR
Chennai
Kolkata
```

---

### Consistency Penalty (Honeypot Detection)

Soft multiplier:

```text
0.30 → 1.00
```

Checks:

* YoE mismatch
* impossible timelines
* expert skills with zero evidence
* suspicious skill inflation

This prevents obvious keyword-stuffers from entering the top ranks.

---

## Retrieval Strategy

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Properties:

* 384 dimensions
* CPU friendly
* deterministic
* strong retrieval quality

---

### Retrieval Pipeline

```text
100,000 candidates
        ↓
FAISS search
        ↓
Top 1,500 retrieved
        ↓
Structured scoring
        ↓
Top 100 selected
```

Single-stage retrieval was intentionally chosen to avoid unnecessary complexity.

---

## Configuration

All major knobs are centralized in:

```text
config.yaml
```

Includes:

* scoring weights
* retrieval_k
* company scores
* honeypot thresholds
* behavioral thresholds
* location preferences
* artifact paths
* output precision

No source-code changes are required for tuning.

---

## Repository Structure

```text
redrob-ranker/
├── README.md
├── requirements.txt
├── config.yaml
├── jd.txt
├── submission_metadata.yaml
│
├── src/
│   ├── constants.py
│   ├── data_loader.py
│   ├── candidate_text.py
│   ├── embeddings.py
│   ├── scoring.py
│   ├── reasoning.py
│   ├── ranking.py
│   └── validation.py
│
├── scripts/
│   ├── precompute.py
│   ├── rank.py
│   ├── validate.py
│   ├── config_sanity.py
│   ├── retrieval_diagnostics.py
│   └── top100_summary.py
│
├── tests/
│   ├── test_determinism.py
│   └── test_quality.py
│
├── docker/
│   └── Dockerfile
│
├── artifacts/      (gitignored)
├── submission/     (gitignored)
└── final_submission/
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Validate Configuration

```bash
python scripts/config_sanity.py
```

Expected:

```text
All checks passed
Configuration looks healthy
```

---

### 3. Precompute Embeddings

```bash
python scripts/precompute.py --candidates candidates.jsonl
```

Creates:

```text
artifacts/
├── faiss_index.bin
├── jd_embedding.npy
├── candidate_embeddings.npy
└── candidate_ids.npy
```

---

### 4. Generate Ranking

```bash
python scripts/rank.py --candidates candidates.jsonl --out submission/submission.csv
```

Runtime after precompute:

```text
~6 seconds
```

---

### 5. Validate Submission

```bash
python scripts/validate.py submission/submission.csv --candidates candidates.jsonl
```

Expected:

```text
PASSED
```

---

### 6. Generate Submission Summary

```bash
python scripts/top100_summary.py \
  --submission submission/submission.csv \
  --candidates candidates.jsonl
```

Displays:

* score spread
* geography breakdown
* notice periods
* reasoning quality
* component averages

---

### 7. Retrieval Diagnostics

```bash
python scripts/retrieval_diagnostics.py \
  --candidates candidates.jsonl \
  --k 500
```

Displays:

* similarity distribution
* retrieval coverage
* top retrieved candidates
* domain relevance metrics

---

## Testing

Run all tests:

```bash
pytest
```

Final results:

```text
16 passed
0 failed
```

Including:

* determinism tests
* quality tests
* scoring sanity checks
* honeypot detection tests

---

## Final Benchmark Results

Full Dataset:

```text
Candidates: 100,000
```

Precompute:

```text
1h 50m CPU
```

Ranking Runtime:

```text
6.1 seconds
```

Top Candidate Score:

```text
0.8873
```

Rank #100 Score:

```text
0.7718
```

Score Spread:

```text
0.1155
```

Geography:

```text
India: 92%
```

Reasoning Quality:

```text
100 unique reasoning strings
0 hallucinations observed
```

Validation:

```text
PASSED
```

Determinism:

```text
Identical output across repeated runs
```

---

## Pre-Submission Checklist

* [x] Precompute completed successfully
* [x] Ranking runtime < 5 minutes
* [x] Validation passed
* [x] Determinism verified
* [x] Top candidates manually reviewed
* [x] Reasoning reviewed
* [x] Submission CSV generated
* [x] GitHub repository prepared
* [x] Final configuration saved

---

## Why This Works

Many naive solutions rank keyword-heavy profiles highly.

This system prioritizes:

1. Actual shipping evidence
2. Relevant AI/search/retrieval experience
3. Skill credibility
4. Candidate availability
5. Timeline consistency

The result is a shortlist that more closely resembles how an experienced recruiter would evaluate candidates rather than how a keyword search engine would rank them.

