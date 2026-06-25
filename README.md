# Redrob Intelligent Candidate Ranking — Final Solution

> Hybrid semantic + structured + behavioral ranking system for the **Redrob Data & AI Challenge**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-all--MiniLM--L6--v2-green)](https://www.sbert.net/)
[![Deterministic](https://img.shields.io/badge/Output-Deterministic-success)]()
[![Challenge](https://img.shields.io/badge/Submission-Ready-red)]()

---

## 🚀 Live Interactive Demo

**Live Demo:** https://DSJamwal2004.github.io/redrob-ranker/demo/

This repository includes an interactive recruiter-style prototype that demonstrates the complete ranking workflow from a recruiter’s perspective. It visualizes semantic retrieval, hybrid scoring, candidate reasoning, comparison mode, analytics, and semantic-vs-keyword behavior.

### Demo Features

- Job Description driven candidate ranking
- Semantic retrieval with FAISS
- Explainable candidate score breakdown
- Recruiter-style shortlist view
- Side-by-side candidate comparison
- Analytics dashboard
- Semantic vs Keyword ranking visualization
- Consistency / honeypot detection signals
- Fully interactive UI prototype

> **Note:** The live demo is a front-end prototype designed to illustrate the product experience and ranking logic. It complements the production ranking engine implemented in this repository.

---

## Project Overview

Recruiters often need to review thousands of profiles, but keyword filters miss strong candidates and over-rank buzzword-heavy profiles. This project solves that problem by ranking **100,000 candidates** for a Senior AI Engineer role using a hybrid method that combines semantic retrieval with structured recruiter-inspired scoring.

Instead of matching only keywords, the system evaluates:

- career evidence,
- skill depth,
- semantic relevance,
- behavioral signals,
- location and logistics,
- and profile consistency.

The result is a shortlist that is closer to how an experienced recruiter would evaluate candidates than how a search engine would.

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

- Career evidence dominates ranking.
- Semantic similarity is used for retrieval support, not ranking dominance.
- Skill depth matters more than skill count.
- Behavioral signals prevent unavailable candidates from ranking highly.
- Honeypot detection uses soft consistency penalties rather than hard filters.
- Fully deterministic output.
- Ranking logic is recruiter-first, not keyword-first.

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

Scores are clipped to `[0, 1]`.

---

## Scoring Components

### Career Fit (50%)

Primary ranking signal.

Measures:

- shipping evidence
- search / ranking / retrieval relevance
- domain density
- career progression
- company-type heuristic

Examples of strong evidence:

- built ranking systems
- shipped retrieval pipelines
- search infrastructure ownership
- recommendation systems
- embeddings infrastructure

---

### Skill Depth (20%)

Measures quality rather than quantity.

Signals:

- proficiency
- endorsements
- duration
- anti-stuffing checks

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

- open-to-work
- recruiter response rate
- last active date
- profile verification
- connection count

Inactive candidates are naturally down-weighted.

---

### Location & Logistics (5%)

Measures:

- India preference
- preferred cities
- willingness to relocate
- notice period

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

- YoE mismatch
- impossible timelines
- expert skills with zero evidence
- suspicious skill inflation

This prevents obvious keyword-stuffers from entering the top ranks while still preserving unusual but legitimate profiles.

---

## Retrieval Strategy

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Properties:

- 384 dimensions
- CPU friendly
- deterministic
- strong retrieval quality

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

Single-stage retrieval was intentionally chosen to avoid unnecessary complexity while still scaling efficiently.

---

## Explanation / Reasoning Layer

Every shortlisted candidate includes a deterministic reasoning string that is built from actual profile signals. The reasoning avoids hallucination and focuses on recruiter-friendly evidence such as:

- job title relevance,
- shipping evidence,
- experience level,
- domain match,
- skill depth,
- logistics fit,
- and consistency checks.

This makes the ranking interpretable and easier to trust.

---

## Configuration

All major knobs are centralized in:

```text
config.yaml
```

Includes:

- scoring weights
- retrieval_k
- company scores
- honeypot thresholds
- behavioral thresholds
- location preferences
- artifact paths
- output precision

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
├── demo/
│   ├── index.html
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
python scripts/top100_summary.py   --submission submission/submission.csv   --candidates candidates.jsonl
```

Displays:

- score spread
- geography breakdown
- notice periods
- reasoning quality
- component averages

---

### 7. Retrieval Diagnostics

```bash
python scripts/retrieval_diagnostics.py   --candidates candidates.jsonl   --k 500
```

Displays:

- similarity distribution
- retrieval coverage
- top retrieved candidates
- domain relevance metrics

---

## Interactive Demo Usage

The live demo is a static front-end prototype that can also be run locally.

### Open the hosted version

```text
https://DSJamwal2004.github.io/redrob-ranker/demo/
```

### Run locally

Open:

```text
demo/index.html
```

You can use it to:

- view the dashboard,
- inspect top-ranked candidates,
- compare candidate profiles,
- study score breakdowns,
- and switch to the semantic-vs-keyword explanation tab.

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

- determinism tests
- quality tests
- scoring sanity checks
- honeypot detection tests

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

## Project Highlights

- Hybrid semantic + structured ranking
- FAISS vector retrieval at scale
- Explainable recruiter reasoning
- Deterministic ranking output
- Soft consistency / honeypot detection
- Fully validated CSV submission
- Interactive live demo prototype
- Product-quality presentation layer

---

## Pre-Submission Checklist

- [x] Precompute completed successfully
- [x] Ranking runtime < 5 minutes
- [x] Validation passed
- [x] Determinism verified
- [x] Top candidates manually reviewed
- [x] Reasoning reviewed
- [x] Submission CSV generated
- [x] GitHub repository prepared
- [x] Final configuration saved
- [x] Live demo prepared

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

The interactive demo makes that logic visible to judges by showing the shortlist, score breakdown, candidate comparisons, and semantic-vs-keyword behavior in one clean interface.

---

## Submission Notes

This repository is designed to support the full Redrob challenge submission:

- clean, working GitHub repo
- final ranked CSV output
- PDF deck explaining the approach
- interactive demo prototype for presentation



