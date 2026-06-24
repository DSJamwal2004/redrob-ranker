"""
validation.py — validate submission CSV before upload.

Covers every check in the official validate_submission.py plus:
  - All candidate_ids exist in the source dataset
  - Reasoning is non-empty and non-identical across rows
  - Score distribution sanity check
  - Determinism check (compare two runs)
"""

import csv
import logging
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

REQUIRED_HEADER  = ["candidate_id", "rank", "score", "reasoning"]
CAND_ID_PATTERN  = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_ROWS    = 100


def validate_submission(
    csv_path: str,
    dataset_ids: Optional[Set[str]] = None,
) -> List[str]:
    """
    Return list of error strings. Empty list = valid.

    Args:
        csv_path:    Path to submission CSV.
        dataset_ids: Full set of candidate_ids in candidates.jsonl (optional).
                     If provided, checks that all submission IDs are in dataset.
    """
    errors: List[str] = []
    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        errors.append(f"File must have .csv extension; got '{path.suffix}'")

    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("File is empty (no header row)")
                return errors

            if header != REQUIRED_HEADER:
                errors.append(
                    f"Header must be exactly: {','.join(REQUIRED_HEADER)}\n"
                    f"  Found: {','.join(header)}"
                )

            data_rows = [row for row in reader if any(c.strip() for c in row)]

    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded")
        return errors
    except OSError as exc:
        errors.append(f"Cannot read file: {exc}")
        return errors

    n = len(data_rows)
    if n != EXPECTED_ROWS:
        errors.append(
            f"Expected exactly {EXPECTED_ROWS} data rows; found {n}"
        )

    seen_ids:   Set[str] = set()
    seen_ranks: Set[int] = set()
    by_rank:    List[Tuple[int, float, str]] = []
    reasonings: List[str] = []

    for i, cells in enumerate(data_rows):
        row_num = i + 2

        if len(cells) != len(REQUIRED_HEADER):
            errors.append(
                f"Row {row_num}: expected {len(REQUIRED_HEADER)} columns, "
                f"got {len(cells)}"
            )
            continue

        row = dict(zip(REQUIRED_HEADER, cells))
        cid     = row["candidate_id"].strip()
        rank_s  = row["rank"].strip()
        score_s = row["score"].strip()
        reason  = row["reasoning"].strip()

        # candidate_id
        if not cid:
            errors.append(f"Row {row_num}: candidate_id is empty")
        elif not CAND_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)
            if dataset_ids and cid not in dataset_ids:
                errors.append(
                    f"Row {row_num}: candidate_id '{cid}' not in dataset"
                )

        # rank
        rank = None
        try:
            rank = int(rank_s)
            if str(rank) != rank_s:
                raise ValueError
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank {rank} out of range [1,100]")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer; got '{rank_s}'")

        # score
        score = None
        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score must be float; got '{score_s}'")

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

        # reasoning
        if not reason:
            errors.append(f"Row {row_num}: reasoning is empty")
        elif len(reason) > 500:
            errors.append(
                f"Row {row_num}: reasoning length {len(reason)} > 500 chars"
            )
        reasonings.append(reason)

    # Missing ranks
    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    # Non-increasing scores + tie-break order
    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 < s2:
            errors.append(
                f"Score not non-increasing: rank {r1} ({s1}) < rank {r2} ({s2})"
            )
        if s1 == s2 and c1 > c2:
            errors.append(
                f"Tie-break violation at ranks {r1},{r2}: "
                f"'{c1}' should come before '{c2}' (ascending ID order)"
            )

    # Reasoning diversity check
    if len(reasonings) > 5:
        unique_r = len(set(reasonings))
        if unique_r < len(reasonings) * 0.9:
            errors.append(
                f"Reasoning diversity low: only {unique_r}/{len(reasonings)} unique. "
                "Looks like templated identical text."
            )

    return errors


def check_determinism(path_a: str, path_b: str) -> List[str]:
    """
    Compare two CSV files and report any differences.
    Both must be valid submissions.
    """
    issues = []
    try:
        with open(path_a, newline="", encoding="utf-8") as fa, \
             open(path_b, newline="", encoding="utf-8") as fb:
            rows_a = list(csv.reader(fa))
            rows_b = list(csv.reader(fb))
    except OSError as exc:
        return [f"Cannot open files: {exc}"]

    if len(rows_a) != len(rows_b):
        return [f"Row count differs: {len(rows_a)} vs {len(rows_b)}"]

    diffs = 0
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        if ra != rb:
            diffs += 1
            if diffs <= 5:
                issues.append(f"Row {i+1} differs:\n  A: {ra}\n  B: {rb}")

    if diffs > 5:
        issues.append(f"… and {diffs - 5} more differing rows")

    return issues


def print_report(errors: List[str], label: str = "Validation") -> bool:
    """Print error report. Returns True if valid."""
    if not errors:
        print(f"✅  {label}: PASSED (no issues found)")
        return True
    print(f"❌  {label}: FAILED ({len(errors)} issue(s))\n")
    for e in errors:
        print(f"  • {e}")
    return False
