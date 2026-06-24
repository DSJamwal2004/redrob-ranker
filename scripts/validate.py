#!/usr/bin/env python3
"""
validate.py — standalone submission validator.

Usage:
    python scripts/validate.py submission/submission.csv
    python scripts/validate.py submission/submission.csv --candidates candidates.jsonl
    python scripts/validate.py submission/run1.csv --compare submission/run2.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_candidate_ids, load_all_candidates
from src.validation import check_determinism, print_report, validate_submission

logging.basicConfig(level=logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="Validate a Redrob submission CSV")
    p.add_argument("csv", help="Path to submission CSV")
    p.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl for ID existence check",
    )
    p.add_argument(
        "--compare",
        default=None,
        help="Second CSV to compare against (determinism check)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    dataset_ids = None
    if args.candidates:
        print(f"Loading dataset IDs from {args.candidates} …")
        dataset_ids = get_candidate_ids(load_all_candidates(args.candidates))
        print(f"  {len(dataset_ids)} candidates in dataset")

    errors = validate_submission(args.csv, dataset_ids=dataset_ids)
    ok = print_report(errors, label=f"Validation of {args.csv}")

    if args.compare:
        print(f"\nDeterminism check: {args.csv} vs {args.compare}")
        diff_issues = check_determinism(args.csv, args.compare)
        if not diff_issues:
            print("✅  Outputs are identical (deterministic)")
        else:
            print(f"❌  Outputs differ ({len(diff_issues)} issue(s))")
            for issue in diff_issues:
                print(f"  • {issue}")
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
