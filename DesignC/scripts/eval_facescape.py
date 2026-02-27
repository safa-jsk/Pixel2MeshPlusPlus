#!/usr/bin/env python3
"""
Design C — FaceScape Evaluation (STUB)

Compute quality metrics (CD, F1@tau) on FaceScape predictions.

Usage:
    python eval_facescape.py \\
        --facescape_root /data/FaceScape \\
        --splits_csv     splits/test.csv \\
        --pred_dir       ../../artifacts/outputs/designC
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Design C: FaceScape evaluation metrics"
    )
    parser.add_argument(
        "--facescape_root", type=str, required=True,
        help="Root directory of FaceScape dataset (for ground-truth meshes)"
    )
    parser.add_argument(
        "--splits_csv", type=str, required=True,
        help="CSV file listing test samples"
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory containing predicted meshes from infer_facescape.py"
    )
    parser.add_argument(
        "--tau", type=float, default=0.0001,
        help="F-score threshold"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Design C — FaceScape Evaluation")
    print("=" * 60)
    print(f"  facescape_root : {args.facescape_root}")
    print(f"  splits_csv     : {args.splits_csv}")
    print(f"  pred_dir       : {args.pred_dir}")
    print(f"  tau            : {args.tau}")
    print()
    print("ERROR: Design C evaluation is not yet implemented.")
    print("       This is a skeleton script for future development.")
    sys.exit(1)


if __name__ == "__main__":
    main()
