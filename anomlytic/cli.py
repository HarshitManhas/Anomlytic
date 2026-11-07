import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore")

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .detector import AnomlyticDetector
from .dataset import load_tsb_ad_file, download_tsb_ad


def parse_args():
    parser = argparse.ArgumentParser(
        description="Anomlytic: 1D CNN + Patch Representation Learning for Time-Series Anomaly Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="data/TSB-AD-U", help="Path to CSV file or directory of CSVs")
    parser.add_argument("--output", type=str, default="results", help="Directory to save scores and metrics")
    parser.add_argument("--download", type=str, choices=["TSB-AD-U", "TSB-AD-M"], help="Download dataset from TSB-AD")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch length L")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride s")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--loss", type=str, default="triplet", choices=["triplet", "contrastive"], help="Metric learning loss")
    parser.add_argument("--iters", type=int, default=100, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k nearest memory prototypes for anomaly score")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', or None for auto)")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download:
        download_tsb_ad(args.download, target_dir="data")
        if not args.data or args.data == "data/TSB-AD-U":
            args.data = f"data/{args.download}"

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    csv_files = [data_path] if data_path.is_file() else sorted(list(data_path.glob("*.csv")))
    if not csv_files:
        print(f"Error: No CSV files found in {data_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f" Anomlytic: Running Anomaly Detection")
    print(f" Target: {data_path} ({len(csv_files)} files)")
    print(f" Config: patch_size={args.patch_size}, iters={args.iters}, loss={args.loss}")
    print(f"==================================================\n")

    summary_records = []
    total_start = time.time()

    for idx, f in enumerate(csv_files, 1):
        print(f"[{idx}/{len(csv_files)}] Processing: {f.name}")
        train_x, train_y, test_x, test_y = load_tsb_ad_file(f)
        full_x = np.concatenate([train_x, test_x], axis=0)
        full_y = np.concatenate([train_y, test_y], axis=0)

        detector = AnomlyticDetector(
            patch_size=args.patch_size,
            stride=args.stride,
            embed_dim=args.embed_dim,
            loss_type=args.loss,
            num_iters=args.iters,
            lr=args.lr,
            batch_size=args.batch_size,
            top_k=args.top_k,
            device=args.device,
        )

        t0 = time.time()
        detector.fit(train_x, verbose=args.verbose)
        scores = detector.predict_score(full_x)
        elapsed = time.time() - t0

        metrics = detector.evaluate(scores, full_y)
        metrics["File"] = f.name
        metrics["Duration(s)"] = round(elapsed, 2)
        summary_records.append(metrics)

        # Save scores
        pd.DataFrame({"True_Label": full_y, "Anomaly_Score": scores}).to_csv(
            scores_dir / f"{f.stem}_scores.csv", index=False
        )

        print(
            f"       AUC-ROC: {metrics['AUC-ROC']:.4f} | AUC-PR: {metrics['AUC-PR']:.4f} | "
            f"Best-F1: {metrics['Best-F1']:.4f} ({elapsed:.1f}s)"
        )

    summary_df = pd.DataFrame(summary_records)
    avg_row = {
        "File": "AVERAGE",
        "AUC-ROC": round(summary_df["AUC-ROC"].mean(), 4),
        "AUC-PR": round(summary_df["AUC-PR"].mean(), 4),
        "Best-F1": round(summary_df["Best-F1"].mean(), 4),
        "Precision": round(summary_df["Precision"].mean(), 4),
        "Recall": round(summary_df["Recall"].mean(), 4),
        "Duration(s)": round(summary_df["Duration(s)"].sum(), 2),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    total_elapsed = time.time() - total_start
    print(f"\n==================================================")
    print(f" Summary Results (Mean across {len(csv_files)} files):")
    print(f"   AUC-ROC: {avg_row['AUC-ROC']:.4f}")
    print(f"   AUC-PR:  {avg_row['AUC-PR']:.4f}")
    print(f"   Best-F1: {avg_row['Best-F1']:.4f}")
    print(f" Saved to: {out_dir / 'summary_metrics.csv'} in {total_elapsed:.1f}s")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()
