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
    parser.add_argument("--data", type=str, default="data/TSB-AD-U", help="Path to CSV file, directory, or comma-separated dirs")
    parser.add_argument("--output", type=str, default="results", help="Directory to save scores and metrics")
    parser.add_argument("--download", type=str, choices=["TSB-AD-U", "TSB-AD-M"], help="Download dataset from TSB-AD")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process (default: all)")
    parser.add_argument("--save_scores", action="store_true", help="Save individual point-wise anomaly score CSVs")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch length L")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride s")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--loss", type=str, default="triplet", choices=["triplet", "contrastive"], help="Metric learning loss")
    parser.add_argument("--iters", type=int, default=30, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k nearest memory prototypes for anomaly score")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', or None for auto)")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    return parser.parse_args()


def collect_csv_files(data_arg: str) -> list[Path]:
    paths = [p.strip() for p in data_arg.split(",")]
    files = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_file() and p.suffix == ".csv":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.csv")))
    return files


def main():
    args = parse_args()

    if args.download:
        download_tsb_ad(args.download, target_dir="data")
        if not args.data or args.data == "data/TSB-AD-U":
            args.data = f"data/{args.download}"

    csv_files = collect_csv_files(args.data)
    if not csv_files:
        print(f"Error: No CSV files found for input: {args.data}", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        csv_files = csv_files[:args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_metrics.csv"

    scores_dir = None
    if args.save_scores:
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f" Anomlytic: Running Benchmark")
    print(f" Total files to evaluate: {len(csv_files)}")
    print(f" Config: patch_size={args.patch_size}, iters={args.iters}, loss={args.loss}")
    print(f" Saving results incrementally to: {summary_path}")
    print(f"==================================================\n")

    summary_records = []
    total_start = time.time()

    for idx, f in enumerate(csv_files, 1):
        try:
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

            # Incremental append to summary CSV so progress is never lost
            row_df = pd.DataFrame([metrics])
            write_header = not summary_path.exists() or idx == 1
            row_df.to_csv(summary_path, mode="w" if idx == 1 else "a", header=write_header, index=False)

            if scores_dir:
                pd.DataFrame({"True_Label": full_y, "Anomaly_Score": scores}).to_csv(
                    scores_dir / f"{f.stem}_scores.csv", index=False
                )

            print(
                f"[{idx}/{len(csv_files)}] {f.name[:45]:<45} | "
                f"AUC-ROC: {metrics['AUC-ROC']:.4f} | AUC-PR: {metrics['AUC-PR']:.4f} | "
                f"Best-F1: {metrics['Best-F1']:.4f} ({elapsed:.1f}s)"
            )

            # Periodic running average every 25 files
            if idx % 25 == 0:
                current_df = pd.DataFrame(summary_records)
                print(
                    f"  >> Running Mean (first {idx} files) -> "
                    f"AUC-ROC: {current_df['AUC-ROC'].mean():.4f} | "
                    f"AUC-PR: {current_df['AUC-PR'].mean():.4f} | "
                    f"Best-F1: {current_df['Best-F1'].mean():.4f}\n"
                )

        except Exception as e:
            print(f"[{idx}/{len(csv_files)}] {f.name} FAILED: {e}", file=sys.stderr)

    if summary_records:
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
        summary_df.to_csv(summary_path, index=False)

        total_elapsed = time.time() - total_start
        print(f"\n==================================================")
        print(f" Benchmark Complete! ({len(summary_records)} files processed)")
        print(f"   Overall Mean AUC-ROC: {avg_row['AUC-ROC']:.4f}")
        print(f"   Overall Mean AUC-PR:  {avg_row['AUC-PR']:.4f}")
        print(f"   Overall Mean Best-F1: {avg_row['Best-F1']:.4f}")
        print(f" Total duration: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
        print(f" Saved to: {summary_path}")
        print(f"==================================================\n")


if __name__ == "__main__":
    main()
