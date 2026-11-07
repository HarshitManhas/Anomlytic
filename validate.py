import os
import sys
import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore")

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anomlytic import AnomlyticDetector, load_tsb_ad_file


def run_validation(file_path: str, output_name: str, patch_size: int = 64, iters: int = 50):
    print(f"\n=======================================================")
    print(f" Validating Baseline on: {Path(file_path).name}")
    print(f"=======================================================")

    # 1. Load data and split
    train_x, train_y, test_x, test_y = load_tsb_ad_file(file_path)
    n_train = len(train_x)
    split_idx = int(n_train * 0.8)
    tr_split_x = train_x[:split_idx]
    val_split_x = train_x[split_idx:]

    print(f"Dataset summary:")
    print(f"  - Full series length: {len(train_x) + len(test_x)} (Channels: {train_x.shape[1]})")
    print(f"  - Normal Train split: {len(tr_split_x)} points")
    print(f"  - Normal Val split:   {len(val_split_x)} points")
    print(f"  - Unseen Test split:  {len(test_x)} points (Anomalous points: {int(test_y.sum())})")

    # 2. Checkpoint & Detector setup
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"best_{output_name}.pt"

    detector = AnomlyticDetector(
        patch_size=patch_size,
        num_iters=iters,
        lr=1e-3,
        batch_size=128,
        device="cuda",
    )

    # 3. Train with train/validation loss recording and checkpointing
    print("\n[1/5] Training model with metric learning...")
    detector.fit(
        train_data=tr_split_x,
        val_data=val_split_x,
        checkpoint_path=ckpt_path,
        verbose=False,
    )
    print(f"[PASS] Training completed successfully! Checkpoint saved to: {ckpt_path}")
    print(f"  - Initial Train Loss: {detector.history['train_loss'][0]:.4f} -> Final: {detector.history['train_loss'][-1]:.4f}")
    print(f"  - Initial Val Loss:   {detector.history['val_loss'][0]:.4f} -> Final: {detector.history['val_loss'][-1]:.4f}")

    # 4. Load best checkpoint to ensure model persistence works
    print("\n[2/5] Loading saved checkpoint for evaluation...")
    detector.load_checkpoint(ckpt_path)
    print("[PASS] Checkpoint loaded and verified.")

    # 5. Inference profiling on unseen test data
    print("\n[3/5] Evaluating on unseen test data & profiling hardware...")
    test_scores, profile = detector.profile_inference(test_x)

    print(f"[PASS] Inference profiled:")
    print(f"  - Total test latency:      {profile['Total_Latency_ms']:.2f} ms")
    print(f"  - Per-point latency:       {profile['Latency_us_per_point']:.2f} us / point")
    print(f"  - Peak GPU VRAM allocated: {profile['Peak_GPU_Memory_MB']:.2f} MB")

    # 6. Metric calculations
    print("\n[4/5] Calculating detection metrics...")
    metrics = detector.evaluate(test_scores, test_y)

    print(f"  - Precision: {metrics['Precision']:.4f}")
    print(f"  - Recall:    {metrics['Recall']:.4f}")
    print(f"  - F1-score:  {metrics['F1-score']:.4f}")
    print(f"  - PR-AUC:    {metrics['PR-AUC']:.4f}")
    print(f"  - ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"  - Threshold: {metrics['Threshold']:.4f}")

    # 7. Generate comparison plot
    print("\n[5/5] Generating validation plots...")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"validation_{output_name}.png"

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(f"Anomlytic Validation Baseline: {Path(file_path).name}", fontsize=14, fontweight="bold")

    # Panel 1: Loss curves
    steps = detector.history["step"]
    axes[0].plot(steps, detector.history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(steps, detector.history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_ylabel("Metric Loss")
    axes[0].set_xlabel("Iteration Step")
    axes[0].set_title("Training & Validation Loss Convergence")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Unseen test time series with ground-truth anomaly regions
    t_axis = np.arange(len(test_x))
    sig = test_x[:, 0] if test_x.ndim > 1 else test_x
    axes[1].plot(t_axis, sig, color="#2ca02c", lw=1.2, label="Unseen Test Signal (Channel 0)")
    # Highlight actual anomaly regions
    anom_indices = np.where(test_y == 1)[0]
    if len(anom_indices) > 0:
        axes[1].scatter(anom_indices, sig[anom_indices], color="#d62728", s=10, label="Ground Truth Anomalies", zorder=3)
    axes[1].set_ylabel("Standardized Value")
    axes[1].set_title(f"Unseen Test Series with Ground Truth Anomalies (T={len(test_x)})")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Predicted anomaly score vs threshold
    axes[2].plot(t_axis, test_scores, color="#9467bd", lw=1.5, label="Anomlytic Anomaly Score")
    axes[2].axhline(metrics["Threshold"], color="#d62728", linestyle="--", lw=1.5, label=f"Optimal F1 Threshold ({metrics['Threshold']:.3f})")
    pred_anom = np.where(test_scores >= metrics["Threshold"])[0]
    if len(pred_anom) > 0:
        axes[2].scatter(pred_anom, test_scores[pred_anom], color="#d62728", s=8, alpha=0.6, label="Predicted Anomalies")
    axes[2].set_ylabel("Anomaly Score [0, 1]")
    axes[2].set_xlabel("Test Time Index")
    axes[2].set_title(f"Predicted Scores (ROC-AUC: {metrics['ROC-AUC']:.4f}, Best-F1: {metrics['F1-score']:.4f})")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[PASS] Plot saved to: {plot_path}")

    return {
        "file": Path(file_path).name,
        "metrics": metrics,
        "profile": profile,
        "plot_path": plot_path,
        "ckpt_path": ckpt_path,
    }


def main():
    uni_path = "data/TSB-AD-U/013_NAB_id_13_Traffic_tr_623_1st_2084.csv"
    mul_path = "data/TSB-AD-M/169_SMAP_id_26_Sensor_tr_1811_1st_4510.csv"

    results_uni = run_validation(uni_path, output_name="traffic_univariate", patch_size=64, iters=50)
    results_mul = run_validation(mul_path, output_name="smap_multivariate", patch_size=64, iters=50)

    print("\n=======================================================")
    print(" BASELINE VALIDATION SUMMARY REPORT")
    print("=======================================================")
    for res in [results_uni, results_mul]:
        m = res["metrics"]
        p = res["profile"]
        print(f"\nTarget Dataset: {res['file']}")
        print(f"  - Precision:             {m['Precision']:.4f}")
        print(f"  - Recall:                {m['Recall']:.4f}")
        print(f"  - F1-score:              {m['F1-score']:.4f}")
        print(f"  - PR-AUC:                {m['PR-AUC']:.4f}")
        print(f"  - ROC-AUC:               {m['ROC-AUC']:.4f}")
        print(f"  - Optimal Threshold:     {m['Threshold']:.4f}")
        print(f"  - Inference Latency:     {p['Total_Latency_ms']:.2f} ms ({p['Latency_us_per_point']:.2f} us/point)")
        print(f"  - Peak GPU VRAM:         {p['Peak_GPU_Memory_MB']:.2f} MB")
        print(f"  - Checkpoint Saved:      {res['ckpt_path']}")
        print(f"  - Visualization Plot:    {res['plot_path']}")


if __name__ == "__main__":
    main()
