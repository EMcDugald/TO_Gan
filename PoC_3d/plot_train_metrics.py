import sys
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Optional hard-coded run (used if no CLI args)
HARD_CODED_DIRS = [
    # Example:
    # "/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232/octant/...",
]


def parse_steps_per_epoch(dirname):
    """
    Parse 'bsX' and 'nsampY' from a checkpoint directory name and
    infer steps_per_epoch = nsamp / bs.
    """
    m_bs = re.search(r"bs(\d+)", dirname)
    m_ns = re.search(r"nsamp(\d+)", dirname)
    if m_bs and m_ns:
        batch_size = int(m_bs.group(1))
        n_samples = int(m_ns.group(1))
        steps_per_epoch = n_samples // batch_size
        return steps_per_epoch, batch_size, n_samples
    else:
        print(f"Warning: Could not parse 'bsX' or 'nsampX' from '{dirname}'")
        return None, None, None


def load_metrics(checkpoint_dir):
    """
    Load metrics.csv from a checkpoint directory and ensure
    all expected columns exist with numeric dtype.
    """
    metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.csv found at {metrics_path}")

    dirname = os.path.basename(checkpoint_dir)
    steps_per_epoch, batch_size, n_samples = parse_steps_per_epoch(dirname)
    if steps_per_epoch is None:
        raise RuntimeError("Could not infer steps_per_epoch from directory name.")

    print(f"[{dirname}] batch_size={batch_size}, n_samples={n_samples}, steps_per_epoch={steps_per_epoch}")

    df = pd.read_csv(metrics_path)
    print(f"[{dirname}] Columns in metrics.csv: {list(df.columns)}")
    print(f"[{dirname}] Rows in metrics.csv: {len(df)}")

    # Basic numeric columns
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        df["epoch"] = df["step"] / steps_per_epoch

    # Ensure all expected metric columns exist
    expected_cols = [
        "L_D_real", "L_D_neg", "L_D_fake",
        "L_G",
        "D_grad_norm", "G_grad_norm",
        "L_div",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, steps_per_epoch


def plot_single_run(df, checkpoint_dir):
    """
    Plot D/G losses (and optional diversity) for a single run.
    """
    run_name = os.path.basename(checkpoint_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"GAN Training Metrics: {run_name}", fontsize=14)

    # Top-left: D losses vs step
    ax = axes[0, 0]
    if not df["L_D_real"].isna().all():
        ax.plot(df["step"], df["L_D_real"], label="D_real", alpha=0.8)
    if not df["L_D_neg"].isna().all():
        ax.plot(df["step"], df["L_D_neg"], label="D_neg", alpha=0.8)
    if not df["L_D_fake"].isna().all():
        ax.plot(df["step"], df["L_D_fake"], label="D_fake", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("D Losses vs Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-right: G loss vs step
    ax = axes[0, 1]
    ax.plot(df["step"], df["L_G"], "g-", linewidth=2, label="G")
    ax.set_xlabel("Step")
    ax.set_ylabel("Generator Loss")
    ax.set_title("G Loss vs Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Bottom-left: D losses vs epoch
    ax = axes[1, 0]
    if not df["L_D_real"].isna().all():
        ax.plot(df["epoch"], df["L_D_real"], label="D_real", alpha=0.8)
    if not df["L_D_neg"].isna().all():
        ax.plot(df["epoch"], df["L_D_neg"], label="D_neg", alpha=0.8)
    if not df["L_D_fake"].isna().all():
        ax.plot(df["epoch"], df["L_D_fake"], label="D_fake", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("D Losses vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Bottom-right: G loss vs epoch (and optional L_div)
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["L_G"], "g-", linewidth=2, label="G")
    has_div = not df["L_div"].isna().all()
    if has_div:
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["L_div"], "r--", alpha=0.6, label="L_div")
        ax2.set_ylabel("Diversity Loss")
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generator Loss")
    ax.set_title("G Loss vs Epoch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(checkpoint_dir, "training_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[{run_name}] Saved single-run plots to {out_path}")


def plot_overlay(runs):
    """
    Given a list of (df, checkpoint_dir), plot overlayed L_G vs epoch
    to compare objectives (e.g., DO vs MDD).
    """
    if len(runs) < 2:
        return

    plt.figure(figsize=(8, 6))
    for df, ckpt in runs:
        name = os.path.basename(ckpt)
        plt.plot(df["epoch"], df["L_G"], linewidth=1.5, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")
    plt.title("G Loss vs Epoch (Overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    # Save to first checkpoint's parent dir if they share a parent,
    # otherwise just to current working dir.
    out_path = os.path.join(os.getcwd(), "training_metrics_overlay.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved overlay plot to {out_path}")


def main():
    # Resolve checkpoint directories from CLI or hard-coded list
    if len(sys.argv) > 1:
        checkpoint_dirs = [d for d in sys.argv[1:] if os.path.isdir(d)]
    elif HARD_CODED_DIRS:
        checkpoint_dirs = [d for d in HARD_CODED_DIRS if os.path.isdir(d)]
        if checkpoint_dirs:
            print("Using hard-coded directories:")
            for d in checkpoint_dirs:
                print("  ", d)
    else:
        print("Usage: python plot_training_metrics.py /path/to/ckpt1 [/path/to/ckpt2 ...]")
        sys.exit(1)

    if not checkpoint_dirs:
        print("No valid checkpoint directories provided.")
        sys.exit(1)

    runs = []
    for ckpt in checkpoint_dirs:
        try:
            df, _ = load_metrics(ckpt)
        except Exception as e:
            print(f"Error loading {ckpt}: {e}")
            continue
        plot_single_run(df, ckpt)
        runs.append((df, ckpt))

    # Optional overlay if multiple runs provided
    if len(runs) >= 2:
        plot_overlay(runs)


if __name__ == "__main__":
    main()