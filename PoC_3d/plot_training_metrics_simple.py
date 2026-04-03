import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional hard-coded run
# HARD_CODED_DIR = "/xdisk/hdb/emcdugald/simple_gan/checkpoints_323232/uncond_epochs1000_bs32_nz300_ngf256_ndf64_nsamp15000_lrD0.0002_lrG0.0002_smoothR0.1_dEvery1_20260326-154500"
HARD_CODED_DIR = "/xdisk/hdb/emcdugald/simple_gan/checkpoints_323232/uncond_epochs1000_bs32_nz300_ngf256_ndf64_nsamp15000_lrD0.0002_lrG0.0002_smoothR0.1_dEvery1_20260326-161618"

def parse_steps_per_epoch(dirname):
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

if len(sys.argv) == 2:
    checkpoint_dir = sys.argv[1]
elif HARD_CODED_DIR and os.path.exists(HARD_CODED_DIR):
    checkpoint_dir = HARD_CODED_DIR
    print(f"Using hardcoded directory: {checkpoint_dir}")
else:
    print("Usage: python plot_training_metrics_uncond.py /path/to/checkpoint_dir")
    sys.exit(1)

metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
if not os.path.exists(metrics_path):
    print(f"No metrics.csv found at {metrics_path}")
    sys.exit(1)

steps_per_epoch, batch_size, n_samples = parse_steps_per_epoch(os.path.basename(checkpoint_dir))
if steps_per_epoch is None:
    print("Could not infer steps_per_epoch; please set it manually.")
    sys.exit(1)

print(f"Detected: batch_size={batch_size}, n_samples={n_samples}")
print(f"steps_per_epoch = {steps_per_epoch}")

df = pd.read_csv(metrics_path)
print(f"Columns in metrics.csv: {list(df.columns)}")
print(f"Actual rows in metrics.csv: {len(df)}")

df["step"] = pd.to_numeric(df["step"], errors="coerce")
if "epoch" in df.columns:
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
else:
    df["epoch"] = df["step"] / steps_per_epoch

loss_cols = ["L_D_real", "L_D_fake", "L_G", "D_grad_norm", "G_grad_norm"]
for c in loss_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Unconditional GAN Training Metrics", fontsize=16)

axes[0, 0].plot(df["step"], df["L_D_real"], label="D_real", alpha=0.8)
axes[0, 0].plot(df["step"], df["L_D_fake"], label="D_fake", alpha=0.8)
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].set_title("Discriminator Losses vs Step")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df["step"], df["L_G"], "g-", linewidth=2, label="Generator")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Generator Loss")
axes[0, 1].set_title("Generator Loss vs Step")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df["epoch"], df["L_D_real"], label="D_real", alpha=0.8)
axes[1, 0].plot(df["epoch"], df["L_D_fake"], label="D_fake", alpha=0.8)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()
axes[1, 0].set_title("Discriminator Losses vs Epoch")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df["epoch"], df["L_G"], "g-", linewidth=2, label="Generator")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Generator Loss")
axes[1, 1].set_title("Generator Loss vs Epoch")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(checkpoint_dir, "training_metrics.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plots to {output_path}")