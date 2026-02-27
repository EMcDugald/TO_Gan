import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Hardcode a conditional-GAN run here if you like
#HARD_CODED_DIR = "/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_10k/epochs2000_bs8_nz200_ngf128_ndf128_nsamp10000_20260212-XXXXXX"
HARD_CODED_DIR = "/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_10k/epochs1000_bs32_nz200_ngf128_ndf128_nsamp5000_20260219-155218"

def parse_steps_per_epoch(dirname):
    m_bs = re.search(r'bs(\d+)', dirname)
    m_ns = re.search(r'nsamp(\d+)', dirname)
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
    print("Usage: python plot_training_metrics_cond.py /path/to/checkpoint_dir")
    sys.exit(1)

metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
if not os.path.exists(metrics_path):
    print(f"No metrics.csv found at {metrics_path}")
    sys.exit(1)

steps_per_epoch, batch_size, n_samples = parse_steps_per_epoch(os.path.basename(checkpoint_dir))
if steps_per_epoch is None:
    print("Could not infer steps_per_epoch; please set it manually.")
    sys.exit(1)

num_epochs = 2000
expected_num_steps = num_epochs * steps_per_epoch
print(f"Detected: batch_size={batch_size}, n_samples={n_samples}")
print(f"steps_per_epoch = {steps_per_epoch}")
print(f"expected_num_steps = {expected_num_steps}")

df = pd.read_csv(metrics_path).dropna()
actual_num_steps = len(df)
print(f"Actual steps in metrics.csv: {actual_num_steps}")

df["epoch"] = df["step"] / steps_per_epoch

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Conditional GAN Training Metrics", fontsize=16)

axes[0,0].plot(df["step"], df["L_D_real"], label="D_real", alpha=0.8)
axes[0,0].plot(df["step"], df["L_D_neg"],  label="D_neg",  alpha=0.8)
axes[0,0].plot(df["step"], df["L_D_fake"], label="D_fake", alpha=0.8)
axes[0,0].set_xlabel("Step")
axes[0,0].set_ylabel("Loss")
axes[0,0].legend()
axes[0,0].set_title("Losses vs Step")
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(df["step"], df["L_G"], "g-", linewidth=2, label="Generator")
axes[0,1].set_xlabel("Step")
axes[0,1].set_ylabel("Generator Loss")
axes[0,1].set_title("Generator Loss vs Step")
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(df["epoch"], df["L_D_real"], label="D_real", alpha=0.8)
axes[1,0].plot(df["epoch"], df["L_D_neg"],  label="D_neg",  alpha=0.8)
axes[1,0].plot(df["epoch"], df["L_D_fake"], label="D_fake", alpha=0.8)
axes[1,0].set_xlabel("Epoch")
axes[1,0].set_ylabel("Loss")
axes[1,0].legend()
axes[1,0].set_title("Losses vs Epoch")
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(df["epoch"], df["L_G"], "g-", linewidth=2, label="Generator")
axes[1,1].set_xlabel("Epoch")
axes[1,1].set_ylabel("Generator Loss")
axes[1,1].set_title("Generator Loss vs Epoch")
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(checkpoint_dir, "training_metrics.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plots to {output_path}")
