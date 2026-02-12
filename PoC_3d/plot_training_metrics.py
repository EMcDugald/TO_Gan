
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Option 1: Hardcode your directory here (no CLI args needed)
#HARD_CODED_DIR = "/xdisk/hdb/emcdugald/to_gan/checkpoints_323232_10k/epochs2000_bs8_nz200_ngf96_ndf96_nsamp2245_20251210-155524"
HARD_CODED_DIR = "/xdisk/hdb/emcdugald/to_gan/checkpoints_323232_10k/epochs2000_bs8_nz200_ngf128_ndf128_nsamp2208_thr1.0e-06_20260203-151724"


def parse_steps_per_epoch(dirname):
    """Extract batch_size and n_samples from directory name."""
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

# Get checkpoint directory
if len(sys.argv) == 2:
    checkpoint_dir = sys.argv[1]
elif HARD_CODED_DIR and os.path.exists(HARD_CODED_DIR):
    checkpoint_dir = HARD_CODED_DIR
    print(f"Using hardcoded directory: {checkpoint_dir}")
else:
    print("Usage: python plot_training_metrics.py /path/to/checkpoint_dir")
    print("Or set HARD_CODED_DIR variable inside script.")
    sys.exit(1)

metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
if not os.path.exists(metrics_path):
    print(f"No metrics.csv found at {metrics_path}")
    sys.exit(1)

# Parse steps_per_epoch from directory name
steps_per_epoch, batch_size, n_samples = parse_steps_per_epoch(os.path.basename(checkpoint_dir))
if steps_per_epoch is None:
    steps_per_epoch = 280  # fallback for your specific run (2245//8)
    print(f"Using fallback steps_per_epoch={steps_per_epoch}")

# Compute total expected steps (for verification)
num_epochs = 2000  # from your dir name
expected_num_steps = num_epochs * steps_per_epoch
print(f"Detected: batch_size={batch_size}, n_samples={n_samples}")
print(f"steps_per_epoch = {steps_per_epoch}")
print(f"expected_num_steps = {expected_num_steps}")

# Load metrics
df = pd.read_csv(metrics_path)
df = df.dropna()
actual_num_steps = len(df)
print(f"Actual steps in metrics.csv: {actual_num_steps}")

# Add epoch column
df['epoch'] = df['step'] / steps_per_epoch

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('GAN Training Metrics', fontsize=16)

# Loss vs step
axes[0,0].plot(df['step'], df['L_D_real'], label='D_real', alpha=0.8)
axes[0,0].plot(df['step'], df['L_D_neg'], label='D_neg', alpha=0.8)
axes[0,0].plot(df['step'], df['L_D_fake'], label='D_fake', alpha=0.8)
axes[0,0].set_xlabel('Step')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].set_title('Losses vs Step')
axes[0,0].grid(True, alpha=0.3)

# Generator loss vs step
axes[0,1].plot(df['step'], df['L_G'], 'g-', linewidth=2, label='Generator')
axes[0,1].set_xlabel('Step')
axes[0,1].set_ylabel('Generator Loss')
axes[0,1].set_title('Generator Loss vs Step')
axes[0,1].grid(True, alpha=0.3)

# Losses vs epoch
axes[1,0].plot(df['epoch'], df['L_D_real'], label='D_real', alpha=0.8)
axes[1,0].plot(df['epoch'], df['L_D_neg'], label='D_neg', alpha=0.8)
axes[1,0].plot(df['epoch'], df['L_D_fake'], label='D_fake', alpha=0.8)
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Loss')
axes[1,0].legend()
axes[1,0].set_title('Losses vs Epoch')
axes[1,0].grid(True, alpha=0.3)

# Generator loss vs epoch
axes[1,1].plot(df['epoch'], df['L_G'], 'g-', linewidth=2, label='Generator')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Generator Loss')
axes[1,1].set_title('Generator Loss vs Epoch')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(checkpoint_dir, 'training_metrics.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved plots to {output_path}")
