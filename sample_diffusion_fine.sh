#!/bin/bash
#SBATCH --job-name=sample_diff_fine
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=sample_diff_fine_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/diffusion_sampler.py"
DEVICE="cuda"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024.npy"
META_FILE="/xdisk/hdb/emcdugald/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024_meta.npz"

CKPT="/xdisk/hdb/emcdugald/checkpoints/diffusion/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel-low_mass_mode-X0_epochs-300_bs-4_c1-32_c2-64_c3-128_20260706-184422/checkpoints/ckpt_best.pth"

OUTDIR="/xdisk/hdb/emcdugald/checkpoints/diffusion/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel-low_mass_mode-X0_epochs-300_bs-4_c1-32_c2-64_c3-128_20260706-184422/diffusion_fine_ckpt_best_4x4"

IMG_SIZE=32
UNET_CH1=32
UNET_CH2=64
UNET_CH3=128
T_EMBED_DIM=128
COND_EMBED_DIM=128
COND_CH=8
MODE="X0"
SAMPLE_ATOL=1e-4
SAMPLE_RTOL=1e-4
SAMPLE_EPS=1e-3

N_RANDOM=4
N_PER_COND=4
RNG_SEED=0
SUBSET="positive_only"

mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "meta file: $META_FILE"
echo "checkpoint: $CKPT"
echo "outdir: $OUTDIR"
echo "mode: diffusion fine"

$PYTHON "$SCRIPT" \
  --data-path "$DATA_FILE" \
  --meta-path "$META_FILE" \
  --checkpoint-path "$CKPT" \
  --outdir "$OUTDIR" \
  --img-size $IMG_SIZE \
  --unet-ch1-dim $UNET_CH1 \
  --unet-ch2-dim $UNET_CH2 \
  --unet-ch3-dim $UNET_CH3 \
  --t-embed-dim $T_EMBED_DIM \
  --cond-embed-dim $COND_EMBED_DIM \
  --cond-ch $COND_CH \
  --mode "$MODE" \
  --sample-atol $SAMPLE_ATOL \
  --sample-rtol $SAMPLE_RTOL \
  --sample-eps $SAMPLE_EPS \
  --n-random $N_RANDOM \
  --n-per-cond $N_PER_COND \
  --rng-seed $RNG_SEED \
  --subset "$SUBSET" \
  --device "$DEVICE"