#!/bin/bash
#SBATCH --job-name=diff_v0709_sample_neg
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=diff_v0709_sample_neg_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0709_diffusion_sampler.py"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_VF/10000_diffusion_condLabelVF_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024.npy"
META_PATH="${DATA_FILE%.npy}_meta.npz"

CHECKPOINT="/xdisk/hdb/emcdugald/checkpoints/diffusion_VF/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel-low_mass_vf-append_to_condition_subset-all_mode-X0_epochs-1000_bs-4_c1-16_c2-32_c3-64_fine_all_vf_poc_20260713-232710/checkpoints/ckpt_best.pth"
OUT_ROOT="/xdisk/hdb/emcdugald/checkpoints/diffusion_VF/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel-low_mass_vf-append_to_condition_subset-all_mode-X0_epochs-1000_bs-4_c1-16_c2-32_c3-64_fine_all_vf_poc_20260713-232710"
OUTDIR="$OUT_ROOT/negative_4cond_x4samples"

DEVICE="cuda"

IMG_SIZE=32
UNET_CH1=16
UNET_CH2=32
UNET_CH3=64
T_EMBED_DIM=64
COND_EMBED_DIM=64
COND_CH=8
MODE="X0"

SAMPLE_ATOL=1e-4
SAMPLE_RTOL=1e-4
SAMPLE_EPS=1e-3

N_RANDOM=4
N_PER_COND=4
RNG_SEED=43
SUBSET="negative_only"

mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "meta path: $META_PATH"
echo "checkpoint: $CHECKPOINT"
echo "outdir: $OUTDIR"
echo "subset: $SUBSET"

$PYTHON "$SCRIPT" \
  --data-path "$DATA_FILE" \
  --meta-path "$META_PATH" \
  --checkpoint-path "$CHECKPOINT" \
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