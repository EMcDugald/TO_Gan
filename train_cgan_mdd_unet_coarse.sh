#!/bin/bash
#SBATCH --job-name=cgan_mdd_unet_coarse_v2
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=cgan_mdd_unet_coarse_v2_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

# -------------------------
# Paths and dataset choice
# -------------------------
DATA_FILE="/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/coarse/7500_labeled_voxels_32x32x32_coarse_bcLoadOnly_massLabel_low_mass_thr0.494781.npy"

CKPT_ROOT="/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_coarse_unet_v2"
SCRIPT="/home/u26/emcdugald/TO_Gan/PoC_3d/train_gan_unet.py"

TAG="coarse_run"
DEVICE="cuda"

BATCH_SIZE=16
N_EPOCHS=1000
NZ=128
NGF=32
NDF=16
LR_D=1e-4
LR_G=1e-4
SMOOTH_REAL=0.1
D_EVERY=8
CKPT_EVERY_EPOCHS=5
EVAL_EVERY_EPOCHS=5
N_VIS_SAMPLES=5
DIVERSITY_WEIGHT=0.01

# -------------------------
# Sanity prints
# -------------------------
echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "data file: $DATA_FILE"
echo "checkpoint root: $CKPT_ROOT"
echo "script: $SCRIPT"
echo "tag: $TAG"

# -------------------------
# Launch training
# -------------------------
$PYTHON "$SCRIPT" \
  --data-path "$DATA_FILE" \
  --checkpoint-root "$CKPT_ROOT" \
  --device "$DEVICE" \
  --batch-size $BATCH_SIZE \
  --nz $NZ \
  --ngf $NGF \
  --ndf $NDF \
  --num-epochs $N_EPOCHS \
  --lr-d $LR_D \
  --lr-g $LR_G \
  --smooth-real $SMOOTH_REAL \
  --d-every $D_EVERY \
  --use-label-smoothing \
  --use-diversity-loss \
  --diversity-weight $DIVERSITY_WEIGHT \
  --n-vis-samples $N_VIS_SAMPLES \
  --ckpt-every-epochs $CKPT_EVERY_EPOCHS \
  --eval-every-epochs $EVAL_EVERY_EPOCHS \
  --tag "$TAG"

# -------------------------
# Example resume usage
# -------------------------
# $PYTHON "$SCRIPT" \
#   --data-path "$DATA_FILE" \
#   --checkpoint-root "$CKPT_ROOT" \
#   --resume-path /path/to/ckpt_step_5000.pt \
#   --device "$DEVICE" \
#   --batch-size $BATCH_SIZE \
#   --nz $NZ \
#   --ngf $NGF \
#   --ndf $NDF \
#   --num-epochs $N_EPOCHS \
#   --lr-d $LR_D \
#   --lr-g $LR_G \
#   --smooth-real $SMOOTH_REAL \
#   --d-every $D_EVERY \
#   --use-label-smoothing \
#   --use-diversity-loss \
#   --diversity-weight $DIVERSITY_WEIGHT \
#   --n-vis-samples $N_VIS_SAMPLES \
#   --ckpt-every-epochs $CKPT_EVERY_EPOCHS \
#   --eval-every-epochs $EVAL_EVERY_EPOCHS \
#   --tag "${TAG}_resume"