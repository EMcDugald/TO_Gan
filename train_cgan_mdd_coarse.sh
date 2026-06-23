#!/bin/bash
#SBATCH --job-name=cgan_mdd_coarse_v2
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=cgan_mdd_coarse_v2_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

# -------------------------
# Paths and dataset choice
# -------------------------
# Choose either fine or coarse dataset here:
DATA_FILE="/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/coarse/7500_labeled_voxels_32x32x32_coarse_bcLoadOnly_massLabel_low_mass_thr0.494781.npy"

CHECKPOINT_ROOT="/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_coarse_v2"
SCRIPT="$HOME/TO_Gan/PoC_3d/train_GAN_MDD_323232.py"

TAG="coarse_run"   # e.g. fine_run, coarse_run, highmass_run
DEVICE="cuda"

BATCH_SIZE=16
N_EPOCHS=1000
NZ=512
NGF=256
NDF=64
LR_D=1e-4
LR_G=1e-4
SMOOTH_REAL=0.1
D_EVERY=8
CKPT_EVERY_EPOCHS=10
EVAL_EVERY_EPOCHS=10
N_VIS_SAMPLES=4

# -------------------------
# Sanity prints
# -------------------------
echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "data file: $DATA_FILE"
echo "checkpoint root: $CHECKPOINT_ROOT"
echo "script: $SCRIPT"
echo "tag: $TAG"

# -------------------------
# Launch training
# -------------------------
$PYTHON "$SCRIPT" \
  --data-path "$DATA_FILE" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --device "$DEVICE" \
  --batch-size $BATCH_SIZE \
  --num-epochs $N_EPOCHS \
  --nz $NZ \
  --ngf $NGF \
  --ndf $NDF \
  --lr-d $LR_D \
  --lr-g $LR_G \
  --smooth-real $SMOOTH_REAL \
  --d-every $D_EVERY \
  --use-label-smoothing \
  --use-diversity-loss \
  --diversity-weight 0.01 \
  --n-vis-samples $N_VIS_SAMPLES \
  --ckpt-every-epochs $CKPT_EVERY_EPOCHS \
  --eval-every-epochs $EVAL_EVERY_EPOCHS \
  --tag "$TAG"