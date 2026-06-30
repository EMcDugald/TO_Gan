#!/bin/bash
#SBATCH --job-name=v0627_gan_mdd_coarse
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=v0627_gan_mdd_coarse_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

# -------------------------
# Paths
# -------------------------
DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy"
CHECKPOINT_ROOT="/xdisk/hdb/emcdugald/v0627/checkpoints/gan"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_gan_mdd_trainer.py"

TAG="coarse_run"
DEVICE="cuda"

BATCH_SIZE=32
N_EPOCHS=150
NZ=512
NGF=256
NDF=128
LR_D=1e-4
LR_G=1e-4
SMOOTH_REAL=0.07
D_EVERY=5
CKPT_EVERY_EPOCHS=10
EVAL_EVERY_EPOCHS=10
N_VIS_SAMPLES=3
DIVERSITY_WEIGHT=0.01

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
  --diversity-weight $DIVERSITY_WEIGHT \
  --n-vis-samples $N_VIS_SAMPLES \
  --ckpt-every-epochs $CKPT_EVERY_EPOCHS \
  --eval-every-epochs $EVAL_EVERY_EPOCHS \
  --tag "$TAG"