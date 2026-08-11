#!/bin/bash
#SBATCH --job-name=masked_vox_manuf
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=masked_vox_manuf_%j.out

# Trains one of the 4 manufacturability models with the UNCHANGED Route C
# trainer (v0804_masked_voxel_diff_trainer.py -- a pure rename of
# v0723_masked_voxel_diff_trainer.py, zero content changes; cond_dim=48 is
# picked up automatically from the data, same as cond_dim=46 always was).
#
# Usage: pass the model number (1-4) as the first argument.
#   sbatch v0804_train_masked_voxels_manuf.sh 1
#   sbatch v0804_train_masked_voxels_manuf.sh 2
#   sbatch v0804_train_masked_voxels_manuf.sh 3
#   sbatch v0804_train_masked_voxels_manuf.sh 4
# (or use v0804_launch_all_manuf_training.sh to submit all 4 at once as
# independent jobs -- do NOT loop these sequentially in one job, each needs
# its own GPU allocation and they're meant to run in parallel.)
#
# Notes:
# * Hyperparameters below are unchanged from the proven 10k/50k config
#   (v0723_train_masked_voxels.sh) -- same lr, batchsize, U-Net width,
#   cond_drop_prob=0.1 (needed for CFG at sampling time).
# * mem=32G is generous here: these files (5564-10564 rows) are much
#   smaller than the 50k run this was sized for -- kept as-is for margin,
#   feel free to lower if you want the memory back for something else.
# * Dataset sizes are far smaller than 50k, so epochs will run much faster
#   in wall-clock time than the 50k run did -- --time=24:00:00 is generous
#   headroom, not a real estimate; watch the first few epochs' timing and
#   adjust NEPOCHS/SAVE_EVERY/SAMPLE_EVERY if you want a shorter/longer run.
# * Watch the in-training manifests: <run_dir>/samples_epoch_*/
#   sample_manifest.txt (mass/speckle/IoU, same as before -- manufacturability
#   status does NOT show in these titles, only in your standalone visualizer).
# * Resume: add --load_ckpt <run_dir>/checkpoints/ckpt_best.pth

MODEL_ID="$1"
if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: sbatch v0804_train_masked_voxels_manuf.sh <1|2|3|4>"
  exit 1
fi

DATA_ROOT="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine_manuf"
SHAPES_SUFFIX="shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine"

case "$MODEL_ID" in
  1)
    DATA_FILE="$DATA_ROOT/model_1_scalar_labeledonly/5564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-scalar_vfmode-append_to_condition.npy"
    TAG="v0804_model1_scalar_labeledonly"
    ;;
  2)
    DATA_FILE="$DATA_ROOT/model_2_binary_labeledonly/5564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-binary_vfmode-append_to_condition.npy"
    TAG="v0804_model2_binary_labeledonly"
    ;;
  3)
    DATA_FILE="$DATA_ROOT/model_3_scalar_plus5000unlabeled/10564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-scalar_vfmode-append_to_condition.npy"
    TAG="v0804_model3_scalar_plus5000unlabeled"
    ;;
  4)
    DATA_FILE="$DATA_ROOT/model_4_binary_plus5000unlabeled/10564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-binary_vfmode-append_to_condition.npy"
    TAG="v0804_model4_binary_plus5000unlabeled"
    ;;
  *)
    echo "Unknown MODEL_ID: $MODEL_ID (expected 1, 2, 3, or 4)"
    exit 1
    ;;
esac

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: data file not found: $DATA_FILE"
  echo "(check the actual filename under $DATA_ROOT/model_${MODEL_ID}_* -- "
  echo "the leading sample count is baked into the filename and must match exactly)"
  exit 1
fi

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0804_masked_voxel_diff_trainer.py"
LOG_ROOT="/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff"
DEVICE="cuda"

# ---- optimization (unchanged from the proven 10k/50k config) ----
BATCHSIZE=32          # drop to 8 if OOM
NEPOCHS=5000
LR=1e-4
GRAD_CLIP=1.0
COND_DROP_PROB=0.1    # keep > 0 so CFG stays available at sampling time
LOSS_WEIGHTING="Simple"

# ---- model (unchanged from the proven 10k/50k config) ----
C1=32
C2=64
C3=128
T_EMBED_DIM=128
COND_EMBED_DIM=128
COND_CH=8

# ---- data ----
NSAMPLES=0            # 0 = the whole file
VAL_FRAC=0.05
VAL_MAX_PARTS=512

# ---- logging / in-training sampling ----
SAVE_EVERY=5
SAMPLE_EVERY=5
SAMPLE_NUM=2
ENSEMBLE_SIZE=2
SAMPLE_N_STEPS=100
SAMPLE_EPS=1e-3
EMA_DECAY=0.9999

NUM_WORKERS=6
SEED=42

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "model id: $MODEL_ID"
echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "tag: $TAG"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed $SEED \
  --batchsize $BATCHSIZE \
  --nepochs $NEPOCHS \
  --lr $LR \
  --grad_clip $GRAD_CLIP \
  --cond_drop_prob $COND_DROP_PROB \
  --loss_weighting "$LOSS_WEIGHTING" \
  --unet_ch1_dim $C1 \
  --unet_ch2_dim $C2 \
  --unet_ch3_dim $C3 \
  --t_embed_dim $T_EMBED_DIM \
  --cond_embed_dim $COND_EMBED_DIM \
  --cond_ch $COND_CH \
  --data_file "$DATA_FILE" \
  --nsamples $NSAMPLES \
  --val_frac $VAL_FRAC \
  --val_max_parts $VAL_MAX_PARTS \
  --log_root "$LOG_ROOT" \
  --save_every_n_epochs $SAVE_EVERY \
  --sample_every_n_epochs $SAMPLE_EVERY \
  --sample_num $SAMPLE_NUM \
  --ensemble_size $ENSEMBLE_SIZE \
  --sample_n_steps $SAMPLE_N_STEPS \
  --sample_eps $SAMPLE_EPS \
  --ema_decay $EMA_DECAY \
  --num_workers $NUM_WORKERS \
  --tag "$TAG"
