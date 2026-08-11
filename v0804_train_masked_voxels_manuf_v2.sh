#!/bin/bash
#SBATCH --job-name=masked_vox_manuf
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=masked_vox_manuf_%j.out

# Trains one of the 4 manufacturability models with the UNCHANGED Route C
# trainer (v0804_masked_voxel_diff_trainer.py -- a pure rename of
# v0723_masked_voxel_diff_trainer.py, zero content changes; cond_dim=48 is
# picked up automatically from the data, same as cond_dim=46 always was).
#
# Usage: pass the model number (1-4) as the first argument. Every
# hyperparameter below can be overridden via environment variable at
# submission time WITHOUT editing this file or making copies of it, e.g.:
#
#   sbatch v0804_train_masked_voxels_manuf.sh 1
#   BATCHSIZE=16 LR=5e-5 sbatch v0804_train_masked_voxels_manuf.sh 1
#   TAG=model1_lowlr sbatch v0804_train_masked_voxels_manuf.sh 1
#
# Each call gets its own timestamped run directory automatically (the
# trainer hard-fails rather than overwriting if a dir somehow already
# exists), so repeated calls -- with the same or different hyperparameters
# -- never clobber each other. Defaults below match the proven 50k config
# exactly.
#
# Notes:
# * mem=32G is generous here: these files (5564-10564 rows) are much
#   smaller than the 50k run this config was sized for.
# * Dataset sizes are far smaller than 50k, so epochs will run much faster
#   in wall-clock time -- --time=24:00:00 is generous headroom, not a real
#   estimate; watch the first few epochs' timing and adjust NEPOCHS/
#   SAVE_EVERY/SAMPLE_EVERY if you want a shorter/longer run.
# * Largest padded shape is still 120x40x16 = 76.8k voxels regardless of
#   dataset row count -- if you OOM, drop BATCHSIZE.
# * Watch the in-training manifests: <run_dir>/samples_epoch_*/
#   sample_manifest.txt (mass/speckle/IoU -- manufacturability status does
#   NOT show in these titles, only in your standalone visualizer).
# * Resume: RESUME_CKPT=<run_dir>/checkpoints/ckpt_best.pth sbatch ... <id>

MODEL_ID="$1"
if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: sbatch v0804_train_masked_voxels_manuf.sh <1|2|3|4>"
  exit 1
fi

DATA_ROOT="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine_manuf"
SHAPES_SUFFIX="shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine"

case "$MODEL_ID" in
  1)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_1_scalar_labeledonly/5564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-scalar_vfmode-append_to_condition.npy"
    DEFAULT_TAG="v0804_manuf_model1_scalar_labeledonly"
    ;;
  2)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_2_binary_labeledonly/5564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-binary_vfmode-append_to_condition.npy"
    DEFAULT_TAG="v0804_manuf_model2_binary_labeledonly"
    ;;
  3)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_3_scalar_plus5000unlabeled/10564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-scalar_vfmode-append_to_condition.npy"
    DEFAULT_TAG="v0804_manuf_model3_scalar_plus5000unlabeled"
    ;;
  4)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_4_binary_plus5000unlabeled/10564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-binary_vfmode-append_to_condition.npy"
    DEFAULT_TAG="v0804_manuf_model4_binary_plus5000unlabeled"
    ;;
  *)
    echo "Unknown MODEL_ID: $MODEL_ID (expected 1, 2, 3, or 4)"
    exit 1
    ;;
esac

# Every value below is overridable via environment variable, e.g.
# `BATCHSIZE=16 sbatch v0804_train_masked_voxels_manuf.sh 1`
DATA_FILE="${DATA_FILE:-$DEFAULT_DATA_FILE}"
TAG="${TAG:-$DEFAULT_TAG}"

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
LOG_ROOT="${LOG_ROOT:-/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff}"
DEVICE="${DEVICE:-cuda}"

# ---- optimization (defaults match the proven 50k config) ----
BATCHSIZE="${BATCHSIZE:-32}"          # drop to 8 if OOM
NEPOCHS="${NEPOCHS:-5000}"
LR="${LR:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
COND_DROP_PROB="${COND_DROP_PROB:-0.1}"   # keep > 0 so CFG stays available at sampling time
LOSS_WEIGHTING="${LOSS_WEIGHTING:-Simple}"

# ---- model (defaults match the proven 50k config) ----
C1="${C1:-32}"
C2="${C2:-64}"
C3="${C3:-128}"
T_EMBED_DIM="${T_EMBED_DIM:-128}"
COND_EMBED_DIM="${COND_EMBED_DIM:-128}"
COND_CH="${COND_CH:-8}"

# ---- data ----
NSAMPLES="${NSAMPLES:-0}"             # 0 = the whole file
VAL_FRAC="${VAL_FRAC:-0.05}"
VAL_MAX_PARTS="${VAL_MAX_PARTS:-512}"

# ---- logging / in-training sampling ----
SAVE_EVERY="${SAVE_EVERY:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
SAMPLE_NUM="${SAMPLE_NUM:-2}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-2}"
SAMPLE_N_STEPS="${SAMPLE_N_STEPS:-100}"
SAMPLE_EPS="${SAMPLE_EPS:-1e-3}"
EMA_DECAY="${EMA_DECAY:-0.9999}"

NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-42}"
RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "model id: $MODEL_ID"
echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "tag: $TAG"
echo "batchsize=$BATCHSIZE lr=$LR nepochs=$NEPOCHS c1=$C1 c2=$C2 c3=$C3 cond_drop_prob=$COND_DROP_PROB"

RESUME_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
  RESUME_ARGS=(--load_ckpt "$RESUME_CKPT")
fi

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
  --tag "$TAG" \
  "${RESUME_ARGS[@]}"