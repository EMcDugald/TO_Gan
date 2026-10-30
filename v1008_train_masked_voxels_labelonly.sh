#!/bin/bash
#SBATCH --job-name=masked_vox_labelonly
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=masked_vox_labelonly_%j.out

# Trains the label-only (class-conditional) manufacturability generator with
# the UNCHANGED Route C trainer (v0804_masked_voxel_diff_trainer.py). The only
# difference from the earlier runs is the data file: its condition vector is
# [label, has_label] (cond_dim = 2), with no TO spec.
#
# Usage (every value is an env override; no editing or copies needed):
#   sbatch v1008_train_masked_voxels_labelonly.sh
#   MANUF_PCTL=50 sbatch v1008_train_masked_voxels_labelonly.sh     # train on the 50/50 split
#   NEPOCHS=3000 BATCHSIZE=16 sbatch v1008_train_masked_voxels_labelonly.sh
#   RESUME_CKPT=<run_dir>/checkpoints/ckpt_best.pth sbatch v1008_train_masked_voxels_labelonly.sh
#
# Notes:
# * COND_DROP_PROB must stay > 0: classifier-free guidance at sampling time is
#   the main lever for pushing the requested class fractions toward 100%/0%.
#   Dropped conditions become [0, 0], the same vector as "no label".
# * In-training sample plots show no BC/load markers (there are none now);
#   that is expected.
# * Run dirs land in $LOG_ROOT with "v1008_labelonly_binary_p<PCTL>" in the
#   name; the class-conditional sampler finds the newest one by that tag.

DATA_ROOT="${DATA_ROOT:-/xdisk/hdb/emcdugald/train_data/diffusion_labelonly}"
MANUF_PCTL="${MANUF_PCTL:-80}"
DATA_DIR="$DATA_ROOT/labelonly_binary_p${MANUF_PCTL}"

if [[ -z "$DATA_FILE" ]]; then
  DATA_FILE=$(ls "$DATA_DIR"/*_vfmode-omit.npy 2>/dev/null | head -n 1)
fi
if [[ -z "$DATA_FILE" || ! -f "$DATA_FILE" ]]; then
  echo "ERROR: no label-only data file found in $DATA_DIR"
  echo "(run v1008_make_train_data_labelonly.sh with MANUF_PCTL=$MANUF_PCTL first, or set DATA_FILE)"
  exit 1
fi
TAG="${TAG:-v1008_labelonly_binary_p${MANUF_PCTL}}"

module load cuda11
module load anaconda

PYTHON="${PYTHON:-$HOME/.conda/envs/to_gan/bin/python}"
SCRIPT="${SCRIPT:-$HOME/TO_Gan/PoC_3d/v0804_masked_voxel_diff_trainer.py}"
LOG_ROOT="${LOG_ROOT:-/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff}"
DEVICE="${DEVICE:-cuda}"

# ---- optimization (same proven defaults as the earlier runs) ----
BATCHSIZE="${BATCHSIZE:-32}"
NEPOCHS="${NEPOCHS:-5000}"
LR="${LR:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
COND_DROP_PROB="${COND_DROP_PROB:-0.1}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-Simple}"

# ---- model ----
C1="${C1:-32}"
C2="${C2:-64}"
C3="${C3:-128}"
T_EMBED_DIM="${T_EMBED_DIM:-128}"
COND_EMBED_DIM="${COND_EMBED_DIM:-128}"
COND_CH="${COND_CH:-8}"

# ---- data ----
NSAMPLES="${NSAMPLES:-0}"
VAL_FRAC="${VAL_FRAC:-0.05}"
VAL_MAX_PARTS="${VAL_MAX_PARTS:-512}"

# ---- logging / in-training sampling ----
SAVE_EVERY="${SAVE_EVERY:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
SAMPLE_NUM="${SAMPLE_NUM:-4}"
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
echo "data file: $DATA_FILE"
echo "tag: $TAG"
echo "batchsize=$BATCHSIZE lr=$LR nepochs=$NEPOCHS cond_drop_prob=$COND_DROP_PROB"

RESUME_ARGS=()
[[ -n "$RESUME_CKPT" ]] && RESUME_ARGS=(--load_ckpt "$RESUME_CKPT")

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
