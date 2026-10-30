#!/bin/bash
#SBATCH --job-name=sample_classcond
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=sample_classcond_%j.out

# Generates class-conditional parts from the label-only model
# (v1008_classcond_sampler.py): N parts per (label, shape), no TO spec.
#
# Usage (all env overrides):
#   sbatch v1008_sample_classcond.sh                                    # 16 pos + 16 neg per shape
#   N_PER_GROUP=100 sbatch v1008_sample_classcond.sh                    # 100 per label per shape
#   LABELS=positive,negative,unconditional sbatch v1008_sample_classcond.sh   # add base-rate group
#   SHAPES=64x32x16 N_PER_GROUP=100 GUIDANCE_SCALE=4 sbatch v1008_sample_classcond.sh
#
# Defaults find the newest checkpoint tagged v1008_labelonly_binary_p<PCTL>
# and the matching label-only data file; set CKPT / DATA_FILE to override.
#
# Outputs (OUTDIR): one folder per (label, shape) with ensemble.npz and a few
# PNGs, plus parts_index.csv (one row per part, with empty surrogate columns
# to fill in), training_data_reference.json and run_summary.json.
#
# Positive and negative groups of the same shape share their noise draws, so
# member k of each starts from the same noise.

MANUF_PCTL="${MANUF_PCTL:-80}"
DATA_ROOT="${DATA_ROOT:-/xdisk/hdb/emcdugald/train_data/diffusion_labelonly}"
CKPT_ROOT="${CKPT_ROOT:-/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff}"

if [[ -z "$DATA_FILE" ]]; then
  DATA_FILE=$(ls "$DATA_ROOT/labelonly_binary_p${MANUF_PCTL}"/*_vfmode-omit.npy 2>/dev/null | head -n 1)
fi
if [[ -z "$CKPT" ]]; then
  RUN_DIR=$(ls -td "$CKPT_ROOT"/*v1008_labelonly_binary_p${MANUF_PCTL}_* 2>/dev/null | head -n 1)
  CKPT="$RUN_DIR/checkpoints/ckpt_best.pth"
fi
[[ -f "$DATA_FILE" ]] || { echo "ERROR: data file not found ($DATA_FILE)"; exit 1; }
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found ($CKPT)"; exit 1; }

module load cuda11
module load anaconda

PYTHON="${PYTHON:-$HOME/.conda/envs/to_gan/bin/python}"
SCRIPT="${SCRIPT:-$HOME/TO_Gan/PoC_3d/v1008_classcond_sampler.py}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

LABELS="${LABELS:-positive,negative}"
SHAPES="${SHAPES:-all}"
N_PER_GROUP="${N_PER_GROUP:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_STEPS="${N_STEPS:-250}"
SAMPLE_EPS="${SAMPLE_EPS:-1e-3}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.0}"
N_PNG_PER_GROUP="${N_PNG_PER_GROUP:-4}"

LABEL_TAG=$(echo "$LABELS" | tr ',' '-')
SHAPE_TAG=$(echo "$SHAPES" | tr ',' '-')
OUTDIR="${OUTDIR:-/xdisk/hdb/emcdugald/samples/classcond_v1008/p${MANUF_PCTL}_${LABEL_TAG}_${SHAPE_TAG}_n${N_PER_GROUP}_w${GUIDANCE_SCALE}_s${N_STEPS}_seed${SEED}}"
mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "ckpt: $CKPT"
echo "data file: $DATA_FILE"
echo "outdir: $OUTDIR"

$PYTHON "$SCRIPT" \
  --ckpt "$CKPT" \
  --data_file "$DATA_FILE" \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  --seed $SEED \
  --labels "$LABELS" \
  --shapes "$SHAPES" \
  --n_per_group $N_PER_GROUP \
  --batch_size $BATCH_SIZE \
  --n_steps $N_STEPS \
  --sample_eps $SAMPLE_EPS \
  --guidance_scale $GUIDANCE_SCALE \
  --n_png_per_group $N_PNG_PER_GROUP
