#!/bin/bash
#SBATCH --job-name=masked_vox_sample_manuf
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=masked_vox_sample_manuf_%j.out

# Standalone ensemble sampling for one of the 4 manufacturability-conditioned
# Route C checkpoints (v0804_masked_voxel_diff_sampler.py).
#
# Usage: pass the model number (1-4) as the first argument. Every sampling
# param can be overridden via environment variable at submission time, same
# pattern as the training script:
#
#   sbatch v0804_sample_masked_voxel_manuf.sh 2
#   MANUF_OVERRIDE=positive sbatch v0804_sample_masked_voxel_manuf.sh 2
#   MANUF_OVERRIDE=value MANUF_OVERRIDE_PERCENTILE=0.1 \
#     sbatch v0804_sample_masked_voxel_manuf.sh 1
#   INDICES="3,17,102" GUIDANCE_SCALE=2 sbatch v0804_sample_masked_voxel_manuf.sh 4
#
# See the message accompanying this script for a full explanation of the
# --manuf_override modes (none/positive/negative/empty/value).
#
# mem=32G assumes these files (5564-10564 rows); OUTDIR encodes model,
# override, and guidance scale so repeated/swept calls don't overwrite
# each other.

MODEL_ID="$1"
if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: sbatch v0804_sample_masked_voxel_manuf.sh <1|2|3|4>"
  exit 1
fi

DATA_ROOT="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine_manuf"
CKPT_ROOT="/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff"
SHAPES_SUFFIX="shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine"

case "$MODEL_ID" in
  1)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_1_scalar_labeledonly/5564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-scalar_vfmode-append_to_condition.npy"
    DEFAULT_CKPT="$CKPT_ROOT/maskedVoxelDiff_bs-32_c1-32_c2-64_c3-128_drop-0.1_v0804_model1_scalar_labeledonly_20260805-105350/checkpoints/ckpt_best.pth"
    ;;
  2)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_2_binary_labeledonly/5564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-binary_vfmode-append_to_condition.npy"
    DEFAULT_CKPT="$CKPT_ROOT/maskedVoxelDiff_bs-32_c1-32_c2-64_c3-128_drop-0.1_v0804_model2_binary_labeledonly_20260805-105348/checkpoints/ckpt_best.pth"
    ;;
  3)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_3_scalar_plus5000unlabeled/10564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-scalar_vfmode-append_to_condition.npy"
    DEFAULT_CKPT="$CKPT_ROOT/maskedVoxelDiff_bs-32_c1-32_c2-64_c3-128_drop-0.1_v0804_model3_scalar_plus5000unlabeled_20260805-105904/checkpoints/ckpt_best.pth"
    ;;
  4)
    DEFAULT_DATA_FILE="$DATA_ROOT/model_4_binary_plus5000unlabeled/10564_neuralfield_condVF_shape_voxels_${SHAPES_SUFFIX}_manuf-binary_vfmode-append_to_condition.npy"
    DEFAULT_CKPT="$CKPT_ROOT/maskedVoxelDiff_bs-32_c1-32_c2-64_c3-128_drop-0.1_v0804_model4_binary_plus5000unlabeled_20260805-105904/checkpoints/ckpt_best.pth"
    ;;
  *)
    echo "Unknown MODEL_ID: $MODEL_ID (expected 1, 2, 3, or 4)"
    exit 1
    ;;
esac

DATA_FILE="${DATA_FILE:-$DEFAULT_DATA_FILE}"
CKPT="${CKPT:-$DEFAULT_CKPT}"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: data file not found: $DATA_FILE"; exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"; exit 1
fi

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0804_masked_voxel_diff_sampler.py"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

# ---- which conditions ----
INDICES="${INDICES:-}"          # e.g. "3,17,102"; empty -> N_RANDOM seeded draws
N_RANDOM="${N_RANDOM:-124}"

# ---- sampling ----
# ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"
# N_STEPS="${N_STEPS:-250}"
# SAMPLE_EPS="${SAMPLE_EPS:-1e-3}"
# GUIDANCE_SCALE="${GUIDANCE_SCALE:-10.0}"

ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"
N_STEPS="${N_STEPS:-300}"
SAMPLE_EPS="${SAMPLE_EPS:-2e-5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"

# ---- manufacturability override: none/positive/negative/empty/value ----
MANUF_OVERRIDE="${MANUF_OVERRIDE:-none}"
MANUF_OVERRIDE_PERCENTILE="${MANUF_OVERRIDE_PERCENTILE:-}"
MANUF_OVERRIDE_RAW_VALUE="${MANUF_OVERRIDE_RAW_VALUE:-}"
MANUFACTURABILITY_CSV="${MANUFACTURABILITY_CSV:-$HOME/TO_Gan/TO_3D_data_scratch/data/summary.csv}"

OUTDIR="${OUTDIR:-/xdisk/hdb/emcdugald/samples/masked_voxel_diff_manuf/model${MODEL_ID}_override-${MANUF_OVERRIDE}_w${GUIDANCE_SCALE}_seed${SEED}}"
mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "model id: $MODEL_ID"
echo "ckpt: $CKPT"
echo "data file: $DATA_FILE"
echo "outdir: $OUTDIR"
echo "manuf_override: $MANUF_OVERRIDE"

EXTRA_ARGS=()
if [[ -n "$MANUF_OVERRIDE_PERCENTILE" ]]; then
  EXTRA_ARGS+=(--manuf_override_percentile "$MANUF_OVERRIDE_PERCENTILE")
fi
if [[ -n "$MANUF_OVERRIDE_RAW_VALUE" ]]; then
  EXTRA_ARGS+=(--manuf_override_raw_value "$MANUF_OVERRIDE_RAW_VALUE"
              --manufacturability_csv "$MANUFACTURABILITY_CSV")
fi

$PYTHON "$SCRIPT" \
  --ckpt "$CKPT" \
  --data_file "$DATA_FILE" \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  --seed $SEED \
  --indices "$INDICES" \
  --n_random $N_RANDOM \
  --ensemble_size $ENSEMBLE_SIZE \
  --n_steps $N_STEPS \
  --sample_eps $SAMPLE_EPS \
  --guidance_scale $GUIDANCE_SCALE \
  --manuf_override "$MANUF_OVERRIDE" \
  "${EXTRA_ARGS[@]}"