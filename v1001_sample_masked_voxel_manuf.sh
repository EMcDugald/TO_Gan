#!/bin/bash
#SBATCH --job-name=masked_vox_sample_pin
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=masked_vox_sample_pin_%j.out

# Test harness for v1001_masked_voxel_diff_sampler.py: same as
# v0804_sample_masked_voxel_manuf.sh plus load/BC solid pinning and
# reproducible condition selection. The v0804 sampler and .sh are untouched.
#
# Usage: model number (1-4) as the first argument; everything else is an
# environment-variable override, e.g.
#
#   sbatch v1001_sample_masked_voxel_manuf.sh 2                     # load pinned (default)
#   PIN_LOAD=0 sbatch v1001_sample_masked_voxel_manuf.sh 2          # unpinned baseline
#   PIN_BC=1 sbatch v1001_sample_masked_voxel_manuf.sh 2            # load + BC pinned
#   INDICES_FROM_RUN=/xdisk/.../model2_override-positive_w3.0_seed0_v5 \
#     sbatch v1001_sample_masked_voxel_manuf.sh 2                   # reuse an old run's conditions
#   VAL_ONLY=1 N_RANDOM=32 sbatch v1001_sample_masked_voxel_manuf.sh 4   # held-out conditions only
#
# Clean A/B: run once with PIN_LOAD=0 and once with PIN_LOAD=1, same SEED and
# same INDICES_FROM_RUN (or INDICES). With pinning off this sampler is
# bit-identical to the v0804 sampler. Compare run_summary.json in each outdir
# (mean_load_contact, mean_bc_contact, mean_pairwise_iou), and in the pinned
# run check total_pin_voxels_forced_at_end (should be 0) and
# mean_frac_pin_in_largest_component (should be near 1.0; low values mean the
# pin produced floating blobs).

MODEL_ID="$1"
if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: sbatch v1001_sample_masked_voxel_manuf.sh <1|2|3|4>"
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
[[ -f "$DATA_FILE" ]] || { echo "ERROR: data file not found: $DATA_FILE"; exit 1; }
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v1001_masked_voxel_diff_sampler.py"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

# ---- which conditions (priority: INDICES_FROM_RUN > INDICES > random) ----
INDICES_FROM_RUN="${INDICES_FROM_RUN:-}"   # earlier sampler outdir to reuse
INDICES="${INDICES:-}"                     # e.g. "3,17,102"
N_RANDOM="${N_RANDOM:-124}"
VAL_ONLY="${VAL_ONLY:-0}"                  # 1 -> held-out val split only
NOISE_SEED_MODE="${NOISE_SEED_MODE:-position}"   # position (v0804 behavior) | index

# ---- sampling (defaults = current v0804_sample_masked_voxel_manuf.sh) ----
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"
N_STEPS="${N_STEPS:-300}"
SAMPLE_EPS="${SAMPLE_EPS:-2e-5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"

# ---- pinning ----
PIN_LOAD="${PIN_LOAD:-1}"
PIN_BC="${PIN_BC:-0}"
PIN_RADIUS="${PIN_RADIUS:-1}"
CONTACT_RADIUS="${CONTACT_RADIUS:-1}"

# ---- manufacturability override: none/positive/negative/empty/value ----
MANUF_OVERRIDE="${MANUF_OVERRIDE:-none}"
MANUF_OVERRIDE_PERCENTILE="${MANUF_OVERRIDE_PERCENTILE:-}"
MANUF_OVERRIDE_RAW_VALUE="${MANUF_OVERRIDE_RAW_VALUE:-}"
MANUFACTURABILITY_CSV="${MANUFACTURABILITY_CSV:-$HOME/TO_Gan/TO_3D_data_scratch/data/summary.csv}"

# ---- outdir: encodes model, override (incl. value target), w, seed, pins ----
OV_TAG="$MANUF_OVERRIDE"
if [[ "$MANUF_OVERRIDE" == "value" ]]; then
  [[ -n "$MANUF_OVERRIDE_PERCENTILE" ]] && OV_TAG="value-pctl${MANUF_OVERRIDE_PERCENTILE}"
  [[ -n "$MANUF_OVERRIDE_RAW_VALUE" ]] && OV_TAG="value-raw${MANUF_OVERRIDE_RAW_VALUE}"
fi
PIN_TAG="nopin"
if [[ "$PIN_LOAD" == "1" && "$PIN_BC" == "1" ]]; then PIN_TAG="pinLB-r${PIN_RADIUS}"
elif [[ "$PIN_LOAD" == "1" ]]; then PIN_TAG="pinL-r${PIN_RADIUS}"
elif [[ "$PIN_BC" == "1" ]]; then PIN_TAG="pinB-r${PIN_RADIUS}"
fi
SEL_TAG=""
[[ "$VAL_ONLY" == "1" ]] && SEL_TAG="_val"
[[ -n "$INDICES_FROM_RUN" ]] && SEL_TAG="${SEL_TAG}_reuse"
OUTDIR="${OUTDIR:-/xdisk/hdb/emcdugald/samples/masked_voxel_diff_manuf_v1001/model${MODEL_ID}_override-${OV_TAG}_w${GUIDANCE_SCALE}_seed${SEED}_${PIN_TAG}${SEL_TAG}}"
mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "model id: $MODEL_ID"
echo "ckpt: $CKPT"
echo "data file: $DATA_FILE"
echo "outdir: $OUTDIR"
echo "manuf_override: $OV_TAG   pins: $PIN_TAG   selection: ${INDICES_FROM_RUN:-${INDICES:-random N=$N_RANDOM}}"

EXTRA_ARGS=()
[[ -n "$MANUF_OVERRIDE_PERCENTILE" ]] && EXTRA_ARGS+=(--manuf_override_percentile "$MANUF_OVERRIDE_PERCENTILE")
[[ -n "$MANUF_OVERRIDE_RAW_VALUE" ]] && EXTRA_ARGS+=(--manuf_override_raw_value "$MANUF_OVERRIDE_RAW_VALUE" --manufacturability_csv "$MANUFACTURABILITY_CSV")
[[ -n "$INDICES_FROM_RUN" ]] && EXTRA_ARGS+=(--indices_from_run "$INDICES_FROM_RUN")
[[ "$VAL_ONLY" == "1" ]] && EXTRA_ARGS+=(--val_only)
[[ "$PIN_LOAD" == "1" ]] && EXTRA_ARGS+=(--pin_load)
[[ "$PIN_BC" == "1" ]] && EXTRA_ARGS+=(--pin_bc)

$PYTHON "$SCRIPT" \
  --ckpt "$CKPT" \
  --data_file "$DATA_FILE" \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  --seed $SEED \
  --indices "$INDICES" \
  --n_random $N_RANDOM \
  --noise_seed_mode "$NOISE_SEED_MODE" \
  --ensemble_size $ENSEMBLE_SIZE \
  --n_steps $N_STEPS \
  --sample_eps $SAMPLE_EPS \
  --guidance_scale $GUIDANCE_SCALE \
  --pin_radius $PIN_RADIUS \
  --contact_radius $CONTACT_RADIUS \
  --manuf_override "$MANUF_OVERRIDE" \
  "${EXTRA_ARGS[@]}"
