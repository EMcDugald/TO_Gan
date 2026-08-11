#!/bin/bash
#SBATCH --job-name=make_train_data_manuf
#SBATCH --account=hdb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=make_train_data_manuf_%j.out

# Generates all 4 manufacturability-conditioned datasets from
# v0804_make_train_data_neural_diff.py (data-gen only, CPU-bound, no GPU
# needed -- adjust --partition above if "standard" isn't the right CPU
# queue on your cluster).
#
# Model 1: labeled-only,             manufacturability = scalar (percentile rank)
# Model 2: labeled-only,             manufacturability = binary (+1/-1 label)
# Model 3: labeled + 5000 unlabeled, manufacturability = scalar
# Model 4: labeled + 5000 unlabeled, manufacturability = binary
#
# All 4 use the SAME 80th-percentile threshold on deformation_p99 (computed
# fresh each run over the full labeled CSV population -- identical result
# every time since it doesn't depend on shape filtering or which run this
# is). Models 2 and 4 therefore use the identical positive/negative split.
#
# NOTE on --n-samples: your real universe has ~5564 labeled parts total
# (across ALL shapes), so after shape-filtering to the 7 Route C shapes,
# models 1/2's actual pool will likely be well under N_SAMPLES=10000 --
# the pool just gets used in full (see "Manufacturability pool: ..." in
# the printed output), it won't error. Run
# v0804_check_labeled_shape_coverage.py first to see the real per-shape
# labeled counts before this job runs, in case any shape bucket is thin
# enough to warrant dropping via --shapes-filter.
#
# Check each run's printed "Manufacturability: ..." line and the saved
# meta.npz (manuf_threshold_value, manuf_n_labeled_used,
# manuf_n_unlabeled_used) to confirm before training.

module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/TO_3D_data_scratch/v0804_make_train_data_neural_diff.py"

DATA_ROOT="$HOME/TO_Gan/TO_3D_data_scratch/data"
MANUF_CSV="$DATA_ROOT/summary.csv"
OUTDIR_ROOT="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine_manuf"

# Same conditioning as the existing 10k/50k Route C data (fine BC/load/DOF,
# VF appended). manufacturability= gets overridden by --manufacturability-mode
# regardless of what's in this string.
CONDITIONING_SPEC="bc_locations=fine,bc_dofs=fine,load_location=fine,load_direction=fine"

MANUF_SCALAR_COLUMN="deformation_p99"
MANUF_PCTL=80.0
RNG_SEED=0

mkdir -p "$OUTDIR_ROOT"

echo "which python: $PYTHON"
echo "manuf csv: $MANUF_CSV"

run_one () {
  local tag=$1
  local mode=$2
  local n_unlabeled=$3
  local n_samples=$4
  local outdir="$OUTDIR_ROOT/model_${tag}"
  mkdir -p "$outdir"
  echo ""
  echo "=== Model $tag: mode=$mode  n_unlabeled_extra=$n_unlabeled  n_samples=$n_samples ==="
  $PYTHON "$SCRIPT" \
    --data-root "$DATA_ROOT" \
    --outdir "$outdir" \
    --n-samples $n_samples \
    --conditioning-spec "$CONDITIONING_SPEC" \
    --vf-mode append_to_condition \
    --include-shape-in-cond true \
    --rng-seed $RNG_SEED \
    --manufacturability-mode "$mode" \
    --manufacturability-csv "$MANUF_CSV" \
    --manuf-scalar-column "$MANUF_SCALAR_COLUMN" \
    --manuf-percentile-threshold $MANUF_PCTL \
    --n-unlabeled-extra $n_unlabeled
}

run_one "1_scalar_labeledonly"        "scalar" 0    10000
run_one "2_binary_labeledonly"        "binary" 0    10000
run_one "3_scalar_plus5000unlabeled"  "scalar" 5000 15000
run_one "4_binary_plus5000unlabeled"  "binary" 5000 15000

echo ""
echo "All 4 datasets written under $OUTDIR_ROOT"
