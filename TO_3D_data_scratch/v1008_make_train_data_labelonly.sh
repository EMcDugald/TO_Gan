#!/bin/bash
#SBATCH --job-name=make_train_data_labelonly
#SBATCH --account=hdb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=make_train_data_labelonly_%j.out

# Label-only (class-conditional) dataset for the manufacturability generator.
#
# Same data-gen code as v0804_make_train_data_manuf.sh, but with every TO
# spec field switched OFF: no BC points/DOFs, no load point/direction, no
# volume fraction, no shape entry. The condition vector is just
#   [manuf_label (+1 positive / -1 negative), manuf_has_label (1)]
# so cond_dim = 2. The part's shape still reaches the model through the
# domain-mask channel, exactly as before.
#
# Why: with the TO spec in the condition, the label is a deterministic
# function of the spec, so the earlier models learned to ignore it and
# requested positive/negative both came out at the ~80% base rate.
#
# Uses only the ~5,564 parts with an Ansys deformation_p99 value. No GPU.
# Can be run directly with `bash` (no Slurm needed); it takes minutes.
#
# Env overrides, e.g.
#   bash v1008_make_train_data_labelonly.sh                 # 80th-percentile split
#   MANUF_PCTL=50 bash v1008_make_train_data_labelonly.sh   # 50/50 split instead
#
# Output: $OUTDIR_ROOT/labelonly_binary_p${MANUF_PCTL}/ with one .npy and its
# _meta.npz. The training .sh finds the file by that folder name.

module load anaconda

PYTHON="${PYTHON:-$HOME/.conda/envs/to_gan/bin/python}"
SCRIPT="${SCRIPT:-$HOME/TO_Gan/TO_3D_data_scratch/v0804_make_train_data_neural_diff.py}"

DATA_ROOT="${DATA_ROOT:-$HOME/TO_Gan/TO_3D_data_scratch/data}"
MANUF_CSV="${MANUF_CSV:-$DATA_ROOT/summary.csv}"
OUTDIR_ROOT="${OUTDIR_ROOT:-/xdisk/hdb/emcdugald/train_data/diffusion_labelonly}"

MANUF_SCALAR_COLUMN="${MANUF_SCALAR_COLUMN:-deformation_p99}"
MANUF_PCTL="${MANUF_PCTL:-80}"     # positive = value <= this percentile
RNG_SEED="${RNG_SEED:-0}"
N_SAMPLES="${N_SAMPLES:-100000}"   # a cap only; the labeled pool (~5,564) is used in full

OUTDIR="$OUTDIR_ROOT/labelonly_binary_p${MANUF_PCTL}"
mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
echo "manuf csv: $MANUF_CSV   column: $MANUF_SCALAR_COLUMN   percentile split: $MANUF_PCTL"
echo "outdir: $OUTDIR"

$PYTHON "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --n-samples $N_SAMPLES \
  --conditioning-spec "bc_locations=omit,bc_dofs=omit,load_location=omit,load_direction=omit" \
  --vf-mode omit \
  --include-shape-in-cond false \
  --rng-seed $RNG_SEED \
  --manufacturability-mode binary \
  --manufacturability-csv "$MANUF_CSV" \
  --manuf-scalar-column "$MANUF_SCALAR_COLUMN" \
  --manuf-percentile-threshold $MANUF_PCTL \
  --n-unlabeled-extra 0

echo ""
echo "Check above: 'Condition dim: 2' and '<N> labeled / 0 unlabeled'."
