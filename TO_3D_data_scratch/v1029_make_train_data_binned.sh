#!/bin/bash
#SBATCH --job-name=make_train_data_binned
#SBATCH --account=hdb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=make_train_data_binned_%j.out

# Binned-TO-spec dataset for the v1029 manufacturability model, then the
# go/no-go diagnostic, then sanity plots. CPU only; can be run with plain
# `bash` (no Slurm needed).
#
#   1. v1029_make_train_data_binned.py   train (80%) / pool (20%) files,
#                                        bucket_index.json, CSVs
#   2. v1029_bucket_diagnostics.py       does the label carry information
#                                        the binned condition does not?
#   3. v1029_visualize_train_data_binned.py   part plots, bucket galleries,
#                                        summary.png
#
# Env overrides, e.g.
#   bash v1029_make_train_data_binned.sh
#   K_BINS=3 bash v1029_make_train_data_binned.sh          # 27 cells instead of octants
#   N_VF_BINS=3 MANUF_PCTL=50 bash v1029_make_train_data_binned.sh
#   SKIP_VIZ=1 bash v1029_make_train_data_binned.sh        # data + diagnostic only
#
# Output: $OUTDIR_ROOT/binned_k${K_BINS}_vf${N_VF_BINS}_p${MANUF_PCTL}_split${SPLIT_SEED}/

module load anaconda

PYTHON="${PYTHON:-$HOME/.conda/envs/to_gan/bin/python}"
SCRIPT_DIR="${SCRIPT_DIR:-$HOME/TO_Gan/TO_3D_data_scratch}"

DATA_ROOT="${DATA_ROOT:-$HOME/TO_Gan/TO_3D_data_scratch/data}"
MANUF_CSV="${MANUF_CSV:-$DATA_ROOT/summary.csv}"
OUTDIR_ROOT="${OUTDIR_ROOT:-/xdisk/hdb/emcdugald/train_data/diffusion_binned}"

MANUF_SCALAR_COLUMN="${MANUF_SCALAR_COLUMN:-deformation_p99}"
MANUF_PCTL="${MANUF_PCTL:-80}"
K_BINS="${K_BINS:-2}"
N_VF_BINS="${N_VF_BINS:-4}"
POOL_FRAC="${POOL_FRAC:-0.2}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SHAPES_FILTER="${SHAPES_FILTER:-}"
SKIP_VIZ="${SKIP_VIZ:-0}"
N_PARTS_VIZ="${N_PARTS_VIZ:-8}"
N_BUCKETS_VIZ="${N_BUCKETS_VIZ:-6}"

OUTDIR="${OUTDIR:-$OUTDIR_ROOT/binned_k${K_BINS}_vf${N_VF_BINS}_p${MANUF_PCTL}_split${SPLIT_SEED}}"
mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
echo "outdir: $OUTDIR"
cd "$SCRIPT_DIR" || exit 1

$PYTHON v1029_make_train_data_binned.py \
  --data-root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --manufacturability-csv "$MANUF_CSV" \
  --manuf-scalar-column "$MANUF_SCALAR_COLUMN" \
  --manuf-percentile-threshold $MANUF_PCTL \
  --k-bins $K_BINS \
  --n-vf-bins $N_VF_BINS \
  --pool-frac $POOL_FRAC \
  --split-seed $SPLIT_SEED \
  --shapes-filter "$SHAPES_FILTER" || exit 1

echo ""
echo "================ bucket diagnostics ================"
$PYTHON v1029_bucket_diagnostics.py --data_dir "$OUTDIR" || exit 1

if [[ "$SKIP_VIZ" != "1" ]]; then
  echo ""
  echo "================ visualizer ================"
  $PYTHON v1029_visualize_train_data_binned.py --data_dir "$OUTDIR" --split train \
    --n_parts $N_PARTS_VIZ --n_buckets $N_BUCKETS_VIZ
fi

echo ""
echo "Check above: 'Condition dim: 28' (k=2, 4 VF bins), the split counts, and the diagnostic flags."
