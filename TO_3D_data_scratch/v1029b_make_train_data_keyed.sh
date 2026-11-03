#!/bin/bash
#SBATCH --job-name=make_train_data_keyed
#SBATCH --account=hdb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=make_train_data_keyed_%j.out

# Keyed binned-TO-spec dataset for the v1029b manufacturability model, then
# the go/no-go diagnostic, then sanity plots. CPU only; run with plain `bash`
# (or nohup / sbatch so an editor reload cannot kill it).
#
# Same pipeline as v1029_make_train_data_binned.sh, but the condition vector
# and the bucket key use only the KEY's fields (shape is always in the bucket
# key and reaches the model through the domain mask). Default key "load+vf"
# (load octant + VF quartile, cond_dim 14) was chosen from
# v1029_key_ablation.py on 2026-10-29: 224 buckets, median 20 parts, no
# singletons, no pool condition without neighbors.
#
# Env overrides, e.g.
#   bash v1029b_make_train_data_keyed.sh
#   KEY=load+ldist+vf bash v1029b_make_train_data_keyed.sh
#   EXCLUDE_SHAPES=120x20x20 bash v1029b_make_train_data_keyed.sh
#   SKIP_VIZ=1 bash v1029b_make_train_data_keyed.sh
#
# Output: $OUTDIR_ROOT/keyed_<key>_k${K_BINS}_vf${N_VF_BINS}_p${MANUF_PCTL}_split${SPLIT_SEED}[_ex<shapes>]/
#   KEY's '+' become '-' in the folder name.

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
EXCLUDE_SHAPES="${EXCLUDE_SHAPES:-}"
KEY="${KEY:-load+vf}"
SKIP_VIZ="${SKIP_VIZ:-0}"
N_PARTS_VIZ="${N_PARTS_VIZ:-8}"
N_BUCKETS_VIZ="${N_BUCKETS_VIZ:-6}"

KEY_TAG=$(echo "$KEY" | tr '+' '-')
EX_TAG=""
[[ -n "$EXCLUDE_SHAPES" ]] && EX_TAG="_ex$(echo "$EXCLUDE_SHAPES" | tr ',' '-')"
OUTDIR="${OUTDIR:-$OUTDIR_ROOT/keyed_${KEY_TAG}_k${K_BINS}_vf${N_VF_BINS}_p${MANUF_PCTL}_split${SPLIT_SEED}${EX_TAG}}"
mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
echo "outdir: $OUTDIR"
cd "$SCRIPT_DIR" || exit 1

$PYTHON v1029b_make_train_data_keyed.py \
  --data-root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --manufacturability-csv "$MANUF_CSV" \
  --manuf-scalar-column "$MANUF_SCALAR_COLUMN" \
  --manuf-percentile-threshold $MANUF_PCTL \
  --k-bins $K_BINS \
  --n-vf-bins $N_VF_BINS \
  --pool-frac $POOL_FRAC \
  --split-seed $SPLIT_SEED \
  --shapes-filter "$SHAPES_FILTER" \
  --exclude-shapes "$EXCLUDE_SHAPES" \
  --key "$KEY" || exit 1

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
echo "Check above: 'Condition dim: 14' for KEY=load+vf (8 load + 4 vf + 2 manuf), the split counts, and the diagnostic flags."
