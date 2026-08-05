#!/bin/bash
#SBATCH --job-name=route_c_sweep
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=route_c_sweep_%j.out

# Guidance_scale x n_steps sweep for the Route C masked voxel diffusion model
# (v0723_route_c_sweep.py). Reuses the exact sampling code from
# v0723_masked_voxel_diff_sampler.py -- this script just loops the grid.
#
# Usage:
#   1. Point CKPT/DATA_FILE at your trained run (same as
#      v0723_sample_masked_voxel.sh).
#   2. sbatch v0723_route_c_sweep.sh
#
# Notes:
# * Same INDICES/N_RANDOM behavior as the standalone sampler: empty INDICES
#   draws N_RANDOM conditions via numpy default_rng(SEED). The sweep script
#   additionally pins the SAME per-condition noise seed across every grid
#   point, so any visible difference between cells is attributable to
#   (guidance_scale, n_steps) only -- not resampled noise.
# * Recommended order: run with the default grid below first (guidance_scale
#   sweep at a fixed, already-used n_steps=250). Once you've picked a
#   guidance_scale from sweep_results.csv, do a second run with
#   GUIDANCE_SCALES="<chosen w>" and a few N_STEPS values to check step
#   convergence.
# * Outputs land in OUTDIR: sweep_results.csv (one row per grid point),
#   sweep_manifest.json (full per-condition detail), previews/*.png
#   (one quick-look image per grid point).
# * mem=32G assumes the 10k data file; use 64G for the 50k file.
# * Grid size = len(GUIDANCE_SCALES) * len(N_STEPS_LIST) * N_RANDOM
#   conditions * ENSEMBLE_SIZE members -- the default grid below (5 scales x
#   1 step count x 4 conditions x 8 members = 160 batched Heun solves of 250
#   steps each) should comfortably fit the 4hr time limit; widen N_STEPS_LIST
#   or increase --time if you extend the grid.

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0723_route_c_sweep.py"

# ---- point this at your trained run ----
CKPT="/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff/maskedVoxelDiff_bs-16_c1-32_c2-64_c3-128_drop-0.1_v0723_10k_20260723-004803/checkpoints/ckpt_best.pth"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine/10000_neuralfield_condVF_shape_voxels_shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_vfmode-append_to_condition.npy"
DEVICE="cuda"
SEED=123

# ---- which conditions (held fixed across the WHOLE grid) ----
INDICES=""            # e.g. "3,17,102,4088"; empty -> N_RANDOM seeded draws
N_RANDOM=8

# ---- sampling ----
ENSEMBLE_SIZE=4

# ---- the sweep grid ----
GUIDANCE_SCALES="0,0.5,1,2,4"
N_STEPS_LIST="250"
SAMPLE_EPS=1e-3

OUTDIR="/xdisk/hdb/emcdugald/samples/masked_voxel_diff/sweep_seed${SEED}"

mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "ckpt: $CKPT"
echo "data file: $DATA_FILE"
echo "outdir: $OUTDIR"
echo "guidance_scales: $GUIDANCE_SCALES"
echo "n_steps_list: $N_STEPS_LIST"

$PYTHON "$SCRIPT" \
  --ckpt "$CKPT" \
  --data_file "$DATA_FILE" \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  --seed $SEED \
  --indices "$INDICES" \
  --n_random $N_RANDOM \
  --ensemble_size $ENSEMBLE_SIZE \
  --guidance_scales "$GUIDANCE_SCALES" \
  --n_steps_list "$N_STEPS_LIST" \
  --sample_eps $SAMPLE_EPS