#!/bin/bash
#SBATCH --job-name=masked_vox_sample
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=masked_vox_sample_%j.out

# Standalone ensemble sampling for the Route C masked voxel diffusion model
# (v0723_masked_voxel_diff_sampler.py).
#
# Usage:
#   1. Point CKPT at your run's checkpoints/ckpt_best.pth (hparams.json is
#      auto-loaded from the run dir, one level above the checkpoint).
#   2. sbatch v0723_sample_masked_voxel.sh
#      -- or run the $PYTHON command below directly inside a salloc session;
#      sampling is fast (seconds per condition), so interactive works fine.
#
# Notes:
# * INDICES="" -> draws N_RANDOM conditions with numpy default_rng(SEED),
#   so the same SEED reproduces the same conditions AND the same noise.
#   To compare specific conditions across checkpoints or guidance scales,
#   pin them explicitly, e.g. INDICES="3,17,102,4088".
# * GUIDANCE_SCALE=0.0 -> plain conditional sampling (max diversity).
#   Values in [1,4] tighten condition adherence at the cost of diversity
#   (valid because training used cond_drop_prob=0.1). A cheap first sweep:
#   run this script several times with w in {0, 0.5, 1, 2, 4}, writing each
#   to its own OUTDIR, with pinned INDICES and the same SEED.
# * Printed per condition: member masses vs GT, speckle fractions
#   (target < 0.05), pairwise IoU (target 0.5-0.8). Per-condition outputs:
#   gt.png, member*.png, ensemble.npz.
# * mem=32G assumes the 10k data file; use 64G for the 50k file.

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0723_masked_voxel_diff_sampler.py"

# ---- point this at your trained run ----
CKPT="/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff/maskedVoxelDiff_bs-16_c1-32_c2-64_c3-128_drop-0.1_v0723_50k_20260727-172339/checkpoints/ckpt_best.pth"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine/50000_neuralfield_condVF_shape_voxels_shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_vfmode-append_to_condition.npy"
DEVICE="cuda"
SEED=0

# ---- which conditions ----
INDICES=""            # e.g. "3,17,102,4088"; empty -> N_RANDOM seeded draws
N_RANDOM=32

# ---- sampling ----
ENSEMBLE_SIZE=8       # members per condition (one batched Heun solve each)
N_STEPS=250
SAMPLE_EPS=1e-3
GUIDANCE_SCALE=3.0

# Output dir encodes the guidance scale so sweeps don't overwrite each other
OUTDIR="/xdisk/hdb/emcdugald/samples/masked_voxel_diff/w${GUIDANCE_SCALE}_seed${SEED}_1"

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
  --indices "$INDICES" \
  --n_random $N_RANDOM \
  --ensemble_size $ENSEMBLE_SIZE \
  --n_steps $N_STEPS \
  --sample_eps $SAMPLE_EPS \
  --guidance_scale $GUIDANCE_SCALE