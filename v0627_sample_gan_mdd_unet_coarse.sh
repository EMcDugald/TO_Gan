#!/bin/bash
#SBATCH --job-name=v0627_sample_unet_gan_coarse
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v0627_sample_unet_gan_coarse_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_gan_mdd_sampler.py"

DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy"
META_FILE="/xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781_meta.npz"

CKPT="/xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel_low_mass_epochs500_bs16_nz256_ngf128_ndf32_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.10_dEvery5_div0.01_coarse_run_20260628-234501/ckpt_best.pt"

OUTDIR="/xdisk/hdb/emcdugald/v0627/sampler_tests/unet_gan_coarse_ckpt_best_20260630"
DEVICE="cuda"

NZ=256
NGF=128
NDF=32
N_PER_COND=3
N_RANDOM=2
RNG_SEED=0

mkdir -p "$OUTDIR"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "meta file: $META_FILE"
echo "checkpoint: $CKPT"
echo "outdir: $OUTDIR"
echo "mode: unet gan coarse"

$PYTHON "$SCRIPT" \
  --arch unet \
  --data-path "$DATA_FILE" \
  --meta-path "$META_FILE" \
  --checkpoint-path "$CKPT" \
  --outdir "$OUTDIR" \
  --nz $NZ \
  --ngf $NGF \
  --ndf $NDF \
  --n-per-cond $N_PER_COND \
  --n-random $N_RANDOM \
  --rng-seed $RNG_SEED \
  --device "$DEVICE"