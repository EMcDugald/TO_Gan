#!/bin/bash
#SBATCH --job-name=sample_gan_fine
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=sample_gan_fine_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

SCRIPT="$HOME/TO_Gan/PoC_3d/gan_mdd_sampler.py"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024.npy"
META_FILE="/xdisk/hdb/emcdugald/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024_meta.npz"

CKPT="/xdisk/hdb/emcdugald/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs300_bs32_nz512_ngf256_ndf128_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260706-184923/ckpt_step_26280.pt"

OUTDIR="/xdisk/hdb/emcdugald/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs300_bs32_nz512_ngf256_ndf128_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260706-184923/gan_fine_ckpt_26280_4x4"
DEVICE="cuda"

NZ=512
NGF=256
NDF=128
N_PER_COND=4
N_RANDOM=4
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
echo "mode: gan nonunet fine"

$PYTHON "$SCRIPT" \
  --arch nonunet \
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