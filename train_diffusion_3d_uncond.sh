#!/bin/bash
#SBATCH --job-name=diff3d_uncond_big
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=diff3d_uncond_big_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

DATA_FILE="/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/octant/10000_labeled_voxels_32x32x32_octmass_score0.7.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/to_diffusion_3d_big/checkpoints_323232_octant"
SCRIPT="PoC_3d/train_diffusion_3d_uncond.py"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "data file: $DATA_FILE"
echo "log root: $LOG_ROOT"
echo "script: $SCRIPT"

$PYTHON "$SCRIPT" \
  --device cuda \
  --data_file "$DATA_FILE" \
  --log_root "$LOG_ROOT" \
  --img_size 32 \
  --seed 42 \
  --batchsize 64 \
  --nepochs 500 \
  --lr 5e-5 \
  --unet_ch1_dim 64 \
  --unet_ch2_dim 128 \
  --unet_ch3_dim 256 \
  --t_embed_dim 256 \
  --mode X0 \
  --loss_weighting Simple \
  --nsamples 0 \
  --val_frac 0.1 \
  --save_every_n_epochs 10 \
  --sample_every_n_epochs 25 \
  --sample_num 4 \
  --ema_decay 0.9999 \
  --sample_atol 1e-4 \
  --sample_rtol 1e-4 \
  --sample_eps 1e-4 \
  --num_workers 2
