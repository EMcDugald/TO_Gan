#!/bin/bash
#SBATCH --job-name=v0627_diff_coarse
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=v0627_diff_coarse_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_diffusion_trainer.py"
DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/v0627/checkpoints/diffusion"
DEVICE="cuda"

BATCHSIZE=4
NEPOCHS=500
LR=5e-5
UNET_CH1=64
UNET_CH2=128
UNET_CH3=256
T_EMBED_DIM=128
COND_EMBED_DIM=128
COND_CH=8
COND_DROP_PROB=0.0
MODE="X0"
LOSS_WEIGHTING="Simple"
IMG_SIZE=32
NSAMPLES=0
VAL_FRAC=0.1
SAVE_EVERY=25
SAMPLE_EVERY=25
SAMPLE_NUM=4
EMA_DECAY=0.9999
SAMPLE_ATOL=1e-4
SAMPLE_RTOL=1e-4
SAMPLE_EPS=1e-3
NUM_WORKERS=4
SEED=42

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "log root: $LOG_ROOT"
echo "mode: coarse"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed $SEED \
  --batchsize $BATCHSIZE \
  --nepochs $NEPOCHS \
  --lr $LR \
  --unet_ch1_dim $UNET_CH1 \
  --unet_ch2_dim $UNET_CH2 \
  --unet_ch3_dim $UNET_CH3 \
  --t_embed_dim $T_EMBED_DIM \
  --cond_embed_dim $COND_EMBED_DIM \
  --cond_ch $COND_CH \
  --cond_drop_prob $COND_DROP_PROB \
  --mode "$MODE" \
  --loss_weighting "$LOSS_WEIGHTING" \
  --data_file "$DATA_FILE" \
  --img_size $IMG_SIZE \
  --nsamples $NSAMPLES \
  --val_frac $VAL_FRAC \
  --log_root "$LOG_ROOT" \
  --save_every_n_epochs $SAVE_EVERY \
  --sample_every_n_epochs $SAMPLE_EVERY \
  --sample_num $SAMPLE_NUM \
  --ema_decay $EMA_DECAY \
  --sample_atol $SAMPLE_ATOL \
  --sample_rtol $SAMPLE_RTOL \
  --sample_eps $SAMPLE_EPS \
  --num_workers $NUM_WORKERS