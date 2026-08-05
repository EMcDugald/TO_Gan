#!/bin/bash
#SBATCH --job-name=diff_v0718_neural_implicit
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=diff_v0718_neural_implicit_%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0718_neural_diff_trainer.py"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine/50000_neuralfield_condVF_shape_voxels_shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_vfmode-append_to_condition.npy"
META_PATH="${DATA_FILE%.npy}_meta.npz"

LOG_ROOT="/xdisk/hdb/emcdugald/checkpoints/neural_diff"
DEVICE="cuda"

BATCHSIZE=1              # per-part; you use chunking for points
NEPOCHS=1000
LR=1e-4
WIDTH=256
DEPTH=4
OMEGA=30.0
T_EMBED_DIM=128
MODE="X0"
LOSS_WEIGHTING="Simple"
NSAMPLES=0
VAL_FRAC=0.1
CHUNK_SIZE=8192
SAVE_EVERY=10
SAMPLE_EVERY=10
SAMPLE_NUM=4
EMA_DECAY=0.9999
SAMPLE_ATOL=1e-4
SAMPLE_RTOL=1e-4
SAMPLE_EPS=1e-3
NUM_WORKERS=4
SEED=42
TAG="neural_implicit_v0718"

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "meta path: $META_PATH"
echo "log root: $LOG_ROOT"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed $SEED \
  --batchsize $BATCHSIZE \
  --nepochs $NEPOCHS \
  --lr $LR \
  --width $WIDTH \
  --depth $DEPTH \
  --omega $OMEGA \
  --t_embed_dim $T_EMBED_DIM \
  --mode "$MODE" \
  --loss_weighting "$LOSS_WEIGHTING" \
  --data_file "$DATA_FILE" \
  --meta_path "$META_PATH" \
  --nsamples $NSAMPLES \
  --val_frac $VAL_FRAC \
  --log_root "$LOG_ROOT" \
  --chunk_size $CHUNK_SIZE \
  --save_every_n_epochs $SAVE_EVERY \
  --sample_every_n_epochs $SAMPLE_EVERY \
  --sample_num $SAMPLE_NUM \
  --ema_decay $EMA_DECAY \
  --sample_atol $SAMPLE_ATOL \
  --sample_rtol $SAMPLE_RTOL \
  --sample_eps $SAMPLE_EPS \
  --num_workers $NUM_WORKERS \
  --tag "$TAG"