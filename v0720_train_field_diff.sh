#!/bin/bash
#SBATCH --job-name=field_diff
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=field_diff_%j.out

# NOTE: dataset is ~50k mixed-shape parts held in RAM as float32 (~8G),
# hence mem=64G. Trainer is single-GPU; to resume a run that hits the
# time limit, add: --load_ckpt <run_dir>/checkpoints/ckpt_best.pth

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0720_field_diff_perceiver_trainer.py"
DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine/50000_neuralfield_condVF_shape_voxels_shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_vfmode-append_to_condition.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/checkpoints/field_diff"
DEVICE="cuda"

BATCHSIZE=8            # parts per step
N_POINTS=2048          # query points per part per step
N_CONTEXT=1024         # context points per part per step
NEPOCHS=500
LR=1e-4
COND_DROP_PROB=0.1     # keep >0 to enable CFG at sampling time

D_MODEL=256
N_LATENTS=256
N_HEADS=4
N_SELF_LAYERS=2
N_DEC_LAYERS=2
FOURIER_FREQS=32
FOURIER_SCALE=8.0
T_EMBED_DIM=64
LOSS_WEIGHTING="Simple"

NSAMPLES=5000             # 0 = all 50k parts; set e.g. 5000 for a fast pilot run
VAL_FRAC=0.05

SAVE_EVERY=1
SAMPLE_EVERY=1        # in-training ensembles are the expensive part; raise if slow
SAMPLE_NUM=1           # conditions per sampling event
ENSEMBLE_SIZE=1        # members per condition
EMA_DECAY=0.9999

SAMPLE_ATOL=1e-4
SAMPLE_RTOL=1e-4
SAMPLE_EPS=1e-3
SAMPLE_CONTEXT=4096
EVAL_CHUNK=16384

NUM_WORKERS=4
SEED=42
TAG="v0720_full"

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

echo "script: $SCRIPT"
echo "data file: $DATA_FILE"
echo "log root: $LOG_ROOT"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed $SEED \
  --batchsize $BATCHSIZE \
  --n_points $N_POINTS \
  --n_context $N_CONTEXT \
  --nepochs $NEPOCHS \
  --lr $LR \
  --cond_drop_prob $COND_DROP_PROB \
  --d_model $D_MODEL \
  --n_latents $N_LATENTS \
  --n_heads $N_HEADS \
  --n_self_layers $N_SELF_LAYERS \
  --n_dec_layers $N_DEC_LAYERS \
  --coord_fourier_freqs $FOURIER_FREQS \
  --coord_fourier_scale $FOURIER_SCALE \
  --t_embed_dim $T_EMBED_DIM \
  --loss_weighting "$LOSS_WEIGHTING" \
  --data_file "$DATA_FILE" \
  --nsamples $NSAMPLES \
  --val_frac $VAL_FRAC \
  --log_root "$LOG_ROOT" \
  --save_every_n_epochs $SAVE_EVERY \
  --sample_every_n_epochs $SAMPLE_EVERY \
  --sample_num $SAMPLE_NUM \
  --ensemble_size $ENSEMBLE_SIZE \
  --ema_decay $EMA_DECAY \
  --sample_atol $SAMPLE_ATOL \
  --sample_rtol $SAMPLE_RTOL \
  --sample_eps $SAMPLE_EPS \
  --sample_context $SAMPLE_CONTEXT \
  --eval_chunk $EVAL_CHUNK \
  --num_workers $NUM_WORKERS \
  --tag "$TAG"