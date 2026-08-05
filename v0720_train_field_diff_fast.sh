#!/bin/bash
#SBATCH --job-name=field_diff_10k
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=field_diff_10k_%j.out

# Notes:
# * mem=32G assumes a DEDICATED 10k data file (~1.6G of voxels in RAM).
#   If you point DATA_FILE at the 50k file and rely on --nsamples, the
#   loader still reads all 50k parts -> set mem=64G instead.
# * Assumes the trainer has been patched to use v0720_fast_patch.py
#   (validate_fast + heun_sampler_field_batched). Without the patch these
#   settings still work, but per-epoch validation and any sampling event
#   will be much slower.
# * To resume after the time limit:
#   add --load_ckpt <run_dir>/checkpoints/ckpt_best.pth

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
#SCRIPT="$HOME/TO_Gan/PoC_3d/v0720_field_diff_perceiver_trainer.py"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0720_field_diff_train_fast.py"

# <-- point this at your new 10k file once generated
DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine/10000_neuralfield_condVF_shape_voxels_shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_vfmode-append_to_condition.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/checkpoints/field_diff_fast"
DEVICE="cuda"

# ---- optimization ----
BATCHSIZE=32           # model is tiny; steps are overhead-bound, so go wide.
                       # If loss is unstable in the first epochs, drop to 16.
N_POINTS=8192          # query points per part per step
N_CONTEXT=1024         # context points per part per step
NEPOCHS=1000            # 9500 train parts / bs 32 ~= 297 steps/epoch
LR=1e-4                # scaled up ~2x with the 4x batch; 1e-4 is the safe fallback
COND_DROP_PROB=0.1     # keep >0 so CFG stays available at sampling time

# ---- model (unchanged from the working config) ----
D_MODEL=256
N_LATENTS=256
N_HEADS=8
N_SELF_LAYERS=4
N_DEC_LAYERS=2
FOURIER_FREQS=128
FOURIER_SCALE=8.0
T_EMBED_DIM=128
LOSS_WEIGHTING="Simple"

# ---- data ----
NSAMPLES=0             # 0 = use everything in DATA_FILE (the whole 10k)
VAL_FRAC=0.05          # 500 val parts; validate_fast caps at max_parts=128

# ---- logging / checkpoints / in-training sampling ----
SAVE_EVERY=25
SAMPLE_EVERY=25        # was 1 -- this was the main training-time sink
SAMPLE_NUM=2           # conditions per sampling event
ENSEMBLE_SIZE=2        # members per condition (one batched Heun solve each)
EMA_DECAY=0.9999

# ---- sampling ----
# atol/rtol/eps only matter if you fall back to the RK45 path; the Heun
# sampler uses n_steps=120 (set in the trainer call), eps below.
SAMPLE_ATOL=1e-4
SAMPLE_RTOL=1e-4
SAMPLE_EPS=1e-3
SAMPLE_CONTEXT=4096
EVAL_CHUNK=16384

NUM_WORKERS=6
SEED=42
TAG="v0721_10k"

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
nvidia-smi

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