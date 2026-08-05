#!/bin/bash
#SBATCH --job-name=masked_vox_10k
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=masked_vox_10k_%j.out

# Route C: shape-bucketed masked voxel diffusion (v0723).
# Notes:
# * mem=32G assumes the dedicated 10k data file (~1.6G voxels in RAM).
#   For the 50k file set mem=64G (loader reads everything regardless of
#   --nsamples).
# * Largest padded shape is 120x40x16 = 76.8k voxels; batchsize 16 of those
#   is ~2x the memory of the proven 4x32^3 runs -- fine on a 16GB+ GPU.
#   If you OOM, drop BATCHSIZE to 8.
# * Resume: add --load_ckpt <run_dir>/checkpoints/ckpt_best.pth
# * Watch the in-training manifests: <run_dir>/samples_epoch_*/
#   sample_manifest.txt prints member masses, speckle fractions
#   (target < 0.05), and pairwise IoU (target 0.5-0.8, i.e. diverse but
#   not random).

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0723_masked_voxel_diff_trainer.py"

DATA_FILE="/xdisk/hdb/emcdugald/train_data/diffusion_neural_fine/50000_neuralfield_condVF_shape_voxels_shapes-32x32x32_40x40x20_60x40x20_64x32x16_80x40x15_120x20x20_120x40x10_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_vfmode-append_to_condition.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/checkpoints/masked_voxel_diff"
DEVICE="cuda"

# ---- optimization ----
BATCHSIZE=32          # drop to 8 if OOM
NEPOCHS=5000
LR=1e-4               # the proven voxel-U-Net setting
GRAD_CLIP=1.0
COND_DROP_PROB=0.1    # keep > 0 so CFG stays available at sampling time
LOSS_WEIGHTING="Simple"

# ---- model (the proven 32^3 U-Net config, + mask channel) ----
C1=32
C2=64
C3=128
T_EMBED_DIM=128
COND_EMBED_DIM=128
COND_CH=8

# ---- data ----
NSAMPLES=0            # 0 = the whole file
VAL_FRAC=0.05
VAL_MAX_PARTS=512

# ---- logging / in-training sampling ----
SAVE_EVERY=50
SAMPLE_EVERY=50
SAMPLE_NUM=2          # conditions per sampling event
ENSEMBLE_SIZE=2       # members per condition (one batched Heun solve)
SAMPLE_N_STEPS=100
SAMPLE_EPS=1e-3
EMA_DECAY=0.9999

NUM_WORKERS=6
SEED=42
TAG="v0723_50k"

mkdir -p "$LOG_ROOT"

echo "which python: $PYTHON"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "script: $SCRIPT"
echo "data file: $DATA_FILE"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed $SEED \
  --batchsize $BATCHSIZE \
  --nepochs $NEPOCHS \
  --lr $LR \
  --grad_clip $GRAD_CLIP \
  --cond_drop_prob $COND_DROP_PROB \
  --loss_weighting "$LOSS_WEIGHTING" \
  --unet_ch1_dim $C1 \
  --unet_ch2_dim $C2 \
  --unet_ch3_dim $C3 \
  --t_embed_dim $T_EMBED_DIM \
  --cond_embed_dim $COND_EMBED_DIM \
  --cond_ch $COND_CH \
  --data_file "$DATA_FILE" \
  --nsamples $NSAMPLES \
  --val_frac $VAL_FRAC \
  --val_max_parts $VAL_MAX_PARTS \
  --log_root "$LOG_ROOT" \
  --save_every_n_epochs $SAVE_EVERY \
  --sample_every_n_epochs $SAMPLE_EVERY \
  --sample_num $SAMPLE_NUM \
  --ensemble_size $ENSEMBLE_SIZE \
  --sample_n_steps $SAMPLE_N_STEPS \
  --sample_eps $SAMPLE_EPS \
  --ema_decay $EMA_DECAY \
  --num_workers $NUM_WORKERS \
  --tag "$TAG"
