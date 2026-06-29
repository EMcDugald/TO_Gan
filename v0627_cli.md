 # test trainers:
PYTHON="$HOME/.conda/envs/to_gan/bin/python"
$PYTHON "$SCRIPT"   --data-path "$DATA_FILE"   --checkpoint-root "$CHECKPOINT_ROOT"   --device "$DEVICE"   --batch-size 16   --num-epochs 2   --nz 512   --ngf 256   --ndf 128   --lr-d 5e-5   --lr-g 5e-5   --smooth-real 0.05   --d-every 5   --use-label-smoothing   --use-diversity-loss   --diversity-weight 0.01   --n-vis-samples 2   --ckpt-every-epochs 1   --eval-every-epochs 1   --tag "$TAG"


module load cuda11
module load anaconda
PYTHON="$HOME/.conda/envs/to_gan/bin/python"

DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy"
CHECKPOINT_ROOT="/xdisk/hdb/emcdugald/v0627/checkpoints/gan_test_coarse"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_gan_mdd_trainer.py"

TAG="coarse_run_test"
DEVICE="cuda"

$PYTHON "$SCRIPT" \
  --data-path "$DATA_FILE" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --device "$DEVICE" \
  --batch-size 32 \
  --num-epochs 2 \
  --nz 128 \
  --ngf 64 \
  --ndf 32 \
  --lr-d 5e-5 \
  --lr-g 5e-5 \
  --smooth-real 0.05 \
  --d-every 5 \
  --use-label-smoothing \
  --use-diversity-loss \
  --diversity-weight 0.01 \
  --n-vis-samples 2 \
  --ckpt-every-epochs 1 \
  --eval-every-epochs 1 \
  --tag "$TAG"



module load cuda11
module load anaconda
PYTHON="$HOME/.conda/envs/to_gan/bin/python"

DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781.npy"
CKPT_ROOT="/xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet_test_fine"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_gan_mdd_unet_trainer.py"

TAG="fine_run_test"
DEVICE="cuda"

$PYTHON "$SCRIPT" \
  --data-path "$DATA_FILE" \
  --checkpoint-root "$CKPT_ROOT" \
  --device "$DEVICE" \
  --batch-size 16 \
  --nz 256 \
  --ngf 128 \
  --ndf 32 \
  --num-epochs 2 \
  --lr-d 5e-5 \
  --lr-g 5e-5 \
  --smooth-real 0.05 \
  --d-every 5 \
  --use-label-smoothing \
  --use-diversity-loss \
  --diversity-weight 0.01 \
  --n-vis-samples 2 \
  --ckpt-every-epochs 1 \
  --eval-every-epochs 1 \
  --tag "$TAG"



module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_diffusion_trainer.py"
DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/v0627/checkpoints/diffusion_test"
DEVICE="cuda"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed 42 \
  --batchsize 2 \
  --nepochs 2 \
  --lr 5e-5 \
  --unet_ch1_dim 16 \
  --unet_ch2_dim 32 \
  --unet_ch3_dim 64 \
  --t_embed_dim 64 \
  --cond_embed_dim 64 \
  --cond_ch 4 \
  --cond_drop_prob 0.0 \
  --mode X0 \
  --loss_weighting Simple \
  --data_file "$DATA_FILE" \
  --img_size 32 \
  --nsamples 128 \
  --val_frac 0.1 \
  --log_root "$LOG_ROOT" \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_num 1 \
  --ema_decay 0.9999 \
  --sample_atol 1e-4 \
  --sample_rtol 1e-4 \
  --sample_eps 1e-3 \
  --num_workers 0 \
  --tag fine_smoke




module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_diffusion_trainer.py"
DATA_FILE="/xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy"
LOG_ROOT="/xdisk/hdb/emcdugald/v0627/checkpoints/diffusion_test"
DEVICE="cuda"

$PYTHON "$SCRIPT" \
  --device "$DEVICE" \
  --seed 42 \
  --batchsize 2 \
  --nepochs 2 \
  --lr 5e-5 \
  --unet_ch1_dim 8 \
  --unet_ch2_dim 16 \
  --unet_ch3_dim 32 \
  --t_embed_dim 32 \
  --cond_embed_dim 32 \
  --cond_ch 4 \
  --cond_drop_prob 0.0 \
  --mode X0 \
  --loss_weighting Simple \
  --data_file "$DATA_FILE" \
  --img_size 32 \
  --nsamples 128 \
  --val_frac 0.1 \
  --log_root "$LOG_ROOT" \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_num 1 \
  --ema_decay 0.9999 \
  --sample_atol 1e-4 \
  --sample_rtol 1e-4 \
  --sample_eps 1e-3 \
  --num_workers 0 \
  --tag coarse_smoke