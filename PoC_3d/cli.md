# Loss curve plots

module load anaconda
PYTHON="$HOME/.conda/envs/to_gan/bin/python"
PLOT_SCRIPT="$HOME/TO_Gan/PoC_3d/v0627_plot_train_metrics_gan_mdd.py"
CKPT_DIR="/xdisk/hdb/emcdugald/v0627/checkpoints/gan/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel_low_mass_epochs500_bs32_nz512_ngf256_ndf128_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.05_dEvery5_div1_coarse_run_20260628-212249"
$PYTHON "$PLOT_SCRIPT" "$CKPT_DIR" --rolling 25


CKPT_UNET="/xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel_low_mass_epochs500_bs16_nz256_ngf128_ndf32_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.1_dEvery5_div1_coarse_run_20260629-022915"

$PYTHON "$PLOT_SCRIPT" "$CKPT_UNET" --rolling 25


# sampler (gan)

python v0627_sampler.py \
  --arch nonunet \
  --data-path /xdisk/hdb/emcdugald/v0627/data/train_bc_load_mass.npy \
  --checkpoint-path /xdisk/hdb/emcdugald/v0627/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs500_bs16_nz512_ngf256_ndf128_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.05_dEvery5_div1_fine_run_20260628-200854/ckpt_best.pt \
  --outdir /xdisk/hdb/emcdugald/v0627/samples/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_nz512_ngf256_ckptbest_3x5 \
  --nz 512 \
  --ngf 256 \
  --n-random 5 \
  --n-per-cond 3 \
  --rng-seed 0 \
  --device cuda

  # sampler (diffusion)

  python v0627_diffusion_sampler.py \
  --data-path /xdisk/hdb/emcdugald/v0627/diffusion_condLabel_voxels_32x32x32_...npy \
  --checkpoint-path /xdisk/hdb/emcdugald/v0627/diffusion/checkpoints/.../ckpt_best.pth \
  --outdir /xdisk/hdb/emcdugald/v0627/diffusion_samples/ckptbest_3x5 \
  --img-size 32 \
  --unet-ch1-dim 32 \
  --unet-ch2-dim 64 \
  --unet-ch3-dim 128 \
  --t-embed-dim 128 \
  --cond-embed-dim 128 \
  --cond-ch 8 \
  --mode X0 \
  --n-random 5 \
  --n-per-cond 3 \
  --rng-seed 0 \
  --device cuda

 
### UPDATED LOSS CURVES (GANS "fine")

cd /home/u26/emcdugald/TO_Gan/PoC_3d

# Non-UNet GAN run
python v0627_plot_train_metrics_gan_mdd.py \
  /xdisk/hdb/emcdugald/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs300_bs32_nz512_ngf256_ndf128_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260706-184923 \
  --rolling 25

# UNet-based GAN run
python v0627_plot_train_metrics_gan_mdd.py \
  /xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs500_bs16_nz256_ngf128_ndf32_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.05_dEvery5_div1_fine_run_20260629-012911 \
  --rolling 25


### UPDATED SAMPLERS (GANS "fine")

cd /home/u26/emcdugald/TO_Gan/PoC_3d

python v0627_gan_mdd_sampler.py \
  --arch nonunet \
  --data-path \
    /xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781.npy \
  --meta-path \
    /xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781_meta.npz \
  --checkpoint-path \
    /xdisk/hdb/emcdugald/v0627/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs500_bs16_nz512_ngf256_ndf128_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.05_dEvery5_div1_fine_run_20260628-222403/ckpt_step_86250.pt \
  --outdir \
    /xdisk/hdb/emcdugald/v0627/sampler_tests/gan_ckpt86250_20260628_2x3_gpu \
  --nz 512 \
  --ngf 256 \
  --ndf 128 \
  --n-per-cond 3 \
  --n-random 2 \
  --rng-seed 0 \
  --device cuda



  python v0627_gan_mdd_sampler.py \
  --arch unet \
  --data-path \
    /xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781.npy \
  --meta-path \
    /xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781_meta.npz \
  --checkpoint-path \
    /xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs200_bs32_nz64_ngf64_ndf16_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260630-100915/ckpt_step_21500.pt \
  --outdir \
    /xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs200_bs32_nz64_ngf64_ndf16_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260630-100915/ckpt21500_2x3_gpu \
  --nz 64 \
  --ngf 64 \
  --ndf 16 \
  --n-per-cond 3 \
  --n-random 2 \
  --rng-seed 0 \
  --device cuda




  # Non-UNet GAN (coarse)
python v0627_plot_train_metrics_gan_mdd.py \
  /xdisk/hdb/emcdugald/v0627/checkpoints/gan/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel_low_mass_epochs500_bs32_nz512_ngf256_ndf128_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.05_dEvery5_div1_coarse_run_20260628-212249 \
  --rolling 25

# UNet-based GAN (coarse)
python v0627_plot_train_metrics_gan_mdd.py \
  /xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel_low_mass_epochs500_bs16_nz256_ngf128_ndf32_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.1_dEvery5_div1_coarse_run_20260629-022915 \
  --rolling 25


## UPDATED SAMPLERS FOR "COARSE" DIRECTORIES

python gan_mdd_sampler.py \
  --arch nonunet \
  --data-path \
    /xdisk/hdb/emcdugald/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024.npy \
  --meta-path \
    /xdisk/hdb/emcdugald/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.581024_meta.npz \
  --checkpoint-path \
    /xdisk/hdb/emcdugald/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs300_bs32_nz512_ngf256_ndf128_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260706-184923/ckpt_step_26280.pt \
  --outdir \
    /xdisk/hdb/emcdugald/checkpoints/gan/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel_low_mass_epochs300_bs32_nz512_ngf256_ndf128_nsamp10000_lrD0.0001_lrG0.0001_smoothR0.07_dEvery5_div1_fine_run_20260706-184923/ckpt26280_20260628_4x4_gpu \
  --nz 512 \
  --ngf 256 \
  --ndf 128 \
  --n-per-cond 4 \
  --n-random 4 \
  --rng-seed 0 \
  --device cuda



  python v0627_gan_mdd_sampler.py \
  --arch unet \
  --data-path \
    /xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy \
  --meta-path \
    /xdisk/hdb/emcdugald/v0627/train_data/gan/10000_gan_labeled_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781_meta.npz \
  --checkpoint-path \
    /xdisk/hdb/emcdugald/v0627/checkpoints/gan_unet/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel_low_mass_epochs500_bs16_nz256_ngf128_ndf32_nsamp10000_lrD5e-05_lrG5e-05_smoothR0.1_dEvery5_div1_coarse_run_20260629-022915/ckpt_step_12420.pt \
  --outdir \
    /xdisk/hdb/emcdugald/v0627/sampler_tests/gan_unet_coarse_ckpt12420_20260629_2x3_gpu \
  --nz 256 \
  --ngf 128 \
  --ndf 32 \
  --n-per-cond 3 \
  --n-random 2 \
  --rng-seed 0 \
  --device cuda


### SAMPLERS FOR DIFFUSION ###

python v0627_diffusion_sampler.py \
  --data-path \
    /xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781.npy \
  --meta-path \
    /xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_low_mass_thr0.494781_meta.npz \
  --checkpoint-path \
    /xdisk/hdb/emcdugald/v0627/checkpoints/diffusion/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel-low_mass_mode-X0_epochs-500_bs-4_c1-64_c2-128_c3-256_20260629-002403/checkpoints/ckpt_best.pth \
  --outdir \
    /xdisk/hdb/emcdugald/v0627/checkpoints/diffusion/bcLoc-coarse_bcDofs-omit_loadLoc-coarse_loadDir-omit_massLabel-low_mass_mode-X0_epochs-500_bs-4_c1-64_c2-128_c3-256_20260629-002403/diffusion_coarse_ckpt_best_20260629 \
  --img-size 32 \
  --unet-ch1-dim 64 \
  --unet-ch2-dim 128 \
  --unet-ch3-dim 256 \
  --t-embed-dim 128 \
  --cond-embed-dim 128 \
  --cond-ch 8 \
  --mode X0 \
  --sample-atol 1e-4 \
  --sample-rtol 1e-4 \
  --sample-eps 1e-3 \
  --n-random 2 \
  --n-per-cond 3 \
  --rng-seed 0 \
  --subset positive_only \
  --device cuda


python v0627_diffusion_sampler.py \
  --data-path \
    /xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781.npy \
  --meta-path \
    /xdisk/hdb/emcdugald/v0627/train_data/diffusion/10000_diffusion_condLabel_voxels_32x32x32_bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_low_mass_thr0.494781_meta.npz \
  --checkpoint-path \
    /xdisk/hdb/emcdugald/v0627/checkpoints/diffusion/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel-low_mass_mode-X0_epochs-500_bs-4_c1-32_c2-64_c3-128_20260629-001905/checkpoints/ckpt_best.pth \
  --outdir \
    /xdisk/hdb/emcdugald/v0627/checkpoints/diffusion/bcLoc-fine_bcDofs-fine_loadLoc-fine_loadDir-fine_massLabel-low_mass_mode-X0_epochs-500_bs-4_c1-32_c2-64_c3-128_20260629-001905/diffusion_fine_ckpt_best_20260629 \
  --img-size 32 \
  --unet-ch1-dim 32 \
  --unet-ch2-dim 64 \
  --unet-ch3-dim 128 \
  --t-embed-dim 128 \
  --cond-embed-dim 128 \
  --cond-ch 8 \
  --mode X0 \
  --sample-atol 1e-4 \
  --sample-rtol 1e-4 \
  --sample-eps 1e-3 \
  --n-random 2 \
  --n-per-cond 3 \
  --rng-seed 0 \
  --subset positive_only \
  --device cuda