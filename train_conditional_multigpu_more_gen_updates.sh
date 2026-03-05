#!/bin/bash
#SBATCH --job-name=cgan3d_conditional_2gpu_more_gen_updates
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=16:00:00
#SBATCH --output=cgan3d_10k_2gpu_more_gen_updates%j.out

module load cuda11
module load anaconda

PYTHON="$HOME/.conda/envs/to_gan/bin/python"

echo "which python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable)"
$PYTHON -c "import torch; print('torch version in job:', torch.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

$PYTHON PoC_3d/TopOpt3d_323232_conditional_mgpu_ls_more_generator_updates.py
