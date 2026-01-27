#!/bin/bash
#SBATCH --job-name=gan3d_10k
#SBATCH --account=hdb
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=gan3d_10k_%j.out

# module load cuda11         
# module load anaconda

# source ~/.bashrc

# conda activate to_gan  

# python PoC_3d/TopOpt3d_323232_refactor.py

module load cuda11
module load anaconda

# Make sure conda is initialized in this non-interactive shell
eval "$(conda shell.bash hook)"
conda activate to_gan

echo "which python: $(which python)"
python -c "import torch; print('torch version in job:', torch.__version__)"

python PoC_3d/TopOpt3d_323232_refactor.py
