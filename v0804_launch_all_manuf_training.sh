#!/bin/bash
# Submits all 4 manufacturability training runs as INDEPENDENT SLURM jobs
# (each gets its own GPU allocation and runs in parallel, rather than 4x
# sequential training on one GPU).
#
# Usage: bash v0804_launch_all_manuf_training.sh

for MODEL_ID in 1 2 3 4; do
  echo "Submitting model $MODEL_ID..."
  sbatch v0804_train_masked_voxels_manuf.sh "$MODEL_ID"
done

echo ""
echo "Submitted 4 jobs. Check status with: squeue -u \$USER"
