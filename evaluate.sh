#!/bin/bash

#----------------------------------------------------------------#
# SLURM Directives
#----------------------------------------------------------------#
#
#SBATCH --job-name=imm-pytorch-eval
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err
#
#SBATCH --partition=dgx1       # Use the same partition as training
#SBATCH --qos=gpu2             # Use the same quality-of-service
#SBATCH --gres=gpu:1           # Evaluation only needs one GPU
#SBATCH --time=00:30:00        # 30 minutes should be plenty
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

#----------------------------------------------------------------#
# Environment Setup
#----------------------------------------------------------------#
#
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "=========================================================="

# --- Loading Conda Environment ---
# Activate the same environment that you successfully trained with
echo "Loading Conda..."
source ~/avinash/miniconda3/etc/profile.d/conda.sh
echo "Activating environment: imm_pytorch_new"
conda activate imm_pytorch_new

#----------------------------------------------------------------#
# Run the Evaluation Script
#----------------------------------------------------------------#
#
echo "--- Starting Python Evaluation Script ---"

python evaluate.py \
    --checkpoint_path ./checkpoints_2gpu/model_epoch_100.pth \
    --dataset_path data/celeba \
    --output_dir ./results \
    --visualize \
    --n_landmarks 5

echo "--- Evaluation Job Finished ---"
echo "=========================================================="
echo "Job finished on $(date)"
echo "=========================================================="