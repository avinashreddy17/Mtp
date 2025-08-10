#!/bin/bash

#----------------------------------------------------------------#
# SLURM Directives for your Server
#----------------------------------------------------------------#
#
#SBATCH --job-name=imm-pytorch-train   # A descriptive name for your job
#SBATCH --output=slurm_train_%j.out    # Standard output file
#SBATCH --error=slurm_train_%j.err     # Standard error file
#
#SBATCH --partition=dgx1               
#SBATCH --qos=gpu2                     
#SBATCH --gres=gpu:2                   
#SBATCH --time=24:00:00                
#SBATCH --cpus-per-task=4              
#SBATCH --mem=32G                      

#----------------------------------------------------------------#
# Environment Setup
#----------------------------------------------------------------#
#
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# --- Loading Conda Environment ---
# Use the exact source command from your previous file
echo "Loading Conda..."
source ~/avinash/miniconda3/etc/profile.d/conda.sh

# Activate the correct conda environment for this project
echo "Activating environment: imm_pytorch"
conda activate imm_pytorch

#----------------------------------------------------------------#
# Run the Training Script
#----------------------------------------------------------------#
#
echo "--- Starting Python Training Script ---"

# Run the updated train.py script.
# The batch size of 32 will be multiplied by the number of GPUs (2),
# resulting in an effective batch size of 64.
python train.py \
    --dataset_path /path/to/your/celeba \
    --batch_size 32 \
    --epochs 150 \
    --num_workers 8 \
    --checkpoint_dir ./checkpoints_2gpu

echo "--- Training Job Finished ---"
echo "=========================================================="
echo "Job finished on $(date)"
echo "=========================================================="
