#!/bin/bash

#----------------------------------------------------------------#
# SLURM Directives for a Quick Debug Job
#----------------------------------------------------------------#
#
#SBATCH --job-name=imm-debug-test      # A different name to distinguish it
#SBATCH --output=slurm_debug_%j.out    # Separate output file
#SBATCH --error=slurm_debug_%j.err     # Separate error file
#
#SBATCH --partition=dgx1               # Still use the same partition
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1                   # <<< Request only 1 GPU
#SBATCH --time=00:10:00                # <<< Request only 10 minutes
#SBATCH --cpus-per-task=2              # Request fewer CPUs
#SBATCH --mem=16G                      # Request less memory

#----------------------------------------------------------------#
# Environment and Execution
#----------------------------------------------------------------#
#
echo "=========================================================="
echo "RUNNING IN DEBUG MODE"
echo "Job started on $(date)"
echo "=========================================================="

# --- Loading Conda Environment ---
source ~/avinash/miniconda3/etc/profile.d/conda.sh
conda activate imm_pytorch

#----------------------------------------------------------------#
# Run the Training Script in Debug Mode
#----------------------------------------------------------------#
#
echo "--- Starting Python Script in Debug Mode ---"

python train.py \
    --dataset_path ~/avinash/Mtp/data/celeba \
    --batch_size 8 \
    --checkpoint_dir ./checkpoints_debug \
    --debug  # <<< Add the new debug flag

echo "--- Debug Job Finished ---"
