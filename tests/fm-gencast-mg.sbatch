#!/bin/bash
#SBATCH --job-name=gpu_1node
#SBATCH --nodes=1
#SBATCH --ntasks=1              # one Python process on this node
#SBATCH --gres=gpu:4            # request 4 GPUs
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module purge
module load anaconda
conda activate gencast-env

echo "Job started on $(date)"
script_path="/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/src/FMGenCast/fm_gencast_mg.py"
echo "Running script: $script_path"

time srun python $script_path
echo "Job ended on $(date)"