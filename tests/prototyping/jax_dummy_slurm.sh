#!/bin/bash
#SBATCH --job-name=gencast
#SBATCH --nodes=2                # Number of nodes
#SBATCH --ntasks-per-node=1      # One process per node
#SBATCH --gres=gpu:1             # GPUs per node
#SBATCH --cpus-per-task=40
#SBATCH --time=00:20:00
#SBATCH --reservation=ilab
#SBATCH --qos=grace-xlarge
#SBATCH --partition=grace
#SBATCH --output=slurm-%j.out

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# Path to your Singularity container
CONTAINER=/explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511.sif

export MASTER_PORT=6000
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$SLURM_PROCID"

# Make sure JAX sees all GPUs
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_TASK-1)))

# Run inside container using srun (1 task per node)
srun singularity exec -B $NOBACKUP --nv $CONTAINER python jax_dummy_model.py
