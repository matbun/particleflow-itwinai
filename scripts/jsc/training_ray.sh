#!/bin/bash

# shellcheck disable=all

# Job configuration
#SBATCH --job-name=test
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:20:00

# Resources allocation
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
# SBATCH --mem-per-gpu=10G
# SBATCH --exclusive


echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load environment modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))

# # Get GPUs info per node
# srun --cpu-bind=none --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c 'echo -e "NODE hostname: $(hostname)\n$(nvidia-smi)\n\n"'

num_gpus=$SLURM_GPUS_PER_NODE
num_cpus=$SLURM_CPUS_PER_TASK


# This tells Tune to not change the working directory to the trial directory
# which makes relative paths accessible from inside a trial
export RAY_CHDIR_TO_TRIAL_DIR=0
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_DISABLE=1

# Disable colors in output
export NO_COLOR=1
export RAY_COLOR_PREFIX=0

#########   Set up Ray cluster   ########

# Get the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t nodes_array <<< "$nodes"

# The head node will act as the central manager (head) of the Ray cluster.
head_node=${nodes_array[0]}
port=7639       # This port will be used by Ray to communicate with worker nodes.

echo "Starting HEAD at $head_node"
# Start Ray on the head node.
# The `--head` option specifies that this node will be the head of the Ray cluster.
# `srun` submits a job that runs on the head node to start the Ray head with the specified 
# number of CPUs and GPUs.
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "$num_cpus" --num-gpus "$num_gpus"  --block &

# Wait for a few seconds to ensure that the head node has fully initialized.
sleep 1

echo HEAD node started.

# Start Ray worker nodes
# These nodes will connect to the head node and become part of the Ray cluster.
worker_num=$((SLURM_JOB_NUM_NODES - 1))    # Total number of worker nodes (excl the head node)
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}   # Get the current worker node hostname.
    echo "Starting WORKER $i at $node_i"

    # Use srun to start Ray on the worker node and connect it to the head node.
    # The `--address` option tells the worker node where to find the head node.
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --redis-password='5241580000000000' \
        --num-cpus "$num_cpus" --num-gpus "$num_gpus" --block &
    
    sleep 5 # Wait before starting the next worker to prevent race conditions.
done
echo All Ray workers started.

##############################################################################################

echo "Starting training"

# Make mlpf visible
export PYTHONPATH="$PWD:$PYTHONPATH"

if [ -z "$EXPERIMENTS_LOCATION" ]; then 
  EXPERIMENTS_LOCATION="experiments"
fi
if [ -z "$N_TRAIN" ]; then 
  N_TRAIN=500
fi
if [ -z "$N_VALID" ]; then 
  N_VALID=500
fi
if [ -z "$BATCH_SIZE" ]; then 
  BATCH_SIZE=32
fi


# when training with Ray Train, --gpus should be equal to toal number of GPUs across the Ray Cluster

# $PYTHON_VENV/bin/python -u $PWD/mlpf/pyg_pipeline.py \

    # --ntrain 500 \
    # --nvalid 500 \

# BS 96 is OK
# --ntrain 700000 \
# --nvalid $((500*SLURM_NNODES)) \
# --gpu-batch-multiplier 90 \

# Run the MLPF model baseline
uv run python -u $PWD/mlpf/pipeline.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /p/scratch/intertwin/datasets/clic/ \
    --prefix "baseline_ddp_ray_N_${SLURM_NNODES}_" \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --num-epochs 2
