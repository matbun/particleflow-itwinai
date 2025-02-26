#!/bin/bash

# SLURM jobscript for Vega systems

# Job configuration
#SBATCH --job-name=ray_train
#SBATCH --account=d2024d11-083-users
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:10:00

# Resources allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
# SBATCH --mem-per-gpu=10G
#SBATCH --exclusive

echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "DEBUG: SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: RAY_CPUS: $((SLURM_CPUS_PER_TASK*SLURM_NNODES))"
echo "DEBUG: RAY_GPUS: $((SLURM_GPUS_PER_NODE*SLURM_NNODES))"
echo

ml --force purge
ml Python/3.11.5-GCCcore-13.2.0 
ml CMake/3.24.3-GCCcore-11.3.0
ml mpi4py
ml OpenMPI
ml CUDA/12.3
ml GCCcore/11.3.0
ml NCCL
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0 # this is needed by horovod!
module unload OpenSSL
# You should have CUDA 12.3 now

source $PYTHON_VENV/bin/activate

# Make mlpf visible
export PYTHONPATH="$PWD:$PYTHONPATH"

export ITWINAI_LOG_LEVEL=DEBUG

# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
export OMP_NUM_THREADS=1
if [ $SLURM_CPUS_PER_GPU -gt 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
fi

export NO_COLOR=1

# Launchers
torchrun_launcher(){
  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
    --no-python \
    --redirects=\$(((SLURM_NODEID)) && echo "3" || echo "1:3,2:3,3:3") \
    $1"
}

srun_launcher (){
    # Create mpirun logs folder
    mkdir -p "logs_srun/$SLURM_JOB_ID"

    # Launch command
    srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE \
        --cpus-per-task=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE)) \
        --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
        /bin/bash -c \
        'if [ $SLURM_PROCID  -ne 0 ]; then exec > "logs_srun/$SLURM_JOB_ID/rank.$SLURM_PROCID" 2>&1; fi; exec '"${1}"
}

ray_launcher(){

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
        ray start --head --node-ip-address="$head_node" --port=$port \
        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE"  --block &

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
            ray start --address "$head_node":"$port" --redis-password='5241580000000000' \
            --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &
        
        sleep 2 # Wait before starting the next worker to prevent race conditions.
    done
    echo All Ray workers started.

    # Run command without srun
    $1 
}


# Dual echo on both stdout and stderr
decho (){
  echo "$@"
  >&2 echo "$@"
}


######################   Initial checks   ######################

# Env vairables check
if [ -z "$DIST_MODE" ]; then 
  >&2 echo "ERROR: env variable DIST_MODE is not set. Allowed values are 'horovod', 'ddp' or 'deepspeed'"
  exit 1
fi
if [ -z "$RUN_NAME" ]; then 
  >&2 echo "WARNING: env variable RUN_NAME is not set. It's a way to identify some specific run of an experiment."
  RUN_NAME=$DIST_MODE
fi
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


# Get GPUs info per node
srun --cpu-bind=none --ntasks-per-node=1 bash -c 'echo -e "NODE hostname: $(hostname)\n$(nvidia-smi)\n\n"'

# Print env variables
echo "RUN_NAME: $RUN_NAME"
echo "DIST_MODE: $DIST_MODE"
# echo "CONTAINER_PATH: $CONTAINER_PATH"
# echo "COMMAND: $COMMAND"

######################   Execute command   ######################

if [ "${DIST_MODE}" == "ddp" ] ; then

  decho -e "\nLaunching DDP strategy with torchrun"
  torchrun_launcher "uv run python -u $PWD/mlpf/pipeline_itwinai.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
    --prefix itwinai_ddp_N_${SLURM_NNODES}_ \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --itwinai-strategy ddp \
    --num-epochs 2 \
    --itwinai-trainerv 3"


decho
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho


decho -e "\nLaunching Ray tests"
  ray_launcher "uv run python -u $PWD/mlpf/pipeline_itwinai.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
    --prefix itwinai_ddp_ray_N_${SLURM_NNODES}_ \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --itwinai-strategy ddp \
    --num-epochs 2 \
    --itwinai-trainerv 3"

elif [ "${DIST_MODE}" == "deepspeed" ] ; then

  decho -e "\nLaunching DeepSpeed strategy with torchrun"
  torchrun_launcher "uv run python -u $PWD/mlpf/pipeline_itwinai.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
    --prefix itwinai_deepspeed_N_${SLURM_NNODES}_ \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --itwinai-strategy deepspeed \
    --num-epochs 2 \
    --itwinai-trainerv 3"

decho
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho

ray_launcher "uv run python -u $PWD/mlpf/pipeline_itwinai.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
    --prefix itwinai_deepspeed_ray_N_${SLURM_NNODES}_ \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --itwinai-strategy deepspeed \
    --num-epochs 2 \
    --itwinai-trainerv 3"

  # decho -e "\nLaunching DeepSpeed strategy with mpirun"
  # mpirun_launcher "python -m ${COMMAND}"

  # decho -e "\nLaunching DeepSpeed strategy with srun"
  # srun_launcher "python -m ${COMMAND}"

elif [ "${DIST_MODE}" == "horovod" ] ; then

  # decho -e "\nLaunching Horovod strategy with mpirun"
  # mpirun_launcher "python -m ${COMMAND}"

  decho -e "\nLaunching Horovod strategy with srun"
  srun_launcher "uv run python -u $PWD/mlpf/pipeline_itwinai.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
    --prefix itwinai_horovod_N_${SLURM_NNODES}_ \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --itwinai-strategy horovod \
    --num-epochs 2 \
    --itwinai-trainerv 3"

decho
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
decho

ray_launcher "uv run python -u $PWD/mlpf/pipeline_itwinai.py \
    --train \
    --ray-train \
    --config parameters/pytorch/pyg-clic-itwinai.yaml \
    --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
    --prefix itwinai_horovod_ray_N_${SLURM_NNODES}_ \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_NNODES)) \
    --gpus $((SLURM_GPUS_PER_NODE*SLURM_NNODES)) \
    --gpu-batch-multiplier $BATCH_SIZE \
    --num-workers $((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE)) \
    --prefetch-factor 8 \
    --nvalid $N_VALID \
    --ntrain $N_TRAIN \
    --experiments-dir $PWD/$EXPERIMENTS_LOCATION \
    --itwinai-strategy horovod \
    --num-epochs 2 \
    --itwinai-trainerv 3"


else
  >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
  exit 1
fi
