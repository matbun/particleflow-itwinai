#!/bin/bash

env(){
    # Load env modules

    ml --force purge
    ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
    ml Python CMake HDF5 PnetCDF libaio mpi4py
}

alloc(){
    # Allocate a node interactively
    
    salloc \
        --partition=develbooster \
        --account=intertwin  \
        --nodes=1 \
        --gres=gpu:4 \
        --gpus-per-node=4 \
        --time=1:59:00 \
        --cpus-per-task=24 \
        --ntasks-per-node=1
}

term(){
    # Open a terminal in the allocated node
    
    srun --jobid $1 --overlap --pty /bin/bash
}

ray(){
    # Create a dumy Ray cluster of 2 nodes

    uv run ray stop
    uv run ray start \
        --head \
        --node-ip-address=127.0.0.1 \
        --port=7639 \
        --num-cpus 1
    echo "HEAD NODE STARTED" 
    uv run ray start \
        --address=127.0.0.1:7639 \
        --num-cpus 1
    echo "WORKER NODE STARTED" 
}

run(){
    # CPU-only execution on login node
    
    RAY_CPUS=32
    RAY_GPUS=0
    
    uv run ray stop
    uv run ray start \
        --head \
        --node-ip-address=localhost \
        --port=7639 \
        --num-cpus=$RAY_CPUS \
        --num-gpus=$RAY_GPUS 
        # --block &
    echo "RAY STARTED"

    # Make mlpf visible
    export PYTHONPATH="$PWD:$PYTHONPATH"
    
    uv run python -u $PWD/mlpf/pipeline.py \
        --train \
        --ray-train \
        --config parameters/pytorch/pyg-clic-itwinai.yaml \
        --data-dir /p/scratch/intertwin/datasets/clic/ \
        --ntrain 50 \
        --nvalid 50 \
        --prefix foo_prefix \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --num-epochs 2
}

export ITWINAI_LOG_LEVEL=DEBUG

run_itwinai(){

    RAY_CPUS=32
    RAY_GPUS=0

    uv run ray stop

    # Make mlpf visible
    export PYTHONPATH="$PWD:$PYTHONPATH"

    uv run python -u \
        $PWD/mlpf/pipeline_itwinai.py \
        --train \
        --ray-train \
        --config parameters/pytorch/pyg-clic-itwinai.yaml \
        --data-dir /p/scratch/intertwin/datasets/clic/ \
        --ntrain 50 \
        --nvalid 50 \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --prefix foo_prefix \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --num-epochs 2  \
        --itwinai-trainerv 3
}

run_itwinai_ray(){

    RAY_CPUS=32
    RAY_GPUS=0
    
    # uv run ray stop
    # uv run ray start \
    #     --head \
    #     --node-ip-address=localhost \
    #     --port=7639 \
    #     --num-cpus=$RAY_CPUS \
    #     --num-gpus=$RAY_GPUS 
    #     # --block &
    # echo "RAY STARTED"

    # Make mlpf visible
    export PYTHONPATH="$PWD:$PYTHONPATH"

    # uv run python -Xfrozen_modules=off -m debugpy --listen 5678 --wait-for-client  \
    uv run python -u \
        $PWD/mlpf/pipeline_itwinai.py \
        --train \
        --ray-train \
        --config parameters/pytorch/pyg-clic-itwinai.yaml \
        --data-dir /p/scratch/intertwin/datasets/clic/ \
        --ntrain 50 \
        --nvalid 50 \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --prefix foo_prefix \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --num-epochs 2 \
        --itwinai-trainerv 3
}