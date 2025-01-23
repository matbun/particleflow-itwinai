#!/bin/bash

env(){
    # Load env modules

    ml --force purge
    ml Python/3.11.5-GCCcore-13.2.0 
    ml CMake/3.24.3-GCCcore-11.3.0
    ml mpi4py
    ml OpenMPI
    ml CUDA/12.3
    ml GCCcore/11.3.0
    ml NCCL
    ml cuDNN/8.9.7.29-CUDA-12.3.0
    ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
    module unload OpenSSL
}

alloc(){
    # Allocate a node interactively
    
    salloc \
        --partition=gpu \
        --account=d2024d11-083-users  \
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

run(){
    # CPU-only execution on login node
    
    RAY_CPUS=32
    RAY_GPUS=1
    
    uv run ray stop
    uv run ray start \
        --head \
        --node-ip-address=localhost \
        --port=7639 \
        --num-cpus=$RAY_CPUS \
        --num-gpus=$RAY_GPUS 
        # --block &
    echo "RAY STARTED"
    
    uv run python -u $PWD/mlpf/pyg_pipeline.py \
        --train \
        --ray-train \
        --config parameters/pytorch/pyg-clic-itwinai.yaml \
        --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clusters \
        --ntrain 500 \
        --nvalid 500 \
        --prefix foo_prefix \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --local \
        --num-epochs 2
}

run_itwinai(){

    RAY_CPUS=32
    RAY_GPUS=1

    uv run python -u \
        $PWD/mlpf/pyg_pipeline_itwinai.py \
        --train \
        --ray-train \
        --config parameters/pytorch/pyg-clic-itwinai.yaml \
        --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clusters \
        --ntrain 500 \
        --nvalid 500 \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --prefix foo_prefix \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --local \
        --experiments-dir $PWD/experiments \
        --num-epochs 2
}