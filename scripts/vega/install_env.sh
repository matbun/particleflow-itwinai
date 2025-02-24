#!/bin/bash
# -*- coding: utf-8 -*-

REQS_FILE="requirements-itwinai.txt"

if [ ! -f "$REQS_FILE" ]; then
  echo "ERROR: $REQS_FILE not found!"
  exit 1
fi

# Load modules
# NOTE: REFLECT THEM IN THE MAIN README! 
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

# You should have CUDA 12.3 now

# Create and install torch env (uv installation: https://docs.astral.sh/uv/getting-started/installation/)
uv venv
uv pip install --no-cache-dir -r "$REQS_FILE" --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match

# # Legacy-style installation
# python -m venv .venv
# source .venv/bin/activate
# pip install --no-cache-dir -r "$REQS_FILE" --extra-index-url https://download.pytorch.org/whl/cu121
