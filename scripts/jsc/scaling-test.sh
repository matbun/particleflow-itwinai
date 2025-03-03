#!/bin/bash

LOGS_SLURM="logs_slurm"
EXPERIMENTS="experiments_scaling"
REPLICAS=1
NODES_LIST="1 2 4" #"1 2 4 8 16"
T="02:00:00" #"00:10:00"
# RUN_NAME="mlpf-pyg-ray-bl"
SCRIPT="scripts/jsc/slurm.jsc.sh"
BASELINE_SCRIPT="scripts/jsc/training_ray.sh"

# Variables for SLURM script
export EXPERIMENTS_LOCATION=$EXPERIMENTS
export BATCH_SIZE=32 #32
export N_TRAIN=100000 #500 #700000

# NOTE: remember to check how many GPUs per node were requested in the slurm scripts!

echo "You are going to delete '$LOGS_SLURM' and '$EXPERIMENTS'."
read -p "Do you really want to delete the existing experiments and repeat the scaling test? [y/N] " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
  rm -rf $LOGS_SLURM logs_torchrun logs_srun mllogs scalability-metrics plots
  mkdir $LOGS_SLURM
  rm -rf $EXPERIMENTS
else
  echo "Keeping existing logs."
fi


# Scaling test
for N in $NODES_LIST
do
    for (( i=0; i < $REPLICAS; i++  )); do

        # Validation data should be just enough so that all workers receive a bit
        export N_VALID=$((500*N))

        export DIST_MODE="ddp"
        export RUN_NAME="ddp-itwinai-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT

        export DIST_MODE="deepspeed"
        export RUN_NAME="ds-itwinai-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT

        export DIST_MODE="horovod"
        export RUN_NAME="horovod-itwinai-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT

        # Baseline without itwinai
        export RUN_NAME="baseline-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $BASELINE_SCRIPT

    done
done



