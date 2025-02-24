#!/bin/bash

LOGS_SLURM="logs_slurm"
EXPERIMENTS="experiments_scaling"
REPLICAS=2
NODES_LIST="1 2 4 8 16"
T="02:15:00"
RUN_NAME="mlpf-pyg-ray-bl"
SCRIPT="scripts/vega/training_ray.sh"

echo "You are going to delete '$LOGS_SLURM' and '$EXPERIMENTS'."
read -p "Do you really want to delete the existing experiments and repeat the scaling test? [y/N] " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
  rm -rf $LOGS_SLURM
  mkdir $LOGS_SLURM
  rm -rf $EXPERIMENTS
else
  echo "Keeping existing logs."
fi

# Scaling test
for N in $NODES_LIST
do
    for (( i=0; i < $REPLICAS; i++  )); do
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT
    done
done



