#!/bin/bash

#srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 \
#    -p hpc-interactive \
#    ./run_sglang.sh

srun -p hpc-high \
    -C gpu \
    --gpus=16 \
    --gpus-per-node=8 \
    --cpus-per-task=128 \
    ./run_sglang.sh
