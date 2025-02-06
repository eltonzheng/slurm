#!/bin/bash

srun -p hpc-interactive \
    -C gpu \
    --gpus=16 \
    --gpus-per-node=8 \
    --cpus-per-task=128 \
    ./run_sglang.sh
