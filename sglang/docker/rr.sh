#!/bin/bash

#srun -p hpc-mid \
srun -p hpc-interactive \
    -C gpu \
    --gpus=16 \
    --gpus-per-node=8 \
    --cpus-per-task=128 \
    --container-image=lmsysorg/sglang:latest \
    --container-mounts="/mnt/vast/deepseek/HF/:/models" \
    --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT \
    bash -i

    #bash -c 'bash /root/slurm/sglang/run_docker.sh'
