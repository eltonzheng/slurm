#!/bin/bash

MODEL=/models/DeepSeek-R1
TP=16
EP=1

#srun -p hpc-mid \
srun -p hpc-interactive \
    -C gpu \
    --gpus=16 \
    --gpus-per-node=8 \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --gres=gpu:8 \
    --container-image=/mnt/home/eltonz/trtllm_0225.sqsh \
    --container-mounts="/mnt/vast/deepseek/HF/:/models,/mnt/home/eltonz/TensorRT-LLM:/app" \
    --mpi=pmi2 \
    bash -c '/app/tensorrt_llm/llmapi/trtllm-llmapi-launch \
            python /app/examples/pytorch/quickstart_advanced.py --model_dir /models/DeepSeek-R1 \
            --tp_size 16 --moe_ep_size 1 \
            --enable_overlap_scheduler \
            --enable_attention_dp'

    #--container-image=nvcr.io\#nvidia/pytorch:25.01-py3 \
    #--container-image=/mnt/home/eltonz/trtllm_0221.sqsh \
    #--container-image=nvcr.io\#nvidia/tritonserver:25.01-trtllm-python-py3 \
    #bash -c 'bash /root/slurm/trtllm/run_docker.sh'


