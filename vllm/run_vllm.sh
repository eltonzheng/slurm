#!/bin/bash

# For conda:
source /opt/conda/etc/profile.d/conda.sh
conda activate vllm

# Print NCCL debug info
echo "----------------------------------------"
echo "NCCL Debug Information:"
python -c "import torch; print(f'NCCL Version: {torch.cuda.nccl.version()}')"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "Node: $(hostname)"

# Ray configuration
export RAY_HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export RAY_HEAD_PORT=6379
export RAY_ADDRESS_FILE=~/ray_head_address
export JOB_ID=$SLURM_JOB_ID  # Use SLURM job ID instead of timestamp
echo "Current JOB_ID: $JOB_ID"  # Debug print

echo "Ray head node: $RAY_HEAD_NODE"
echo "Node rank: $SLURM_NODEID"

# Start Ray based on node rank
if [ "$SLURM_NODEID" -eq 0 ]; then
    # Head node
    echo "Starting Ray head node..."
    # Clean up any existing address file
    rm -f $RAY_ADDRESS_FILE

    RAY_HEAD_ADDRESS=$(ray start --head 2>&1 | grep -oP '(?<=--address='"'"')\S+(?='"'"')')
    echo "Ray head address: $RAY_HEAD_ADDRESS"
    echo "Writing job ID $JOB_ID to file"  # Debug print
    echo "${JOB_ID}:${RAY_HEAD_ADDRESS}" > $RAY_ADDRESS_FILE
    echo "File contents: $(cat $RAY_ADDRESS_FILE)"  # Debug print
else
    # Worker node
    echo "Waiting for Ray head address file..."
    while true; do
        if [ -f $RAY_ADDRESS_FILE ]; then
            IFS=':' read -r FILE_JOB_ID RAY_HEAD_ADDRESS < $RAY_ADDRESS_FILE
            echo "Read JOB_ID: $FILE_JOB_ID"  # Debug print
            echo "Current JOB_ID: $JOB_ID"  # Debug print
            if [ "$FILE_JOB_ID" = "$JOB_ID" ]; then
                echo "Found valid Ray head address: $RAY_HEAD_ADDRESS"
                break
            fi
        fi
        sleep 2
        echo "Waiting for fresh address file..."
    done
    echo "Connecting to Ray head at: $RAY_HEAD_ADDRESS"

    # Start Ray in the background using nohup to detach it from the current shell
    ray start --address="$RAY_HEAD_ADDRESS" > ray_worker.log 2>&1 &

    echo "Ray worker started. Keeping the job alive..."
    # Keep the job alive so that SLURM does not clean up the background process
    tail -f /dev/null
fi

# Wait 10 seconds for Ray to initialize
sleep 10

# Only run vllm serve on the head node
if [ "$SLURM_NODEID" -eq 0 ]; then
    vllm serve /mnt/vast/deepseek/HF/DeepSeek-R1 \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 2 \
        --max-model-len 8192 \
        --trust-remote-code \
        --gpu-memory-utilization 0.8 \
        --host 0.0.0.0 \
        --port 40000
fi
