#!/bin/bash

# For conda:
#source /opt/conda/etc/profile.d/conda.sh
#conda activate deepseek

# NCCL Configuration
#export NCCL_IB_DISABLE=0         # Enable InfiniBand
#export NCCL_NET_GDR_LEVEL=2      # Enable GPUDirect RDMA
#export NCCL_P2P_DISABLE=0        # Enable P2P between GPUs
#export NCCL_DEBUG=INFO           # Enable NCCL debugging
#export NCCL_SOCKET_IFNAME=ibp0,eth0

#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=0
#export NCCL_NET_GDR_LEVEL=2
#export NCCL_SOCKET_IFNAME=ibp0,ibp1,ibp2,ibp3,ibp4,ibp5,ibp6,ibp7,eth0
#export NCCL_IB_HCA=mlx5               # Specify the HCA type
#export NCCL_IB_TIMEOUT=22             # Timeout in seconds

#export NCCL_IB_DISABLE=0         # Enable IB (default is 0)
#export NCCL_SOCKET_IFNAME=ib0    # Use IB interface (change ib0 if necessary)
#export NCCL_NET_GDR_LEVEL=PHB    # Enable GPU Direct RDMA (GDR)
#export NCCL_P2P_DISABLE=0        # Enable peer-to-peer GPU communication

# Print NCCL debug info
echo "----------------------------------------"
echo "NCCL Debug Information:"
python3 -c "import torch; print(f'NCCL Version: {torch.cuda.nccl.version()}')"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "Node: $(hostname)"

export MASTER_PORT=5000

# Set persistent cache directories
export OUTLINES_CACHE_DIR=./node_cache_${SLURM_NODEID}
export TORCH_INDUCTOR_CACHE_DIR=./torch_cache_${SLURM_NODEID}

# Create cache directories if they don't exist
mkdir -p $OUTLINES_CACHE_DIR
mkdir -p $TORCH_INDUCTOR_CACHE_DIR

echo "Using cache directory: $OUTLINES_CACHE_DIR"
echo "Using cache directory: $TORCH_INDUCTOR_CACHE_DIR"

get_first_node() {
    local nodelist=$1
    local first_node=""

    if [[ $nodelist =~ .*,.* ]]; then
        # Format: h100-183-003,h100-202-005
        first_node=$(echo $nodelist | cut -d',' -f1)
    elif [[ $nodelist =~ .*\[.*\].* ]]; then
        # Format: h100-183-[003-004]
        # Extract the base and first number
        local base=$(echo $nodelist | sed 's/\[.*\]//')
        local first_num=$(echo $nodelist | grep -o '\[.*\]' | tr -d '[]' | cut -d'-' -f1)
        first_node="${base}${first_num}"
    else
        # Single node format: h100-183-003
        first_node=$nodelist
    fi

    echo $first_node
}

export HEAD_NODE=$(get_first_node "$SLURM_JOB_NODELIST")
#export SGLANG_TORCH_PROFILER_DIR=$HOME/profile_logs
#/root/nsight-systems-2025.1.1/bin/nsys launch --trace=cuda,nvtx,osrt,cudnn,cublas \
python3 -m sglang.launch_server \
    --model-path /models/DeepSeek-R1 \
    --tp 16 \
    --dist-init-addr $HEAD_NODE:$MASTER_PORT \
    --nnodes 2 \
    --node-rank $SLURM_NODEID \
    --trust-remote-code \
    --enable-torch-compile --torch-compile-max-bs 8 \
    --host 0.0.0.0 \
    --port 40000
