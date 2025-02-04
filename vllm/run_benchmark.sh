#!/bin/bash

num_runs=10

# Set ENGINE_DIR if provided, otherwise use "./engines"
MODEL_DIR=${1:-~/models}

# Set ENGINE_DIR if provided, otherwise use "./engines"
RESULT_FILE=${2:-./vllm_NDH100x16_benchmark_result}

# Function to detect GPU type
detect_gpu_type()
{
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    if [[ $gpu_info == *"A100"* ]]; then
        mig_info=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader)
        if [[ $mig_info == "Enabled" ]]; then
            echo "A100-MIG"
        else
            echo "A100"
        fi
    elif [[ $gpu_info == *"H100"* ]]; then
        echo "H100"
    elif [[ $gpu_info == *"A10"* ]]; then
        echo "A10"
    elif [[ $gpu_info == *"T4"* ]]; then
        echo "T4"
    elif [[ $gpu_info == *"V100"* ]]; then
        echo "V100"
    else
        echo "Unknown GPU type"
        exit 1
    fi
}

extra_options=""

# Get the CUDA architecture based on GPU type
CUDA_ARCHITECTURE=$(detect_gpu_type)

if [[ "$CUDA_ARCHITECTURE" == "T4" ]]; then
    models=(
        "llama2-7b,gptq,$MODEL_DIR/llama2-7b-gptq"
        "llama2-7b,awq,$MODEL_DIR/llama2-7b-awq"
        "llama3-8b,gptq,$MODEL_DIR/Meta-Llama-3-8B-gptq"
        "llama3-8b,awq,$MODEL_DIR/Meta-Llama-3-8B-awq"
        "mistral-7b-v0.3,gptq,$MODEL_DIR/Mistral-7B-Instruct-v0.3-gptq"
        "mistral-7b-v0.3,awq,$MODEL_DIR/Mistral-7B-Instruct-v0.3-awq"
        "phi-2,fp16,$MODEL_DIR/phi-2"
        "phi-3-mini-4k,fp16,$MODEL_DIR/phi-3-mini-4k-instruct"
        "qwen2-0.5b,fp16,$MODEL_DIR/Qwen2-0.5B-Instruct"
        "qwen2-0.5b,gptq,$MODEL_DIR/Qwen2-0.5B-Instruct-gptq"
        "qwen2-0.5b,awq,$MODEL_DIR/Qwen2-0.5B-Instruct-awq"
        "qwen2-1.5b,fp16,$MODEL_DIR/Qwen2-1.5B-Instruct"
        "qwen2-1.5b,gptq,$MODEL_DIR/Qwen2-1.5B-Instruct-gptq"
        "qwen2-1.5b,awq,$MODEL_DIR/Qwen2-1.5B-Instruct-awq"
    )
    in_out_sizes_mistral=("1:2048,256" "1:4096,256")
    in_out_sizes_llama2=("1:1024,256" "1:2048,256")
elif [[ "$CUDA_ARCHITECTURE" == "A10" ]]; then
    models=(
        "llama2-7b,fp16,$MODEL_DIR/llama2-7b-hf"
        "llama2-7b,gptq,$MODEL_DIR/llama2-7b-gptq"
        "llama2-7b,awq,$MODEL_DIR/llama2-7b-awq"
        "llama3-8b,fp16,$MODEL_DIR/Meta-Llama-3-8B"
        "llama3-8b,gptq,$MODEL_DIR/Meta-Llama-3-8B-gptq"
        "llama3-8b,awq,$MODEL_DIR/Meta-Llama-3-8B-awq"
        "mistral-7b-v0.3,fp16,$MODEL_DIR/Mistral-7B-Instruct-v0.3"
        "mistral-7b-v0.3,gptq,$MODEL_DIR/Mistral-7B-Instruct-v0.3-gptq"
        "mistral-7b-v0.3,awq,$MODEL_DIR/Mistral-7B-Instruct-v0.3-awq"
        "phi-2,fp16,$MODEL_DIR/phi-2"
        "phi-3-mini-4k,fp16,$MODEL_DIR/phi-3-mini-4k-instruct"
        "qwen2-0.5b,fp16,$MODEL_DIR/Qwen2-0.5B-Instruct"
        "qwen2-0.5b,gptq,$MODEL_DIR/Qwen2-0.5B-Instruct-gptq"
        "qwen2-0.5b,awq,$MODEL_DIR/Qwen2-0.5B-Instruct-awq"
        "qwen2-1.5b,fp16,$MODEL_DIR/Qwen2-1.5B-Instruct"
        "qwen2-1.5b,gptq,$MODEL_DIR/Qwen2-1.5B-Instruct-gptq"
        "qwen2-1.5b,awq,$MODEL_DIR/Qwen2-1.5B-Instruct-awq"
    )
    in_out_sizes_mistral=("1:2048,256" "4:2048,256" "8:2048,256" "1:4096,256" "4:4096,256" "8:4096,256")
    in_out_sizes_llama2=("1:1024,256" "4:1024,256" "8:1024,256" "1:2048,256" "4:2048,256" "8:2048,256")
    extra_options+=" --enforce-eager"
else
    models=(
        #"llama2-7b,fp16,$MODEL_DIR/llama2-7b-hf"
        #"llama2-7b,fp8,$MODEL_DIR/llama2-7b-hf"
        #"llama2-7b,gptq,$MODEL_DIR/llama2-7b-gptq"
        #"llama2-7b,awq,$MODEL_DIR/llama2-7b-awq"
        #"llama3-8b,fp16,$MODEL_DIR/Meta-Llama-3-8B"
        #"llama3-8b,fp8,$MODEL_DIR/Meta-Llama-3-8B"
        #"llama3-8b,gptq,$MODEL_DIR/Meta-Llama-3-8B-gptq"
        #"llama3-8b,awq,$MODEL_DIR/Meta-Llama-3-8B-awq"
        #"mistral-7b-v0.3,fp16,$MODEL_DIR/Mistral-7B-Instruct-v0.3"
        #"mistral-7b-v0.3,fp8,$MODEL_DIR/Mistral-7B-Instruct-v0.3"
        #"mistral-7b-v0.3,gptq,$MODEL_DIR/Mistral-7B-Instruct-v0.3-gptq"
        #"mistral-7b-v0.3,awq,$MODEL_DIR/Mistral-7B-Instruct-v0.3-awq"
        #"mixtral-8x7b,fp16,$MODEL_DIR/Mixtral-8x7B-v0.1"
        #"phi-3-mini-4k,fp16,$MODEL_DIR/phi-3-mini-4k-instruct"
        #"phi-3-mini-4k,fp8,$MODEL_DIR/phi-3-mini-4k-instruct"
        #"phi-4,fp16,$MODEL_DIR/phi-4"
        #"phi-4,fp8,$MODEL_DIR/phi-4"
        #"qwen2-0.5b,fp16,$MODEL_DIR/Qwen2-0.5B-Instruct"
        #"qwen2-0.5b,fp8,$MODEL_DIR/Qwen2-0.5B-Instruct"
        #"qwen2-0.5b,gptq,$MODEL_DIR/Qwen2-0.5B-Instruct-gptq"
        #"qwen2-0.5b,awq,$MODEL_DIR/Qwen2-0.5B-Instruct-awq"
        #"qwen2-1.5b,fp16,$MODEL_DIR/Qwen2-1.5B-Instruct"
        #"qwen2-1.5b,fp8,$MODEL_DIR/Qwen2-1.5B-Instruct"
        #"qwen2-1.5b,gptq,$MODEL_DIR/Qwen2-1.5B-Instruct-gptq"
        #"qwen2-1.5b,awq,$MODEL_DIR/Qwen2-1.5B-Instruct-awq"
        #"deepseek-r1-dist-qwen-32b,fp16,$MODEL_DIR/DeepSeek-R1-Distill-Qwen-32B"
        #"deepseek-r1-dist-qwen-32b,fp8,$MODEL_DIR/DeepSeek-R1-Distill-Qwen-32B"
        "deepseek-r1,fp8,/mnt/vast/deepseek/HF/DeepSeek-R1"
    )
    in_out_sizes_mistral=("1:1024,256" "4:1024,256" "8:1024,256" "16:1024,256" "32:1024,256" "64:1024,256" "128:1024,256" "1:2048,256" "4:2048,256" "8:2048,256" "16:2048,256" "1:4096,256" "4:4096,256" "8:4096,256" "16:4096,256")
    #in_out_sizes_mistral=("32:1024,256" "64:1024,256" "128:1024,256")
fi

#in_out_sizes_llama2=("1:4096,256")
#in_out_sizes_mistral=("1:1024,256")

json_to_csv() {
    # Read input JSON file
    local json_file=$1
    local csv_file=$2
    local model_name=$3

    # Write CSV header if file doesn't exist
    if [ ! -f "$csv_file" ]; then
        echo "Models,DataType,Batch,InputLen,OutputLen,PromptTokenPerSec,GenerationTokenPerSec,TimeToFirstToken(ms),InterTokenLatency(ms)" > "$csv_file"
    fi

    # Process JSON and extract values
    while IFS= read -r line; do
        # Extract values using jq or awk
        batch=$(echo "$line" | grep -o '"max_concurrency": [0-9]*' | awk '{print $2}')
        input_len=$(echo "$line" | grep -o '"random_input_len": [0-9]*' | awk '{print $2}')
        output_len=$(echo "$line" | grep -o '"random_output_len": [0-9]*' | awk '{print $2}')
        ttft=$(echo "$line" | grep -o '"mean_ttft_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
        tpot=$(echo "$line" | grep -o '"mean_tpot_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
        input_throughput=$(echo "($batch * $input_len * 1000) / $ttft" | bc | awk '{printf "%.0f", $1}')
        output_throughput=$(echo "1000 / $tpot" | bc | awk '{printf "%.0f", $1}')

        # Write to CSV
        echo "$model_name,fp8,$batch,$input_len,$output_len,$input_throughput,$output_throughput,$ttft,$tpot" >> "$csv_file"
    done < "$json_file"
}

function test_model() {
    local model_name="$1"
    local data_type="$2"
    local model_dir="$3"

    local in_out_sizes_var="in_out_sizes_mistral"
    local tp=1

    if [[ "$CUDA_ARCHITECTURE" != *"H100"* && $data_type == "fp8" ]]; then
        return
    fi

    if [[ $data_type == *"fp16"* ]]; then
        quant_option="--dtype float16"
    else
        quant_option="--quantization $data_type"
    fi

    if [[ $model_name == *"llama2"* || $model_name == *"phi"* || $model_name == *"qwen"* ]]; then
        in_out_sizes_var="in_out_sizes_llama2"
    fi

    if [[ $model_name == *"mixtral"* ]]; then
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        tp=2
    fi

    eval in_out_sizes=(\${$in_out_sizes_var[@]})

    PACKAGE_VERSION=$(pip show sglang 2>/dev/null | grep -oP 'Version: \K[0-9]+\.[0-9]+\.[0-9]+')
	json_file=${RESULT_FILE}_${PACKAGE_VERSION}.jsonl

    for in_out in "${in_out_sizes[@]}"
    do
        batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
        in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')

        # Replace commas with spaces, then use read to put each word into an array
        read -ra in_out_pair <<< "$(echo "$in_out_dims" | tr ',' ' ')"

        echo ""
        echo "==========================================================================================="
        echo "Python Benchmark - Model: $model_name, Type: $data_type, BS: $batch_size, ISL/OSL: ${in_out_pair[0]}/${in_out_pair[1]}"
        echo "==========================================================================================="

        BACKEND="vllm"
        #BACKEND="sglang"
        python3 -m sglang.bench_serving \
            --backend $BACKEND \
            --num-prompt $((num_runs * batch_size)) \
            --port 40000 \
            --random-range-ratio 1.0 \
            --random-input-len ${in_out_pair[0]} \
            --random-output-len ${in_out_pair[1]} \
            --dataset-name random \
            --max-concurrency $batch_size \
            --model $model_dir \
            --output-file ${json_file}
    done

	json_to_csv "$json_file" "${RESULT_FILE}_${PACKAGE_VERSION}.csv" "$model_name"
}

for model in "${models[@]}"
do
    IFS=',' read -r model_name model_type model_dir <<< "$model"
    test_model "$model_name" "$model_type" "$model_dir"
done

