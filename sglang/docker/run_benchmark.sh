#!/bin/bash

#export SGLANG_TORCH_PROFILER_DIR=$HOME/profile_logs

num_runs=8

# Set ENGINE_DIR if provided, otherwise use "./engines"
MODEL_DIR=${1:-~/models}

# Set ENGINE_DIR if provided, otherwise use "./engines"
RESULT_FILE=${2:-./sglang_NDH100x16_benchmark_result}

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
        "deepseek-r1,fp8,/models/DeepSeek-R1"
    )
    in_out_sizes_mistral=("1:1024,256" "4:1024,256" "8:1024,256" "16:1024,256" "32:1024,256" "64:1024,256" "128:1024,256" "1:2048,256" "4:2048,256" "8:2048,256" "16:2048,256" "32:2048,256" "64:2048,256" "128:2048,256")
fi

#in_out_sizes_mistral=("1:1024,256")

json_to_csv() {
    # Read input JSON file
    local json_file=$1
    local csv_file=$2
    local model_name=$3
    local data_type=$4

    # Write CSV header if file doesn't exist
    if [ ! -f "$csv_file" ]; then
        echo "Models,DataType,Batch,InputLen,OutputLen,PromptTokenPerSec,GenerationTokenPerSec,TimeToFirstToken(ms),InterTokenLatency(ms)" > "$csv_file"
    fi

    # Create temporary arrays to store values
    declare -A values_ttft
    declare -A values_tpot
    declare -A batch_sizes
    declare -A input_lens
    declare -A output_lens

    # First pass: collect all values for each configuration
    while IFS= read -r line; do
        batch=$(echo "$line" | grep -o '"max_concurrency": [0-9]*' | awk '{print $2}')
        input_len=$(echo "$line" | grep -o '"random_input_len": [0-9]*' | awk '{print $2}')
        output_len=$(echo "$line" | grep -o '"random_output_len": [0-9]*' | awk '{print $2}')
        ttft=$(echo "$line" | grep -o '"mean_ttft_ms": [0-9.]*' | awk '{print $2}')
        tpot=$(echo "$line" | grep -o '"mean_tpot_ms": [0-9.]*' | awk '{print $2}')

        # Create a key for this configuration
        key="${batch}_${input_len}_${output_len}"

        # Store the configuration values
        batch_sizes[$key]=$batch
        input_lens[$key]=$input_len
        output_lens[$key]=$output_len

        # Append values to arrays
        values_ttft[$key]="${values_ttft[$key]:-}${ttft} "
        values_tpot[$key]="${values_tpot[$key]:-}${tpot} "
    done < "$json_file"

    # Second pass: calculate averages excluding min and max
    sorted_keys=( $(for key in "${!batch_sizes[@]}"; do echo "$key"; done | sort -t '_' -k2,2n -k1,1n) )

    for key in "${sorted_keys[@]}"; do
        # Convert space-separated strings to arrays
        ttft_array=(${values_ttft[$key]})
        tpot_array=(${values_tpot[$key]})

        # Need at least 3 values to exclude min and max
        if [ ${#ttft_array[@]} -ge 3 ]; then
            echo "Processing key: $key"
            echo "Original TTFT values: ${ttft_array[*]}"
            echo "Original TPOT values: ${tpot_array[*]}"

            # Sort arrays numerically
            IFS=$'\n' ttft_sorted=($(sort -n <<<"${ttft_array[*]}"))
            IFS=$'\n' tpot_sorted=($(sort -n <<<"${tpot_array[*]}"))
            unset IFS

            echo "Sorted TTFT values: ${ttft_sorted[*]}"
            echo "Sorted TPOT values: ${tpot_sorted[*]}"

            # Remove first (min) and last (max) elements
            ttft_sorted=("${ttft_sorted[@]:1:${#ttft_sorted[@]}-2}")
            tpot_sorted=("${tpot_sorted[@]:1:${#tpot_sorted[@]}-2}")

            echo "After removing min/max TTFT: ${ttft_sorted[*]}"
            echo "After removing min/max TPOT: ${tpot_sorted[*]}"

            # Calculate averages
            ttft_sum=0
            tpot_sum=0
            for i in "${!ttft_sorted[@]}"; do
                ttft_sum=$(echo "$ttft_sum + ${ttft_sorted[$i]}" | bc)
                tpot_sum=$(echo "$tpot_sum + ${tpot_sorted[$i]}" | bc)
            done

            avg_ttft=$(echo "scale=2; $ttft_sum / ${#ttft_sorted[@]}" | bc)
            avg_tpot=$(echo "scale=2; $tpot_sum / ${#tpot_sorted[@]}" | bc)

            echo "Final averages - TTFT: $avg_ttft, TPOT: $avg_tpot"
            echo "----------------------------------------"

            # Calculate throughputs using averages
            input_throughput=$(echo "(${batch_sizes[$key]} * ${input_lens[$key]} * 1000) / $avg_ttft" | bc | awk '{printf "%.0f", $1}')
            output_throughput=$(echo "(${batch_sizes[$key]} * 1000) / $avg_tpot" | bc | awk '{printf "%.0f", $1}')

            # Write merged results to CSV
            echo "$model_name,$data_type,${batch_sizes[$key]},${input_lens[$key]},${output_lens[$key]},$input_throughput,$output_throughput,$avg_ttft,$avg_tpot" >> "$csv_file"
        fi
    done
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

        #BACKEND="vllm"
        BACKEND="sglang"
        for ((run=1; run<=num_runs; run++)); do
            echo "Run $run of $num_runs"
            python3 -m sglang.bench_serving \
                --backend $BACKEND \
                --model ./tokenizer \
                --num-prompt $batch_size \
                --port 40000 \
                --random-range-ratio 1.0 \
                --random-input-len ${in_out_pair[0]} \
                --random-output-len ${in_out_pair[1]} \
                --dataset-name random \
                --max-concurrency $batch_size \
                --output-file ${json_file}
        done
    done

	json_to_csv "$json_file" "${RESULT_FILE}_${PACKAGE_VERSION}.csv" "$model_name" "$data_type"
}

for model in "${models[@]}"
do
    IFS=',' read -r model_name model_type model_dir <<< "$model"
    test_model "$model_name" "$model_type" "$model_dir"
done
