#!/bin/bash

# Set ENGINE_DIR if provided, otherwise use "./engines"
MODEL_DIR=${1:-~/models}

# Set ENGINE_DIR if provided, otherwise use "./engines"
RESULT_FILE=${2:-./sglang_benchmark_result_$(date +%Y%m%d_%H%M%S)}

# Function to extract values from a output log file and save to a CSV file
json_to_csv() {
    local json_file=$1
    local csv_file=$2
    local model=$3
    local backend=$4
    local expected_concurrency=$5

    # Read the last line from the JSONL file (most recent record)
    local json_line=$(tail -n 1 "$json_file")

    # Extract values using grep and awk, format floats to 2 decimal places
    local completed=$(echo "$json_line" | grep -o '"completed": [0-9]*' | awk '{print $2}')
    local concurrency=$(echo "$json_line" | grep -o '"concurrency": [0-9.]*' | awk '{printf "%.2f", $2}')
    local duration=$(echo "$json_line" | grep -o '"duration": [0-9.]*' | awk '{printf "%.2f", $2}')
    local total_input_tokens=$(echo "$json_line" | grep -o '"total_input_tokens": [0-9]*' | awk '{print $2}')
    local total_output_tokens=$(echo "$json_line" | grep -o '"total_output_tokens": [0-9]*' | awk '{print $2}')
    local request_throughput=$(echo "$json_line" | grep -o '"request_throughput": [0-9.]*' | awk '{printf "%.2f", $2}')
    local input_throughput=$(echo "$json_line" | grep -o '"input_throughput": [0-9.]*' | awk '{printf "%.2f", $2}')
    local output_throughput=$(echo "$json_line" | grep -o '"output_throughput": [0-9.]*' | awk '{printf "%.2f", $2}')
    local mean_ttft=$(echo "$json_line" | grep -o '"mean_ttft_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
    local median_ttft=$(echo "$json_line" | grep -o '"median_ttft_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
    local p99_ttft=$(echo "$json_line" | grep -o '"p99_ttft_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
    local mean_tpot=$(echo "$json_line" | grep -o '"mean_tpot_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
    local median_tpot=$(echo "$json_line" | grep -o '"median_tpot_ms": [0-9.]*' | awk '{printf "%.2f", $2}')
    local p99_tpot=$(echo "$json_line" | grep -o '"p99_tpot_ms": [0-9.]*' | awk '{printf "%.2f", $2}')

    # Write to CSV file
    {
        echo "Counter Name,Value"
        echo "Model,$model"
        echo "Backend,$backend"
        echo "Successful requests,$completed"
        echo "Expected Concurrency,$expected_concurrency"
        echo "Actual Concurrency,$concurrency"
        echo "Benchmark duration (s),$duration"
        echo "Total input tokens,$total_input_tokens"
        echo "Total generated tokens,$total_output_tokens"
        echo "Request throughput (req/s),$request_throughput"
        echo "Input token throughput (tok/s),$input_throughput"
        echo "Output token throughput (tok/s),$output_throughput"
        echo "Mean TTFT (ms),$mean_ttft"
        echo "Median TTFT (ms),$median_ttft"
        echo "P99 TTFT (ms),$p99_ttft"
        echo "Mean TPOT (ms),$mean_tpot"
        echo "Median TPOT (ms),$median_tpot"
        echo "P99 TPOT (ms),$p99_tpot"
    } > "$csv_file"
}

function run_benchmark() {
    declare -n model_config=$1
    
    local model=${model_config["model"]}
    local tokenizer=${model_config["tokenizer"]}
    local num_prompts=${model_config["num_prompts"]}
    local base_url=${model_config["base_url"]}
    local concurrency_list=(${model_config["concurrency"]})
    
    PACKAGE_VERSION=$(pip show sglang 2>/dev/null | grep -oP 'Version: \K[0-9]+\.[0-9]+\.[0-9]+')

    for concurrency in "${concurrency_list[@]}"; do
        echo ""
        echo "==========================================================================================="
        echo "Python Benchmark - Model: $model, Concurrency: $concurrency"
        echo "==========================================================================================="

        total_num_prompts=$(( num_prompts * concurrency ))
        total_num_prompts=$(( total_num_prompts > 10000 ? 10000 : total_num_prompts ))
        json_file=${RESULT_FILE}_${model}_${PACKAGE_VERSION}_${concurrency}.jsonl

        python3 -m sglang.bench_serving \
            --backend papyrus \
            --model $model \
            --tokenizer $tokenizer \
            --num-prompt $total_num_prompts \
            --base-url $base_url \
            --dataset-name sharegpt \
            --max-concurrency $concurrency \
            --output-file ${json_file}

        json_to_csv "$json_file" "${RESULT_FILE}_${PACKAGE_VERSION}_${model}_${concurrency}.csv" "$model" "papyrus" "$concurrency"
    done

}

DATASET_PATH=./ShareGPT_V3_unfiltered_cleaned_split.json
NUM_PROMPTS=100
CONCURRENCY_LIST=(1 4 8 16 32 64 128 256 512)
#CONCURRENCY_LIST=(1)
TOKENIZER_PATH="/data/benchmark/models/deepseek-v3"
BASE_URL="https://WestUS2Large.papyrus.binginternal.com/chat/completions"

declare -A deepseekr1_eval
deepseekr1_eval["model"]=deepseekr1-eval
deepseekr1_eval["tokenizer"]=$TOKENIZER_PATH
deepseekr1_eval["num_prompts"]=$NUM_PROMPTS
deepseekr1_eval["base_url"]=$BASE_URL
deepseekr1_eval["concurrency"]=${CONCURRENCY_LIST[@]}

declare -A deepseekr1_batch
deepseekr1_batch["model"]=deepseekr1-batch
deepseekr1_batch["tokenizer"]=$TOKENIZER_PATH
deepseekr1_batch["num_prompts"]=$NUM_PROMPTS
deepseekr1_batch["base_url"]=$BASE_URL
deepseekr1_batch["concurrency"]=${CONCURRENCY_LIST[@]}

run_benchmark deepseekr1_eval &
run_benchmark deepseekr1_batch &