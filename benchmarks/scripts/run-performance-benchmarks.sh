#!/bin/bash

set -xe

BENCHMARK_SERVING_PATH=/mnt/deepseek/benchmarks

check_npus() {
  # shellcheck disable=SC2155
  declare -g npu_count=$(npu-smi info -l | grep "Total Count" | awk -F ':' '{print $2}' | tr -d ' ')
  
  if [[ -z "$npu_count" || "$npu_count" -eq 0 ]]; then
    echo "Need at least 1 NPU to run benchmarking."
    exit 1
  else
    echo "found NPU conut: $npu_count"
  fi

  npu_type=$(npu-smi info | grep -E "^\| [0-9]+" | awk -F '|' '{print $2}' | awk '{$1=$1;print}' | awk '{print $2}')

  echo "NPU type is: $npu_type"
}

ensure_sharegpt_downloaded() {
  local FILE="/github/home/.cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
  local DIR
  DIR=$(dirname "$FILE")

  if [ ! -f "$FILE" ]; then
    echo "$FILE not found, downloading from hf-mirror ..."
    mkdir -p "$DIR"
    wget -O "$FILE" https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    if [ $? -ne 0 ]; then
      echo "Download failed!" >&2
      return 1
    fi
    echo "Download completed and saved to $FILE"
  else
    echo "$FILE already exists."
  fi
}

json2args() {
  # transforms the JSON string to command line args, and '_' is replaced to '-'
  # example:
  # input: { "model": "--meta-llama/Llama-2-7b-chat-hf", "--tensor_parallel_size": 1, "--trust_remote_code": "" }
  # output: --model=meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size=1 --trust-remote-code
  local json_string=$1
  local args
  args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map(
        if .value == "" then
          (.key | gsub("_"; "-"))
        else
          (.key | gsub("_"; "-")) + "=" + (.value | tostring)
        end
      ) |
      join(" ")
    '
  )
  echo "$args"
}

json2env() {
  # transforms the JSON string to command line env,
  # example:
  # input: { "VLLM_USE_V1": "1", "run.sh": "" }
  # output: export VLLM_USE_V1=1 && run.sh
  local json_string=$1
  local args
  args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map(
        if .value == "" then
          .key
        else
          "export "+  .key + "=" + (.value | tostring)
        end
      ) |
      join(" && ")
    '
  )
  echo "$args"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 300 bash -c '
    until curl -s -X POST localhost:8006/v1/completions >/dev/null 2>&1; do
      sleep 1
    done' && return 0 || return 1
}

get_cur_npu_id() {
    npu-smi info -l | awk -F ':' '/NPU ID/ {print $2+0; exit}'
}

kill_npu_processes() {
  ps -aux
  lsof -t -i:8006 | xargs -r kill -9
  pgrep python | xargs -r kill -9
  
  sleep 4
  rm -rf ~/.config/vllm

}

update_json_field() {
  local json_file="$1"
  local field_name="$2"
  local field_value="$3"

  jq --arg value "$field_value" \
     --arg key "$field_name" \
     '.[$key] = $value' "$json_file" > "${json_file}.tmp" && \
     mv "${json_file}.tmp" "$json_file"
}

run_latency_tests() {
  # run latency tests using `benchmark_latency.py`
  # $1: a json file specifying latency test cases

  local latency_test_file
  latency_test_file=$1

  # Iterate over latency tests
  jq -c '.[]' "$latency_test_file" | while read -r params; do
    # get the test name, and append the NPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^latency_ ]]; then
      echo "In latency-test.json, test_name must start with \"latency_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get arguments
    latency_params=$(echo "$params" | jq -r '.parameters')
    latency_args=$(json2args "$latency_params")

    latency_command="python3 vllm_benchmarks/benchmark_latency.py \
      --output-json $RESULTS_FOLDER/${test_name}.json \
      $latency_args"

    echo "Running test case $test_name"
    echo "Latency command: $latency_command"

    # run the benchmark
    eval "$latency_command"
    # echo model_name to result file
    model_name=$(echo "$latency_params" | jq -r '.model')
    update_json_field "$RESULTS_FOLDER/${test_name}.json" "model_name" "$model_name"
    kill_npu_processes

  done
}

run_throughput_tests() {
  # run throughput tests using `benchmark_throughput.py`
  # $1: a json file specifying throughput test cases

  local throughput_test_file
  throughput_test_file=$1

  # Iterate over throughput tests
  jq -c '.[]' "$throughput_test_file" | while read -r params; do
    # get the test name, and append the NPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^throughput_ ]]; then
      echo "In throughput-test.json, test_name must start with \"throughput_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get arguments
    throughput_params=$(echo "$params" | jq -r '.parameters')
    throughput_args=$(json2args "$throughput_params")

    throughput_command="python3 vllm_benchmarks/benchmark_throughput.py \
      --output-json $RESULTS_FOLDER/${test_name}.json \
      $throughput_args"

    echo "Running test case $test_name"
    echo "Throughput command: $throughput_command"

    # run the benchmark
    eval "$throughput_command"
    # echo model_name to result file
    model_name=$(echo "$throughput_params" | jq -r '.model')
    update_json_field "$RESULTS_FOLDER/${test_name}.json" "model_name" "$model_name"
    kill_npu_processes

  done
}

run_serving_tests() {
  # run serving tests using `benchmark_serving.py`
  # $1: a json file specifying serving test cases

  local serving_test_file
  serving_test_file=$1

  # Iterate over serving tests
  jq -c '.[]' "$serving_test_file" | while read -r params; do
    # get the test name, and append the NPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^serving_ ]]; then
      echo "In serving-test.json, test_name must start with \"serving_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get client and server arguments
    server_params=$(echo "$params" | jq -r '.server_parameters')
    client_params=$(echo "$params" | jq -r '.client_parameters')
    env_params=$(echo "$params" | jq -r '.envs')
    server_args=$(json2args "$server_params")
    client_args=$(json2args "$client_params")
    set_env_command=$(json2env "$env_params")
    # check if server model and client model is aligned
    server_model=$(echo "$server_params" | jq -r '.model')
    client_model=$(echo "$client_params" | jq -r '.model')
    if [[ $server_model != "$client_model" ]]; then
      echo "Server model and client model must be the same. Skip testcase $test_name."
      continue
    fi

    server_command=" $set_env_command && \
      python \
      -m vllm.entrypoints.openai.api_server \
      $server_args"
    
    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"
    bash -c "$server_command" &
    server_pid=$!

    # wait until the server is alive
    if wait_for_server; then
      echo ""
      echo "vllm server is up and running."
    else
      echo ""
      echo "vllm failed to start within the timeout period."
    fi
    current_time=$(date +%m%d%H%M%S)
    new_test_name=$test_name"_"$current_time
    client_command="python vllm_benchmarks/benchmark_serving.py  \
    --save-result \
    --result-dir $RESULTS_FOLDER \
    --result-filename ${new_test_name}.json \
    $client_args"

    echo "Running test case $new_test_name"
    echo "Client command: $client_command"
    bash -c "$client_command"

    # clean up
    kill -9 $server_pid
    kill_npu_processes
  done
}

cleanup() {
  rm -rf ./vllm_benchmarks
}

cleanup_on_error() {
  echo "An error occurred. Cleaning up results folder..."
  rm -rf $RESULTS_FOLDER
}

get_benchmarks_scripts() {
  cp -r $BENCHMARK_SERVING_PATH vllm_benchmarks
}

main() {

  START_TIME=$(date +%s)
  check_npus

  # dependencies
  (which wget && which curl) || (yum install -y wget curl)
  (which jq) || (yum -y install jq)
  (which lsof) || (yum install -y lsof)
  (which hostname) || (yum install -y hostname)

  # get the current IP address, required by benchmark_serving.py
  # shellcheck disable=SC2155
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  # turn of the reporting of the status of each request, to clean up the terminal output
  export VLLM_LOG_LEVEL="WARNING"

  SCRIPT_DIR=$(dirname "$0")
  SCRIPT_DIR=$(cd "$SCRIPT_DIR" && pwd) 
  QUICK_BENCHMARK_ROOT=$(dirname "$SCRIPT_DIR")
  echo "QUICK_BENCHMARK_ROOT: $QUICK_BENCHMARK_ROOT"

  # prepare for benchmarking
  mkdir -p benchmarks && cd benchmarks
  get_benchmarks_scripts
  # trap cleanup EXIT

  declare -g RESULTS_FOLDER=results
  mkdir -p $RESULTS_FOLDER

  # benchmarks
  run_serving_tests $QUICK_BENCHMARK_ROOT/tests/serving-tests.json
  # run_latency_tests $QUICK_BENCHMARK_ROOT/tests/latency-tests.json
  # run_throughput_tests $QUICK_BENCHMARK_ROOT/tests/throughput-tests.json

  END_TIME=$(date +%s)
  ELAPSED_TIME=$((END_TIME - START_TIME))
  echo "Total execution time: $ELAPSED_TIME seconds"

}

main "$@"
