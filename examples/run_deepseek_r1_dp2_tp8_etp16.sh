export VLLM_ENABLE_MC2=0
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_LAUNCH_BLOCKING=0
export VLLM_VERSION=0.9.0

nohup python -m vllm.entrypoints.openai.api_server --model=/mnt/deepseek/DeepSeek-R1-W8A8-VLLM \
    --quantization ascend \
    --served-model-name auto \
    --trust-remote-code \
    --distributed-executor-backend=mp \
    --port 8006 \
    -tp=8 \
    -dp=2 \
    --max-num-seqs 24 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --block-size 128 \
    -O 0 \
    --no-enable-prefix-caching \
    --additional-config '{"torchair_graph_batch_sizes":[24],"expert_tensor_parallel_size":16,"enable_inter_dp_scheduling":true,"init_torchair_graph_batch_sizes":true,"trace_recompiles":true,"ascend_scheduler_config":{},"enable_graph_mode":true}' \
    --gpu-memory-utilization 0.95 &> run.log &
disown