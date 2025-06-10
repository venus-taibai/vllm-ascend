#!/bin/bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export MODEL_NAME="/mnt/deepseek/DeepSeek-R1-W8A8-VLLM"

export VLLM_ENABLE_MC2=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=2
export USING_LCCL_COM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_VERSION=0.8.5.post1
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=1200
export VLLM_LOGGING_LEVEL="DEBUG"

rm -rf .torchair_cache

vllm serve \
    $MODEL_NAME \
    --max-model-len 256 \
    --gpu-memory-utilization 0.80 \
    --trust_remote_code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --enable_expert_parallel \
    --compilation_config 0 \
    --additional-config '{
        "enable_graph_mode":true, 
        "enable_inter_dp_scheduling":true, 
        "ascend_scheduler_config":{}
    }'
