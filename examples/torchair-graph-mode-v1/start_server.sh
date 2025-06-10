export VLLM_ENABLE_MC2=0
export VLLM_USE_V1=1

export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_GRAPH_MODE=0
export USING_LCCL_COM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python -m vllm.entrypoints.openai.api_server --model=/mnt/deepseek/DeepSeek-R1-W8A8-VLLM \
 --trust-remote-code \
 --distributed-executor-backend=mp \
 -tp=16 \
 --port 8006 \
 --max-model-len 2048 \
 --max-num-batched-tokens 4096 \
 --block-size 128 \
 --compilation_config 0 \
 --disable-log-stats \
 --disable-log-requests \
 --gpu-memory-utilization 0.96 \
 --additional-config '{"ascend_scheduler_config":{},"enable_graph_mode":true}'
