export TASK_QUEUE_ENABLE=2
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PROMETHEUS_MULTIPROC_DIR=/tmp/

python -m vllm.entrypoints.openai.api_server --model=/mnt/deepseek/QWen/qwen3-32b/qwen3-32b \
 --served-model-name auto \
 --load-format=prefetch_auto \
 --trust-remote-code \
 --distributed-executor-backend=mp \
 --port 8010 \
 -tp=8 \
 --max-num-seqs 140 \
 --max-model-len 32768 \
 --max-num-batched-tokens 32768 \
 --enable-prefix-caching \
 --block-size 128 \
 --gpu-memory-utilization 0.97