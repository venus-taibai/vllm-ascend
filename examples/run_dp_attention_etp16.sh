rm -rf .torchair_cache
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_ENABLE_MC2=0
export HCCL_BUFFSIZE=200

export VLLM_USE_V1=1

export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_GRAPH_MODE=0
export USING_LCCL_COM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_OP_BASE_FFTS_MODE_ENABLE=true
export HCCL_IF_BASE_PORT=6000
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1

export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/ext:/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu

nohup python -m vllm.entrypoints.openai.api_server --model=/mnt/deepseek/DeepSeek-R1-W8A8-VLLM \
 --trust-remote-code \
 --distributed-executor-backend=mp \
 -tp=8 \
 -dp=2 \
 --port 8006 \
 --max-num-seqs 32 \
 --max-model-len 32768 \
 --max-num-batched-tokens 32768 \
 --block-size 128 \
 --enable-expert-parallel \
 --compilation_config 0 \
 --gpu-memory-utilization 0.96 \
 --additional-config '{"expert_tensor_parallel_size":16, "enable_inter_dp_scheduling":true,"trace_recompiles":true,"ascend_scheduler_config":{},"enable_graph_mode":true}' &> run.log &
