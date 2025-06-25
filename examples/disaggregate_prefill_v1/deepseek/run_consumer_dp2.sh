# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=2
# export ASCEND_GLOBAL_EVENT_ENABLE=1
# export HCCL_ENTRY_LOG_ENABLE=1
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=xx.xx.xx.84
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
export DISAGGREGATED_RPEFILL_RANK_TABLE_PATH="/home/fjw/json/ranktable_84.json"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
#export VLLM_LLMDD_CHANNEL_PORT=6658
#export VLLM_LLMDD_CHANNEL_PORT=15272
export VLLM_LLMDD_CHANNEL_PORT=14012
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCAKE_CONFIG_PATH="/home/fjw/json/mooncake_barebone.json"
export VLLM_USE_V1=1
export VLLM_BASE_PORT=9800
export ENV_RANKTABLE_PATH="/home/fjw/json/hccl_8p_01234567_xx.xx.xx.84.json"

vllm serve "/home/weights/deepseekv3-lite-base-latest-w8a8-dynamic" \
  --host xx.xx.xx.84 \
  --port 8200 \
  --tensor-parallel-size 2\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000 \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address xx.xx.xx.84 \
  --data-parallel-rpc-port 8100 \
  --data-parallel-start-rank 0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1_barebone",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20002",
  "engine_id": 1,
  "kv_rank": 1,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector_v1_barebone",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 2
             }
      }
  }'  \
  --additional-config \
  '{}'\
