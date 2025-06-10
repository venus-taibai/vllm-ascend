#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/basic.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_MC2"] = "1"
os.environ["HCCL_BUFFSIZE"] = "480"

if __name__ == "__main__":
    prompts = [
        #"Who are you?"
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    # Create an LLM.
    llm = LLM(
        #model="/data/weights/deepseek-ai/deepseekv3-lite-base-latest",
        tensor_parallel_size=16,
        model="/mnt/deepseek/DeepSeek-R1-W8A8-VLLM",
        #model="/mnt/deepseek/DeepSeek-V2-Lite",
        enforce_eager=False,
        trust_remote_code=True,
        max_num_seqs=48,
        enable_expert_parallel=True,
        additional_config={
            'enable_graph_mode': True,
            'ascend_scheduler_config': {},
        },
        max_model_len=1024)

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
