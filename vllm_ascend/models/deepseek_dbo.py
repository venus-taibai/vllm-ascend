# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# # Adapted from
# # vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py
# # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# # vllm-project/vllm/vllm/model_executor/models/deepseek_v2.py
# """Inference-only DeepseekV2/DeepseekV3 model."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.deepseek_v2 import \
    DeepseekV2ForCausalLM  # noqa: E501
from vllm.model_executor.models.deepseek_v2 import \
    yarn_get_mscale  # noqa: E501
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2Attention,
                                                    DeepseekV2DecoderLayer,
                                                    DeepseekV2MLAAttention)
from vllm.model_executor.models.utils import (
    PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group
from vllm_ascend.models.deepseek_v2 import CustomDeepseekV2MLP
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.context import (
    advance_step_multistream_layer_context, get_multistream_comm_context,
    get_multistream_layer_context, set_multistream_context)
from vllm_ascend.multistream.layers import (MultiStreamPostTransformerLayer,
                                            MultiStreamPreTransformerLayer)
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamStepMetadata,
                                              make_multistream_metadata_ds)
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.utils import (FusedMoEState, dispose_tensor,
                               get_fused_moe_state, npu_prefetch)

VLLM_ASCEND_ENABLE_DBO: bool = envs_ascend.VLLM_ASCEND_ENABLE_DBO


class CustomDeepseekDBOMLP(CustomDeepseekV2MLP):

    def _forward_ms_mlp(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        current_ms_metadata = get_multistream_comm_context()
        if current_ms_metadata is None:
            x, _ = self.down_proj(x)
            return x
        current_ms_metadata.before_comm_event.record(
            current_ms_metadata.comp_stream)
        with torch.npu.stream(current_ms_metadata.comm_stream):
            current_ms_metadata.before_comm_event.wait(
                current_ms_metadata.comm_stream)
            x, _ = self.down_proj(x)
            current_ms_metadata.after_comm_event.record(
                current_ms_metadata.comm_stream)
        return x


class CustomDeepseekDBOMoE(nn.Module):

    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = AscendFusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias)

        if config.n_shared_experts is not None:
            self.all_reduce_merge = envs_ascend.VLLM_ASCEND_SHARED_ROUTER_ALL_REDUCE_MERGE
            reduce_results = not self.all_reduce_merge
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=reduce_results,
                force_replicate=self.enable_multistream_moe,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None  # type: ignore
        CustomDeepseekDBOMoE.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size
        self.dp_rank = get_dp_group().rank_in_group

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()
        self.rm_router_logits = envs_ascend.VLLM_ASCEND_RM_ROUTER_LOGITS

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        # TODO: need a better flag to indicate whether in profile run or not.
        if attn_metadata is None:
            # for profile run
            is_prefill = True
            enable_force_load_balance = False
        else:
            is_prefill = attn_metadata.num_prefills > 0
            enable_force_load_balance = False
            if hasattr(attn_metadata, 'with_prefill_across_dp'):
                is_prefill = is_prefill or attn_metadata.with_prefill_across_dp

        # router_logits: (num_tokens, n_experts)
        router_logits = None
        if not self.rm_router_logits:
            router_logits, _ = self.gate(hidden_states)

        experts_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=CustomDeepseekDBOMoE.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=self.shared_experts,
            gate=self.gate)

        hidden_states = (
            experts_hidden_states[0] * self.routed_scaling_factor +
            experts_hidden_states[1])
        if self.all_reduce_merge:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states

    # ----------------------------------------- TBO-related --------------------------------------------
    def _forward_ms_op_shared_expert(
        self,
        hidden_states: torch.Tensor,
    ):
        shared_output = self.shared_experts._forward_ms_mlp(hidden_states)
        return shared_output

    def _forward_ms_op_gate(
        self,
        hidden_states: torch.Tensor,
    ):
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        return router_logits

    def _forward_ms_op_pre_moe_comm(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        attn_metadata: torch.Tensor,
        num_tokens: int,
        is_deepseek_v3_r1: bool,
        cu_dbo_tokens_across_dp_cpu: Optional[torch.Tensor],
        is_prefill: bool = False,
    ):

        chunk_hidden_states = None

        fused_moe_state = get_fused_moe_state(
            self.experts.moe_parallel_config.ep_size, is_prefill,
            is_deepseek_v3_r1)

        tp_size = get_tensor_model_parallel_world_size()
        if (tp_size > 1 and fused_moe_state != FusedMoEState.AllGather
                and fused_moe_state != FusedMoEState.AllGatherEP):
            if num_tokens < tp_size:
                hidden_states = nn.functional.pad(
                    hidden_states, (0, 0, 0, tp_size - num_tokens))
                router_logits = nn.functional.pad(
                    router_logits, (0, 0, 0, tp_size - num_tokens))
            chunk_hidden_states = torch.tensor_split(hidden_states,
                                                     tp_size,
                                                     dim=0)
            chunk_router_logits = torch.tensor_split(router_logits,
                                                     tp_size,
                                                     dim=0)
            tp_rank = get_tensor_model_parallel_rank()
            hidden_states = chunk_hidden_states[tp_rank]
            router_logits = chunk_router_logits[tp_rank]

        if self.experts.dp_size > 1 and (
                fused_moe_state == FusedMoEState.AllGather
                or fused_moe_state == FusedMoEState.AllGatherEP):
            if not self.experts.torchair_graph_enabled and not is_prefill:
                if attn_metadata is not None:
                    max_num_tokens_across_dp = attn_metadata.max_num_tokens_across_dp
                    if num_tokens < max_num_tokens_across_dp:
                        hidden_states = nn.functional.pad(
                            hidden_states,
                            (0, 0, 0, max_num_tokens_across_dp - num_tokens))
                        if not self.experts.rm_router_logits:
                            router_logits = nn.functional.pad(
                                router_logits,
                                (0, 0, 0,
                                 max_num_tokens_across_dp - num_tokens))

            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                current_ms_metadata.before_comm_event.record(
                    current_ms_metadata.comp_stream)
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    current_ms_metadata.before_comm_event.wait(
                        current_ms_metadata.comm_stream)
                    if is_prefill:
                        hidden_states = self.naive_multicast(
                            hidden_states, cu_dbo_tokens_across_dp_cpu)
                    else:
                        hidden_states = get_dp_group().all_gather(
                            hidden_states, 0)
                    if self.experts.rm_router_logits:
                        router_logits = self._forward_ms_op_gate(hidden_states)
                    else:
                        router_logits = get_dp_group().all_gather(
                            router_logits, 0)
                    current_ms_metadata.after_comm_event.record(
                        current_ms_metadata.comm_stream)
            else:
                if is_prefill:
                    hidden_states = self.naive_multicast(
                        hidden_states, cu_dbo_tokens_across_dp_cpu)
                else:
                    hidden_states = get_dp_group().all_gather(hidden_states, 0)
                if self.rm_router_logits:
                    router_logits = self._forward_ms_op_gate(hidden_states)
                else:
                    router_logits = get_dp_group().all_gather(router_logits, 0)

        return hidden_states, router_logits, chunk_hidden_states

    def _forward_ms_op_post_moe_comm(
        self,
        e_hidden_states: torch.Tensor,
        chunk_hidden_states: Optional[torch.Tensor],
        shared_hidden_states: Optional[torch.Tensor],
        num_tokens: int,
        is_deepseek_v3_r1: bool,
        cu_dbo_tokens_across_dp_cpu: Optional[torch.Tensor],
        is_prefill: bool = False,
    ):
        current_ms_metadata = get_multistream_comm_context()
        assert current_ms_metadata is not None

        tp_size = get_tensor_model_parallel_world_size()
        fused_moe_state = get_fused_moe_state(
            self.experts.moe_parallel_config.ep_size, is_prefill,
            is_deepseek_v3_r1)

        # TODO: support scenariso that not enable allreduce merge
        if (tp_size > 1 and fused_moe_state != FusedMoEState.AllGather
                and fused_moe_state != FusedMoEState.AllGatherEP):
            dist.all_gather(list(chunk_hidden_states), e_hidden_states,
                            self.experts.tp_group)
            final_hidden_states = torch.cat(chunk_hidden_states, dim=0)
            if self.experts.enable_multistream_moe:
                final_shared_states = torch.empty_like(final_hidden_states)
                dist.all_gather_into_tensor(final_shared_states,
                                            shared_hidden_states,
                                            self.experts.tp_group)
                shared_hidden_states = final_shared_states
            if num_tokens < tp_size:
                final_hidden_states = final_hidden_states[:num_tokens]
            dispose_tensor(e_hidden_states)
        elif self.dp_size > 1 and (fused_moe_state == FusedMoEState.AllGather
                                   or fused_moe_state
                                   == FusedMoEState.AllGatherEP):
            if is_prefill:
                start = 0 if self.experts.dp_rank == 0 else cu_dbo_tokens_across_dp_cpu[
                    self.dp_rank - 1]
                end = cu_dbo_tokens_across_dp_cpu[self.dp_rank]
                final_hidden_states = get_dp_group().all_reduce(
                    e_hidden_states)
                final_hidden_states = final_hidden_states[start:end, :]
                dispose_tensor(e_hidden_states)
            else:
                final_hidden_states_shape = (
                    e_hidden_states.size(0) //
                    self.dp_size, ) + e_hidden_states.shape[1:]
                final_hidden_states = torch.empty(
                    final_hidden_states_shape,
                    dtype=e_hidden_states.dtype,
                    device=e_hidden_states.device)
                dist.reduce_scatter_tensor(final_hidden_states,
                                           e_hidden_states,
                                           op=dist.ReduceOp.SUM,
                                           group=get_dp_group().device_group)
                final_hidden_states = final_hidden_states[:num_tokens]
                dispose_tensor(e_hidden_states)
        else:
            final_hidden_states = e_hidden_states

        if tp_size > 1 and not self.all_reduce_merge and (
                fused_moe_state == FusedMoEState.AllGather
                or fused_moe_state == FusedMoEState.AllGatherEP):
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        hidden_states = (final_hidden_states * self.routed_scaling_factor +
                         shared_hidden_states)

        if self.all_reduce_merge:
            current_ms_metadata.before_comm_event.record(
                current_ms_metadata.comp_stream)
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait(
                    current_ms_metadata.comm_stream)
                # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                current_ms_metadata.after_comm_event.record(
                    current_ms_metadata.comm_stream)

        return hidden_states


class CustomDeepseekDBOMLAAttention(DeepseekV2MLAAttention):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_mla = \
            ascend_config.torchair_graph_config.enable_multistream_mla

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        forward_kwargs = {}
        if self.q_lora_rank is not None:
            enable_multistream_mla = (self.enable_multistream_mla
                                      and self.torchair_graph_enabled
                                      and attn_metadata is not None and
                                      not attn_metadata.with_prefill_across_dp
                                      and attn_metadata.num_decodes > 0)
            npu_prefetch(self.q_a_proj.weight,
                         hidden_states,
                         enabled=enable_multistream_mla)
            ckq = self.q_a_proj(hidden_states)[0]
            hidden_states_or_q_c = self.q_a_layernorm(ckq)
            forward_kwargs["enable_multistream_mla"] = enable_multistream_mla
            forward_kwargs['ckq'] = ckq if enable_multistream_mla else None
        else:
            hidden_states_or_q_c = hidden_states
        if self.torchair_graph_enabled:
            if envs.VLLM_USE_V1:
                output_shape = hidden_states.shape
                output = torch.empty(output_shape,
                                     dtype=hidden_states_or_q_c.dtype,
                                     device=hidden_states_or_q_c.device)
                forward_kwargs['output'] = output

            output = self.mla_attn.impl.forward(self.mla_attn,
                                                hidden_states_or_q_c,
                                                hidden_states, None, kv_cache,
                                                attn_metadata,
                                                **forward_kwargs)
            if envs.VLLM_USE_V1:
                output = output.view(-1, output_shape[-1])
            return output
        else:
            kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
            return self.mla_attn(hidden_states_or_q_c,
                                 kv_c_normed,
                                 k_pe,
                                 output_shape=hidden_states.shape)


class CustomDeepseekDBODecoderLayer(DeepseekV2DecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        # TODO: enable mla in vllm-ascend
        if model_config.use_mla:
            attn_cls = CustomDeepseekDBOMLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_size = get_dp_group().world_size

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = CustomDeepseekDBOMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            previous_hidden_states, previous_residual = hidden_states, residual
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            # Dispose hidden_states and residual from the previous layer
            # to save npu memory because they're no longer used.
            dispose_tensor(previous_hidden_states)
            dispose_tensor(previous_residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        if isinstance(self.mlp, CustomDeepseekDBOMoE):
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        if isinstance(
                self.mlp,
                CustomDeepseekDBOMLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        return hidden_states, residual

    # ----------------------------------------- TBO-related --------------------------------------------
    def _forward_ms_layer(
        self,
        positions: List[torch.Tensor],
        hidden_states: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor],
        previous_residual: Optional[torch.Tensor],
        residual: List[torch.Tensor],
        attn_metadata: List[AttentionMetadata],
        cu_dbo_tokens_across_dp: List[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        layer_index, ms_metadata, _ = get_multistream_layer_context()
        assert layer_index >= 0 and ms_metadata is not None
        num_micro_batchs = ms_metadata.ms_config.num_micro_batches
        assert isinstance(self.mlp, CustomDeepseekDBOMoE)
        assert len(positions) == num_micro_batchs
        assert len(hidden_states) == num_micro_batchs
        assert residual is not None
        assert attn_metadata is not None
        router_logits = [None] * num_micro_batchs
        num_tokens = [None] * num_micro_batchs
        chunk_hidden_states = [None] * num_micro_batchs
        shared_hidden_states = [None] * num_micro_batchs

        # block 1 : attention
        # block 2 : attn tp communication
        # the attn computation of microbatch 1 can be overlapped with the moe
        # communication in the previous layer, and the attn computation of microbatch 2
        # can be overlapped with the attn communication of microbatch 1
        for i in range(num_micro_batchs):
            # wait last layer moe finishing communication
            ms_metadata.try_wait_event(layer_index - 1, i,
                                       MSEventKey.FFN_AR_FINISH,
                                       ms_metadata.calculate_stream)
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                comp_stream=ms_metadata.calculate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_AR_FINISH],
            )

            with set_multistream_context(context, i):
                forward_context = get_forward_context()
                forward_context.attn_metadata = attn_metadata[i]

                # input layernorm
                hidden_states[i], residual[
                    i] = self._forward_ms_op_input_layernorm(
                        hidden_states[i], residual[i])
                # dispose tensor from previous dense layers in the first moe layer
                if previous_hidden_states is not None:
                    dispose_tensor(previous_hidden_states)
                if previous_residual is not None:
                    dispose_tensor(previous_residual)
                # attention and tp allreduce
                hidden_states[i], residual[i] = self._forward_ms_op_attn(
                    positions[i], hidden_states[i], residual[i], kv_cache,
                    attn_metadata[i])

        # block 3 : shared experts + post layernorm + moe previous comm
        for i in range(num_micro_batchs):
            ms_metadata.try_wait_event(layer_index, i,
                                       MSEventKey.ATTN_AR_FINISH,
                                       ms_metadata.calculate_stream)
            hidden_states[i], residual[
                i] = self._forward_ms_op_post_attn_layernorm(
                    hidden_states[i], residual[i])

            # moe layer forward
            if attn_metadata[i] is None:
                is_prefill = True
                enable_force_load_balance = False
            else:
                is_prefill = attn_metadata[i].num_prefills > 0
                enable_force_load_balance = False
                if hasattr(attn_metadata[i], 'with_prefill_across_dp'):
                    is_prefill = is_prefill or attn_metadata[
                        i].with_prefill_across_dp

            router_logits[i] = None
            if not self.mlp.rm_router_logits:
                router_logits[i] = self.mlp._forward_ms_op_gate(
                    hidden_states[i])

            num_tokens[i], _ = hidden_states[i].shape
            is_deepseek_v3_r1 = self.mlp.experts.global_num_experts == 256

            # compute shared expert, currently it is not support enable_multistream_moe
            shared_hidden_states[i] = self.mlp._forward_ms_op_shared_expert(
                hidden_states[i])

            # moe pre comm
            hidden_states[i], router_logits[i], chunk_hidden_states[
                i] = self.mlp._forward_ms_op_pre_moe_comm(
                    hidden_states=hidden_states[i],
                    router_logits=router_logits[i],
                    attn_metadata=attn_metadata[i],
                    num_tokens=num_tokens[i],
                    is_deepseek_v3_r1=is_deepseek_v3_r1,
                    cu_dbo_tokens_across_dp_cpu=cu_dbo_tokens_across_dp[i]
                    if cu_dbo_tokens_across_dp is not None else None,
                    is_prefill=is_prefill)

            # block 5: moe post comp and comm
            if CustomDeepseekDBOMoE.top_k:
                real_top_k = CustomDeepseekDBOMoE.top_k
            else:
                real_top_k = self.mlp.experts.top_k

            e_hidden_states = self.mlp.experts._forward_ms_fused_moe_comp(
                hidden_states=hidden_states[i],
                router_logits=router_logits[i],
                is_prefill=is_prefill,
                real_top_k=real_top_k,
                enable_force_load_balance=enable_force_load_balance,
                shared_experts=self.mlp.shared_experts)

            if self.mlp.shared_experts:
                # e_hidden_states should not be tuple in dbo
                if isinstance(e_hidden_states, tuple):
                    e_hidden_states, shared_hidden_states = e_hidden_states

            # the following kernels will be submitted to the comm stream to overlap the computation of the
            # moe computation of next microbatch and the attn computation of next layer
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                comp_stream=ms_metadata.calculate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_AR_FINISH],
            )

            with set_multistream_context(context, i):
                hidden_states[i] = self.mlp._forward_ms_op_post_moe_comm(
                    e_hidden_states=e_hidden_states,
                    chunk_hidden_states=chunk_hidden_states[i],
                    shared_hidden_states=shared_hidden_states[i],
                    num_tokens=num_tokens[i],
                    cu_dbo_tokens_across_dp_cpu=cu_dbo_tokens_across_dp[i]
                    if cu_dbo_tokens_across_dp is not None else None,
                    is_deepseek_v3_r1=is_deepseek_v3_r1,
                    is_prefill=is_prefill)

        return hidden_states, residual

    # should split ops in Decoder Layer
    def _forward_ms_op_input_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            previous_hidden_states, previous_residual = hidden_states, residual
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            # Dispose hidden_states and residual from the previous layer
            # to save npu memory because they're no longer used.
            dispose_tensor(previous_hidden_states)
            dispose_tensor(previous_residual)
        return hidden_states, residual

    def _forward_ms_op_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor
        return hidden_states, residual

    def _forward_ms_op_post_attn_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual


class CustomDeepseekDBOModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomDeepseekDBODecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        # tbo related members
        if VLLM_ASCEND_ENABLE_DBO:
            self.dp_size = parallel_config.data_parallel_size
            self.dp_rank = parallel_config.data_parallel_rank
            self.use_mla = model_config.use_mla
            self.multistream_config = MultiStreamConfig()
            multistream_metadata = make_multistream_metadata_ds(
                start_layer=self.start_layer + self.first_k_dense_replace,
                end_layer=self.end_layer,
                causal_lm=getattr(config, "causal_lm", True),
                multistream_config=self.multistream_config,
            )
            self.ms_pre_layer = MultiStreamPreTransformerLayer(
                multistream_metadata)
            self.ms_post_layer = MultiStreamPostTransformerLayer(
                multistream_metadata)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        num_normal_layers = (self.first_k_dense_replace
                             if VLLM_ASCEND_ENABLE_DBO and self.can_run_ms()
                             else self.end_layer - self.start_layer)

        moe_start_layer = self.start_layer + num_normal_layers
        for i in range(self.start_layer, min(moe_start_layer, self.end_layer)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata)

        if moe_start_layer < self.end_layer:
            # if we enable multistream/dbo, process sparse layers here
            hidden_states, residual = self._forward_ms_layers(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                moe_start_layer=moe_start_layer,
                kv_caches=kv_caches,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def can_run_ms(self):
        attn_metadata = get_forward_context().attn_metadata
        # enable prefill overlap
        if attn_metadata is None or attn_metadata.num_prefills == 0 or not attn_metadata.enable_dbo_across_dp:
            return False
        return True

    def _forward_ms_layers(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        moe_start_layer: int,
        kv_caches: Optional[List[torch.Tensor]] = None,
        is_prefill: bool = False,
    ):

        if moe_start_layer == self.end_layer:
            return hidden_states, residual

        previous_hidden_states, previous_residual = hidden_states, residual
        attn_metadata, [positions, hidden_states,
                        residual] = self.ms_pre_layer(
                            [positions, hidden_states, residual], )

        if self.dp_size > 1:
            cu_dbo_tokens_across_dp = []
            # update the cu_dbo_tokens_across_dp
            for i in range(self.multistream_config.num_micro_batches):
                batchsize = attn_metadata[i].num_input_tokens
                num_dbo_tokens_across_dp = self._num_dbo_tokens_across_dp(
                    batchsize, self.dp_size, self.dp_rank)
                cu_dbo_token = torch.cumsum(num_dbo_tokens_across_dp, dim=0)
                cu_dbo_tokens_across_dp.append(cu_dbo_token)
        else:
            cu_dbo_tokens_across_dp = None

        # the rest layers
        for i in range(moe_start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer._forward_ms_layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                previous_hidden_states=previous_hidden_states,
                previous_residual=previous_residual,
                attn_metadata=attn_metadata,
                cu_dbo_tokens_across_dp=cu_dbo_tokens_across_dp,
                kv_cache=kv_caches[i - self.start_layer]
                if kv_caches is not None else None,
                is_prefill=is_prefill)
            previous_hidden_states = previous_residual = None
            advance_step_multistream_layer_context()

        [hidden_states,
         residual] = self.ms_post_layer([hidden_states, residual], )
        return hidden_states, residual

    def _num_dbo_tokens_across_dp(self, num_tokens: int, dp_size: int,
                                  dp_rank: int) -> torch.Tensor:
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor


class CustomDeepseekDBOForCausalLM(DeepseekV2ForCausalLM):
    # add `packed_modules_mapping` in `DeepseekV2ForCausalLM` to support weight merging
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomDeepseekDBOModel(vllm_config=vllm_config,
                                            prefix=maybe_prefix(
                                                prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states
