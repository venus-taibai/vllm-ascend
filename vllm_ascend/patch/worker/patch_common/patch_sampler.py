from typing import Optional
import torch
import torch_npu
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p_tpu, random_sample


def forward_npu(
    self,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    if p is not None and k is not None:
        # npu_top_k_top_p's parameter order is (logits, p, k), not (logits, k, p)
        logits = torch_npu.npu_top_k_top_p(logits, p, k)
    else:
        logits = apply_top_k_top_p_tpu(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, generators)


TopKTopPSampler.forward_native = forward_npu
