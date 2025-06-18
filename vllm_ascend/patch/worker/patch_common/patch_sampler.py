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
    if k is None or p is None:
        logits = apply_top_k_top_p_tpu(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)
    logits_new = logits.to(torch.float16)
    k_new = k.unsqueeze(1)
    p_new = p.to(torch.float16).unsqueeze(1)
    out1, out2 = torch_npu._npu_topk_topp_sampling(logits_new, k_new, p_new)
    return out1.view(-1)

TopKTopPSampler.forward_native = forward_npu