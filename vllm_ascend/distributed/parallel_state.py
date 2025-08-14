from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# vllm-ascend will maintain its own EP GroupCoordinator and ETP GroupCoordinator for
# customize parallel solution
_EP: Optional[GroupCoordinator] = None
_ETP: Optional[GroupCoordinator] = None
_OTP: Optional[GroupCoordinator] = None
_MC2: Optional[GroupCoordinator] = None

def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2

def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, ("expert model parallel group is not initialized")
    return _EP

def get_etp_group() -> GroupCoordinator:
    assert _ETP is not None, (
        "expert tensor parallel group is not initialized")
    return _ETP

def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, (
        "output tensor parallel group is not initialized")
    return _OTP

def model_parallel_initialized():
    return (_ETP is not None and _EP is not None)

def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    oproj_tensor_parallel_size: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
):
    global _MC2
    if _MC2 is None:
        all_ranks = torch.arange(world_size).reshape(-1, expert_parallel_size)
        group_ranks = all_ranks.unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _MC2 = init_model_parallel_group(group_ranks,
                        get_world_group().local_rank,
                        backend,
                        group_name="mc2")
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = world_size or torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = expert_tensor_parallel_size
    num_expert_tensor_parallel_groups = expert_parallel_size

    global _EP
    group_ranks = []
    for i in range(num_expert_parallel_groups):
        ranks = list(range(i, world_size, num_expert_parallel_groups))
        group_ranks.append(ranks)

    _EP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="ep")

    group_ranks = []
    global _ETP
    for i in range(num_expert_tensor_parallel_groups):
        ranks = list(
            range(i * expert_tensor_parallel_size,
                  (i + 1) * expert_tensor_parallel_size))
        group_ranks.append(ranks)

    _ETP = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="etp")
    
    if oproj_tensor_parallel_size is not None:
        group_ranks = []
        global _OTP
        num_oproj_tensor_parallel_groups: int = (world_size //
                                                oproj_tensor_parallel_size)
        for i in range(num_oproj_tensor_parallel_groups):
            ranks = list(
                range(i * oproj_tensor_parallel_size,
                    (i + 1) * oproj_tensor_parallel_size))
            group_ranks.append(ranks)
        _OTP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="otp")


def destory_ascend_model_parallel():
    global _EP
    if _EP:
        _EP.destroy()
    _EP = None

    global _ETP
    if _ETP:
        _ETP.destroy()
    _ETP = None
    
    global _OTP
    if _OTP:
        _OTP.destroy()  
    _OTP = None