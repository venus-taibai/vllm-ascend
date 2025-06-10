import torch
import os
import torch_npu
import numpy as np
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import pandas

# 控制模式
quant_mode = 2                              # 0为dispatch非量化，1非dispatch静态量化，2为动态量化
is_dispatch_scales = True                   # 动态量化可选择是否传scales
input_dtype = torch.bfloat16                # 输出type
server_num = 1                              # 单机或者多机模式
server_index = 0                            # 多机每个脚本保持一致并递增index值
master_ip = '127.0.0.1'
dev_num = 16                                # 一个host跑几张die，默认跑满16 die
world_size = server_num * dev_num           # 总die数
rank_per_dev = int(world_size / server_num) # 每个host有几个die
sharedExpertRankNum = 0                     # 共享专家数
moeExpertNum = 16                           # moe专家数
bs = 8                                      # token数量
h = 7168                                    # 每个token的长度
k = 8
random_seed = 0                             # 随机数用于随机x和topk
tp_world_size = 1
ep_world_size = int(world_size / tp_world_size)
moe_rank_num = ep_world_size - sharedExpertRankNum
local_moe_expert_num = moeExpertNum // moe_rank_num
globalBS = bs * ep_world_size
is_shared = (sharedExpertRankNum > 0)
is_quant = (quant_mode > 0)

def generate_unique_topK_tensor(shape, low, high):
    return (torch.randperm(high - low) + low)[:shape]

def gen_x(shape, dtype, tp_world_size):
    x_list = []
    for tp_id in range(tp_world_size):
        cur_x = torch.empty(shape, dtype=dtype).uniform_(-5, 5)
        x_list.append(cur_x)
    return x_list

def init_hccl_comm(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print("current device:", torch_npu.npu.current_device())
    print('[INFO] device_{} 创建HCCL通信链路 '.format(rank))
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method='tcp://' + master_ip + ':50001')
    print(f"device_{rank} init_process_group success")
    print("device %d 初始化EP域" % rank, flush=True)
    # 初始化EP域
    ep_ranks_list = []
    tp_ranks_list = []
    for i in range(tp_world_size):
        single_ep_rank_list = list(range(i, world_size, tp_world_size))
        ep_ranks_list.append(single_ep_rank_list)
    for i in range(ep_world_size):
        single_tp_rank_list = list(range(i * tp_world_size, (i + 1) * tp_world_size))
        tp_ranks_list.append(single_tp_rank_list)

    for i in range(tp_world_size):
        # ep_ranks_list = [[0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15]]，如果指定ep list,按照指定ep list划分ep域
        ep_ranks = ep_ranks_list[i]
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_tmp = ep_group
    print("device %d 初始化TP域" % rank, flush=True)
    # 初始化TP域
    for i in range(ep_world_size):
        # tp_ranks_list = [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]]，如果指定tp list,按照指定tp list划分tp域
        tp_ranks = tp_ranks_list[i]
        tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_tmp = tp_group
    ep_hcomm_info = ep_group_tmp._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm_info = tp_group_tmp._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return dist, ep_hcomm_info, tp_hcomm_info, ep_group_tmp, tp_group_tmp


def run_npu_process(queue, rank, x, expert_idxs, scales, expert_scales, golden_tokens):
    dist, ep_hcomm_info, tp_hcomm_info, ep_group, tp_group = init_hccl_comm(rank)
    print('[INFO] device_{} 构造output_npu数据'.format(rank))
    input_scales = scales.npu() if is_quant and is_dispatch_scales else None
    x = x.npu().to(input_dtype)
    expert_scales = expert_scales.to(torch.float32).npu()
    expert_idxs = expert_idxs.npu().to(torch.int32)
    print(f'[INFO] device_{rank} group_tp={tp_hcomm_info}')
    expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch(
                                                                            x=x,
                                                                            expert_ids=expert_idxs,
                                                                            group_ep=ep_hcomm_info,
                                                                            ep_world_size=ep_world_size,
                                                                            ep_rank_id=rank // tp_world_size,
                                                                            expert_shard_type=0,
                                                                            shared_expert_rank_num=sharedExpertRankNum,
                                                                            moe_expert_num=moeExpertNum,
                                                                            scales=input_scales,
                                                                            quant_mode=quant_mode,
                                                                            global_bs=globalBS)
    print(f'[INFO] device_{rank} expand_x={expand_x}')
    if is_quant:
        expand_x = expand_x.to(input_dtype)
    x = torch_npu.npu_moe_distribute_combine(expand_x=expand_x,
                                                    expert_ids=expert_idxs,
                                                    expand_idx=expand_idx,
                                                    ep_send_counts=ep_recv_counts,
                                                    expert_scales=expert_scales,
                                                    group_ep=ep_hcomm_info,
                                                    ep_world_size=ep_world_size,
                                                    ep_rank_id=rank//tp_world_size,
                                                    expert_shard_type=0,
                                                    shared_expert_rank_num=sharedExpertRankNum,
                                                    moe_expert_num=moeExpertNum,
                                                    global_bs=globalBS)
    print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
    queue.put((rank, [torch.tensor([rank]).cpu(), x.cpu()]))

def gen_npu(x_list, expert_idxs_list, scales_list, scalesInput_list):
    p_list = []
    port = 29500 + np.random.randint(0, 10000)
    rank_list = list(range(int(world_size)))
    print("rank list is: ", rank_list)

    from torch.multiprocessing import Manager
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)
    for rank in rank_list:
        ep_id = int(rank // tp_world_size)
        tp_id = rank % tp_world_size
        scales = scales_list[tp_id].cpu() if is_quant and is_dispatch_scales else None
        expert_scales = None
        expert_scales = scalesInput_list[tp_id][ep_id].cpu()
        p = Process(target=run_npu_process, args=(result_queue, rank, x_list[tp_id][ep_id].cpu(), 
                                expert_idxs_list[tp_id][ep_id].cpu(), scales, expert_scales, None))
        p.start()
        p_list.append(p)
    results = {}
    for p in p_list:
        p.join()
        id, result = result_queue.get()
        results[id] = result
    result_lists = list(results.values())
    indexed_data = list(enumerate(result_lists))
    sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1][0].item())
    sorted_data = [result_lists[index] for index, sublist in sorted_indexed_data]
    concat_results = []
    # 遍历每个位置的张量
    for i in range(len(sorted_data[0])):   
        # 提取每个列表中相同位置的张量
        tensors_at_position = [tensor_list[i] for tensor_list in sorted_data]
        # 使用torch.cat将这些张量连接在一起
        concatenated_tensor = torch.cat(tensors_at_position, dim=0)
        # 将合并后的张量添加到结果列表中
        concat_results.append(concatenated_tensor)
    return concat_results

def chunk_tensor(tensor, num_chunks):
    return list(tensor.chunk(num_chunks))

if __name__ == "__main__":
    print(f"bs={bs}")
    print(f"global_bs={globalBS}")
    print(f"shared_expert_rank_num={sharedExpertRankNum}")
    print(f"moe_expert_num={moeExpertNum}")
    print(f"k={k}")
    print(f"quant_mode={quant_mode}", flush=True)
    print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
    print(f"tp_world_size={tp_world_size}", flush=True)
    print(f"ep_world_size={ep_world_size}", flush=True)

    if tp_world_size != 1 and local_moe_expert_num > 1:
        print("unSupported tp = 2 and local moe > 1")
        exit(0)

    if sharedExpertRankNum > ep_world_size:
        print("sharedExpertRankNum 不能大于 ep_world_size")
        exit(0)

    if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
        print("ep_world_size必须是sharedExpertRankNum的整数倍")
        exit(0)

    if moeExpertNum % moe_rank_num != 0:
        print("moeExpertNum必须是moe_rank_num的整数倍")
        exit(0)

    xShape = [globalBS, h]
    topkShape = [globalBS, k]
    # 设置随机种子
    torch.manual_seed(random_seed)
    # 生成左右入参矩阵
    x_list = gen_x(xShape, input_dtype, tp_world_size)
    x_list = [x.view(-1, h) for x in x_list]
    # 生成topk部分
    topk_ids = []
    for _ in range(tp_world_size):
        for i in range(globalBS):
            topk_ids.append(generate_unique_topK_tensor(k, 0, moeExpertNum))
    topk_ids = torch.cat(topk_ids, dim=0)

    topk_ids_list = chunk_tensor(topk_ids, tp_world_size)
    expert_idx_list = [topk_ids.int().reshape(topkShape).contiguous() for topk_ids in topk_ids_list]
    scalesInput_list = []
    expert_scales_list = [torch.empty(size=[globalBS, k], dtype=torch.float32).uniform_(-1, 1) for _ in range(tp_world_size)]
    for tp_id in range(tp_world_size):
        expert_idx = expert_idx_list[tp_id]
        scalesInput = (torch.empty(1+moeExpertNum, h) if is_shared else torch.empty(moeExpertNum, h)) if is_dispatch_scales else None
        scales = torch.empty(size=[sharedExpertRankNum+moeExpertNum, h], dtype=torch.float32).uniform_(-1, 1)
        if is_shared:
            for i in range(1, sharedExpertRankNum):
                scales[i, :] = scales[0, :]
            scalesInput[0,:] = scales[0, :]
            scalesInput[1:1+moeExpertNum,:] = scales[sharedExpertRankNum:sharedExpertRankNum+moeExpertNum, :]
        else:
            scalesInput[:moeExpertNum,:] = scales[sharedExpertRankNum:sharedExpertRankNum+moeExpertNum,:]
        scalesInput_list.append(scalesInput)
    _, npu_expand_x = gen_npu(
        [chunk_tensor(x, ep_world_size) for x in x_list],
        [chunk_tensor(expert_idx, ep_world_size) for expert_idx in expert_idx_list], 
        scalesInput_list,
        [chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list])
    print("run npu success.")

