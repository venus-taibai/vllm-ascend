
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger("msit_logger")

def save_matrix_to_json(output_path, file_name, deployment):
    # 构建两层嵌套字典
    num_layers = deployment.shape[0]
    num_cards = deployment.shape[1]

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            # 将 1*4 的行矩阵转换为列表
            device = {"device_id": j, "device_expert": deployment[i, j].tolist()}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = f"{output_path}{file_name}.json"

    # 保存为 JSON 文件
    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"写入文件 {file_name} 时出错: {e}")

def compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert, is_only):
    route_expert_num = len(origin_weights)
    route_expert_redundancy = [[] for _ in range(route_expert_num)]
    if is_only == 1:
        sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
        weights = [origin_weights[idx] for idx in sorted_indices]
        for i in range(num_redundancy_expert):
            route_expert_redundancy[weights[i][0]].append(route_expert_num + i)
            avg_weight = weights[i][1] / (len(route_expert_redundancy[weights[0][0]]) + 1)
            weights[i] = (weights[i][0], avg_weight)
    else:
        for i in range(num_redundancy_expert):
            sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
            weights = [origin_weights[idx] for idx in sorted_indices]
            tmp_raw_weight = weights[0][1] * (len(route_expert_redundancy[weights[0][0]]) + 1)
            route_expert_redundancy[weights[0][0]].append(route_expert_num + i)
            avg_weight = tmp_raw_weight / (len(route_expert_redundancy[weights[0][0]]) + 1)
            weights[0] = (weights[0][0], avg_weight)
            origin_weights = weights

    expert_num = route_expert_num + num_redundancy_expert
    items_per_box = expert_num // card_num
    remaining_items = expert_num % card_num

    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num
    box_counts = [0] * card_num
    index = 0
    for i in range(route_expert_num):
        redundancy_num = len(route_expert_redundancy[i])
        for _ in range(redundancy_num):
            cur_weight = 0
            for item, weight in origin_weights:
                if item == i:
                    cur_weight = weight
            if index >= card_num:
                logger.error("Index Out of Bounds")
                break
            boxes[index].append(i)
            boxes_weights[index].append(cur_weight)
            box_weights[index] += cur_weight
            box_counts[index] += 1
            index += 1

    sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
    origin_weights = [origin_weights[idx] for idx in sorted_indices]
    for item_id, weight in origin_weights:
        min_box_index = -1
        for i in range(card_num):
            if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                    min_box_index = i

        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    result = []
    for i in range(card_num):
        result.append({
            "box_index": i + 1,
            "items": boxes[i],
            "weight": boxes_weights[i],
            "total_weight": box_weights[i],
            "item_count": box_counts[i]
        })

    return result, boxes


# 冗余专家部署
def lb_and_intra_layer_affinity_redundancy_deploy(
        layer_workloads,  
        num_redundancy_expert, 
        num_npus=64, 
        num_original_expert=256,):
    """
    :param layer_workloads[layer_num, expert_num] 58*256
    :return: optimized layer_deployment: [layer_num, card_num, card_expert_num] 58*64*4
    """
    layer_num = layer_workloads.shape[0]
    expert_num = layer_workloads.shape[1]
    if num_original_expert != expert_num:
        raise ValueError(f"原始专家数量 {num_original_expert} 必须等于 expert_num {expert_num}")

    if num_npus <= 0:
        raise ValueError("NPUs 数量必须大于 0")

    if num_npus < num_redundancy_expert:
        raise ValueError(f"NPUs 数量 {num_npus} 必须大于或等于冗余专家数量 {num_redundancy_expert}")

    global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]

    original_weights = []
    max_weights = []
    average_weights = []
    y_list = []
    for layer in range(layer_num):
        weights = np.zeros((expert_num,), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads[layer]):
            weights[expert_id] = (expert_id, workload_weight)

        result, layer_deployment = compute_balanced_pack_redundancy(weights, num_npus, num_redundancy_expert, 0)

        max_weight = 0
        for box in result:
            if max_weight < box['total_weight']:
                max_weight = box['total_weight']
            print(layer,
                f"before: Box {box['box_index']}: "
                f"Items = {box['items']}, weight = {box['weight']}, "
                f"Total Weight = {box['total_weight']}, Item Count = {box['item_count']}"
            )
        new_value = layer_workloads[layer].reshape(num_npus, -1)
        row_sum = np.sum(new_value, axis=1)
        original_weights.append(row_sum.max())
        max_weights.append(max_weight)
        average_weights.append((np.sum(layer_workloads[layer]) / num_npus))

        global_deployment[layer] = layer_deployment

    y_list.append(original_weights)
    y_list.append(max_weights)
    y_list.append(average_weights)

    return global_deployment, y_list

def calculate_average(lst):
    """计算一维列表的平均值"""
    if not lst:
        raise ValueError("列表不能为空")

    total = 0
    count = 0

    for element in lst:
        # 检查元素是否为数值类型
        if isinstance(element, (int, float, np.int64, np.float64)):
            total += element
            count += 1
        else:
            # 非数值类型元素会被忽略，并打印警告
            print(f"警告: 元素 {element} 不是数值类型，已被忽略")

    if count == 0:
        raise ValueError("列表中不包含任何数值类型的元素")

    return total / count

def layer_imblance_polt(y_list, label_names, device_num, output_path, file_name):

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    x = [i for i in range(58)]
    for index, y in enumerate(y_list):
        plt.plot(x, y, label=rf'{label_names[index]}，avg={calculate_average(y)}')

    # 显示图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title(rf'Load Change (Number of Cards={device_num})')
    plt.xlabel('layer')
    plt.ylabel('Card Load')

    # 显示网格线
    plt.grid(True)

    plt.savefig(os.path.join(output_path, file_name), dpi=300)

    # 清理当前图表
    plt.close()

def deepseek_deploy(workload, num_redundancy_expert, num_groups, num_nodes, num_gpus, num_original_expert):
    from eplb_deepseek import rebalance_experts
    num_replicas = num_original_expert + num_redundancy_expert
    hy2log, log2phy, logcnt = rebalance_experts(workload, num_replicas, num_groups, num_nodes, num_gpus)

    # convert to global_deployment
    workload = workload.cpu().numpy()
    global_deployment = []
    layer_num = log2phy.shape[0]
    num_physical_experts_local = (num_original_expert + num_redundancy_expert) // num_gpus
    for layer_idx in range(layer_num):
        layer_deployment = []
        for gpu_idx in range(num_gpus):
            local_deployment = hy2log[layer_idx][gpu_idx*num_physical_experts_local:(gpu_idx+1)*num_physical_experts_local]
            local_deployment = local_deployment.flatten()
            layer_deployment.append(local_deployment.tolist())
        global_deployment.append(layer_deployment)

    # remapping expert distribution according to log2phy
    original_weights = []
    max_weights = []
    average_weights = []
    y_list = []
    for layer_idx in range(layer_num):
        new_value = workload[layer_idx].reshape(num_gpus, -1)
        row_sum = np.sum(new_value, axis=1)
        original_weights.append(row_sum.max())
        average_weights.append((np.sum(workload[layer_idx]) / num_gpus))
        
        opt_workload = np.zeros((num_original_expert + num_redundancy_expert), dtype=np.float64)
        for expert_idx in range(num_original_expert):
            physical_expert_idxs = log2phy[layer_idx][expert_idx]
            physical_expert_idxs = physical_expert_idxs.flatten()
            physical_expert_idxs = physical_expert_idxs[physical_expert_idxs != -1]
            for physical_expert_idx in physical_expert_idxs:
                opt_workload[physical_expert_idx] += workload[layer_idx][expert_idx] / len(physical_expert_idxs)
        opt_workload = opt_workload.reshape(num_gpus, -1)
        row_sum = np.sum(opt_workload, axis=1)
        max_weights.append(row_sum.max())

    y_list = [original_weights, max_weights, average_weights]
    return global_deployment, y_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="gsm8k_temp0.0")
    parser.add_argument("--num_original_expert", type=int, default=256)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--num_redundancy_expert", type=int, default=0)
    parser.add_argument("--num_devices", type=int, default=32)
    parser.add_argument("--num_groups", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="default", choices=["default", "deepseek"])
    args = parser.parse_args()
    exp_name = args.exp_name
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    num_redundancy_expert = args.num_redundancy_expert
    num_devices = args.num_devices
    num_original_expert = args.num_original_expert
    num_groups = args.num_groups
    num_nodes = args.num_nodes

    workload = torch.load(input_path, map_location=torch.device('cpu'))
    if args.strategy == "default":
        workload = workload.float().int().numpy()
        global_deployment, y_list = lb_and_intra_layer_affinity_redundancy_deploy(workload, num_redundancy_expert, num_devices, num_original_expert)
    elif args.strategy == "deepseek":
        global_deployment, y_list = deepseek_deploy(workload, num_redundancy_expert, num_groups, num_nodes, num_devices, num_original_expert)

    file_name = f"{exp_name}_{num_devices}_{num_redundancy_expert}"
    save_matrix_to_json(output_path, file_name, np.array(global_deployment))
    label_names = [f'Temperature of hottest card in sequential deployment',
                    f'Temperature of hottest card after load balancing',
                    f'Average card temperature']
    new_file_name = f"{exp_name}_{num_devices}_{num_redundancy_expert}.png"
    layer_imblance_polt(y_list, label_names, num_devices, output_path, new_file_name)