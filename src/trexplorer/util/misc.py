# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
import sys
import logging
import numpy as np
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_stats(input_dict):
    """
    Args:
        input_dict (dict): all the values will be gathered
    Gather the values in the dictionary from all processes. Returns a dict with the same fields as
    input_dict, after gathering.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        index = 0
        indices = [index]
        names = []
        values = []
        shapes = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            shapes.append(input_dict[k].shape)
            index += len(input_dict[k].flatten())
            indices.append(index)
            values.append(input_dict[k].flatten())
        values = torch.cat(values, dim=0)
        values_list = [torch.empty(values.shape).to('cuda') for _ in range(world_size)]
        dist.all_gather(values_list, values)
        gathered_dict = {}
        for i, name in enumerate(names):
            value = []
            for j in range(world_size):
                value.append(values_list[j][indices[i]: indices[i + 1]].reshape(shapes[i]))
            stacked = torch.stack(value, dim=1)
            interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)
            gathered_dict[name] = interleaved

    return gathered_dict


def gather_list(input_list):
    # Buggy function
    world_size = get_world_size()
    if world_size < 2:
        return input_list
    with torch.no_grad():
        values = torch.as_tensor(input_list).to('cuda')
        values_list = [torch.empty(values.shape).to('cuda') for _ in range(world_size)]
        dist.all_gather(values_list, values)

        stacked = torch.stack(values_list, dim=1)
        interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)

    return interleaved


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
        # slurm process but not interactive
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(seconds=3600))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sigmoid_focal_loss(inputs, targets, query_mask, weights):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    gamma = 2.0
    loss = ce_loss * ((1 - p_t) ** gamma)
    loss = weights * loss

    if query_mask is not None:
        loss = torch.stack([ls[m].mean(0)
                            if m.any() else torch.zeros(loss.shape[2]).to(loss.device)
                            for ls, m in zip(loss, query_mask)])
        loss = loss.mean(0)
        loss = loss.sum()
    else:
        loss = loss.mean(1).sum()

    return loss


def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace


def nested_dict_to_device(dictionary, device):
    output = {}
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            output[key] = nested_dict_to_device(value, device)
        return output
    return dictionary.to(device)


def restore_config(args):
    checkpoint = torch.load(args.resume, map_location='cpu')
    args_n = checkpoint['args']

    # add eval args
    if args.eval_only:
        args_n.resume = args.resume
        args_n.output_dir = args.output_dir
        args_n.dataset = args.dataset
        args_n.data_dir = args.data_dir
        args_n.eval_only = args.eval_only
        args_n.mask = args.mask

    # add possible missing args
    for k in args.__dict__:
        if k not in args_n.__dict__.keys():
            args_n.__dict__[k] = args.__dict__[k]

    return args_n


def get_lr_scheduler(args, optimizer, len_dataloader):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=int(args.epochs) * len_dataloader,  # Maximum number of iterations
                                                           eta_min=args.lr * args.min_lr_mltp)  # Minimum learning rate.

    return scheduler


def setup_logger(args):
    fileHandler = logging.FileHandler(args.log_file, mode='a')
    consoleHandler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[fileHandler, consoleHandler]
    )

    # print the arguments
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    return logging


def get_node(node_id, tree):
    root_node_info = tree.nodes[node_id]
    root_node_info['id'] = node_id
    return root_node_info


def generate_past_trajectory_pp_nx(tree, node_id, num_prev_pos, sub_vol_size):
    past_traj_label = []
    current_node = get_node(node_id, tree)
    parent = current_node
    root_position = np.asarray(current_node['position']).astype(int)

    for i in range(num_prev_pos):
        node_info_list = []
        node_position = np.asarray(parent['position'])
        rel_position = (node_position - root_position).astype(float)
        rel_position /= (sub_vol_size // 2)
        node_info_list.append(rel_position)
        node_radius = np.asarray(parent['radius']).reshape(1)
        node_radius /= (sub_vol_size // 2)
        node_info_list.append(node_radius)
        past_traj_label.append(np.concatenate(node_info_list))

        parent_id = list(tree.predecessors(parent['id']))
        if not len(parent_id):
            break
        parent = get_node(parent_id[0], tree)

    past_traj_label_np = np.asarray(past_traj_label)

    num_missing_points = num_prev_pos - past_traj_label_np.shape[0]
    if num_missing_points:
        last_point_info = past_traj_label_np[-1:]
        padding_info = np.tile(last_point_info, (num_missing_points, 1))
        past_traj_label_np = np.vstack((past_traj_label_np, padding_info))

    past_traj_label_t = torch.FloatTensor(past_traj_label_np).flatten()

    return past_traj_label_t
