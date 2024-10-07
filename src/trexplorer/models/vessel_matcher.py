# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified Hungarian Matcher for Texplorer
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        min_values = values.min()
        max_values = values.max()
        m = min(cost_matrix.shape)

        positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
        if max_inf:
            place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
        elif min_inf:
            place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    """

    def __init__(self, cost_class: float = 1, cost_direction: float = 1, cost_radius: float = 1,
                 class_dict=None, custom_focal_weights_value=None, num_bifur_queries=16):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_direction: This is the relative weight of the L1 error of the next point direction
                       in the matching cost
            cost_radius: This is the relative weight of the radius of next points in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_direction = cost_direction
        self.cost_radius = cost_radius
        self.focal_gamma = 2.0
        self.class_dict = class_dict
        self.custom_focal_weights_value = torch.tensor(custom_focal_weights_value).unsqueeze(0).to('cuda')
        self.num_bifur_queries = num_bifur_queries
        assert cost_class != 0 or cost_direction != 0 or cost_radius != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, prev_targets, prev_indices, bifur_list):
        """
        Performs the matching
        """
        batch_size, num_queries = outputs["class_logits"].shape[:2]

        tgt_direc = torch.cat([v["directions"] for v in targets])
        tgt_cls = torch.cat([v["labels"] for v in targets])
        tgt_cls_fg_idx = tgt_cls != self.class_dict['background']
        tgt_cls = tgt_cls[tgt_cls_fg_idx]
        tgt_direc = tgt_direc[tgt_cls_fg_idx]
        tgt_radii = torch.cat([v["radii"] for v in targets]).unsqueeze(dim=1)
        tgt_radii = tgt_radii[tgt_cls_fg_idx]

        sample_fg = [v["labels"] != self.class_dict['background'] for v in targets]
        sizes = [torch.count_nonzero(v).item() for v in sample_fg]
        fg_idx_to_all = [torch.nonzero(v, as_tuple=True)[0] for v in sample_fg]
        fg_idx_to_all_dict = [{index: element.item() for index, element in enumerate(v)} for v in fg_idx_to_all]

        if prev_targets:
            prev_num_branches = [len(v["labels"]) for v in prev_targets]
            curr_num_branches = [len(v["labels"]) for v in targets]
            if sum(curr_num_branches) == sum(prev_num_branches):
                return prev_indices

        out_cls = outputs["class_logits"].flatten(0, 1).sigmoid()
        out_direc = outputs["direc_logits"].flatten(0, 1)  # flatten first two dims
        out_radii = outputs["radius_logits"].flatten(0, 1)

        focal_alpha = self.custom_focal_weights_value
        neg_cost_class = (1 - focal_alpha) * (out_cls ** self.focal_gamma) * (-(1 - out_cls + 1e-8).log())
        pos_cost_class = focal_alpha * ((1 - out_cls) ** self.focal_gamma) * (-(out_cls + 1e-8).log())
        cost_cls = pos_cost_class[:, tgt_cls] - neg_cost_class[:, tgt_cls]
        cost_direc = torch.cdist(out_direc, tgt_direc, p=1)
        cost_radii = torch.cdist(out_radii, tgt_radii, p=1)
        cost_matrix = self.cost_class * cost_cls + self.cost_direction * cost_direc
        cost_matrix += self.cost_radius * cost_radii

        # if we have bifurcations, we match the targets to the allocated queries only
        if bifur_list:
            all_to_fg_idx_dict = [{element.item(): index
                                   for index, element in enumerate(v)} for v in fg_idx_to_all]
            cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
            cost_matrix_split = cost_matrix.split(sizes, -1)
            queries_mapped = []
            targets_mapped = []
            for sample_i, sample in enumerate(bifur_list):
                sample_queries_mapped = []
                sample_targets_mapped = []
                for group in sample:
                    group_queries = sample[group][1]
                    group_targets = sample[group][2]
                    group_targets_fg = [all_to_fg_idx_dict[sample_i][idx] for idx in group_targets]
                    group_cost_matrix = cost_matrix_split[sample_i][[sample_i]][:, group_queries][:, :, group_targets_fg].squeeze(0)
                    group_indices = linear_sum_assignment_with_inf(group_cost_matrix)
                    group_indices_mapped_fg = [[group_queries[idx] for idx in group_indices[0]],
                                            [group_targets_fg[idx] for idx in group_indices[1]]]
                    group_indices_mapped = [group_indices_mapped_fg[0], [fg_idx_to_all_dict[sample_i][idx]
                                                                         for idx in group_indices_mapped_fg[1]]]
                    sample_queries_mapped += group_indices_mapped[0]
                    sample_targets_mapped += group_indices_mapped[1]
                queries_mapped.append(sample_queries_mapped)
                targets_mapped.append(sample_targets_mapped)
            indices = [[torch.cat((prev_query, torch.as_tensor(queries_mapped[sample], dtype=torch.int64))),
                        torch.cat((prev_target, torch.as_tensor(targets_mapped[sample], dtype=torch.int64)))]
                       for sample, (prev_query, prev_target) in enumerate(prev_indices)]
        else:
            cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
            cost_matrix = cost_matrix[:, torch.arange(self.num_bifur_queries), :]
            indices = [list(linear_sum_assignment_with_inf(c[i])) for i, c in enumerate(cost_matrix.split(sizes, -1))]
            for sample_i, sample in enumerate(indices):
                sample[1] = np.array([fg_idx_to_all_dict[sample_i][idx] for idx in sample[1]])
            indices = [[torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)] for i, j in indices]

        for sample in indices:
            sorted_targets, sort_perm = torch.sort(sample[1])
            if (sample[1] != sorted_targets).any():
                for i in range(2):
                    sample[i] = sample[i][sort_perm]

        return indices


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_direction=args.set_cost_direction,
        cost_radius=args.set_cost_radius,
        class_dict=args.class_dict,
        custom_focal_weights_value=args.custom_focal_weights_value,
        num_bifur_queries=args.num_bifur_queries,
    )
