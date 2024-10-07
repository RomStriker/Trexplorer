# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import numpy as np
from typing import Iterable
import time
import networkx as nx
import tqdm
import torch
from .util import misc as utils
from src.trexplorer.util.misc import get_node, generate_past_trajectory_pp_nx
from src.trexplorer.datasets.transforms import CropAndPad
from monai.transforms import LoadImage, ScaleIntensity
from src.trexplorer.datasets.transforms import LoadAnnotPickle


class Trexplorer:
    def __init__(self, args, logger, device):
        self.args = args
        self.logger = logger
        self.device = device

    @staticmethod
    def get_single_step_targets(targets, step):
        """
        Extracting targets for current step
        """
        curr_step_targets = []
        for target in targets:
            curr_step_targets_dict = {'labels': torch.tensor(target['labels'][step]),
                                      'directions': torch.tensor(target['rel_positions'][step]).type(torch.FloatTensor),
                                      'radii': torch.tensor(target['radii'][step]).type(torch.FloatTensor),
                                      'parent_idxs': torch.tensor(target['parent_idxs'][step]).type(torch.LongTensor)}
            curr_step_targets.append(curr_step_targets_dict)

        return curr_step_targets

    def train_one_epoch(self, model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                        scaler: torch.cuda.amp.GradScaler):
        """
        Training function for one epoch
        """
        model.train()
        criterion.train()
        skip_lr_step = False
        epoch_metrics = {'scaled_losses': 0, 'unscaled_losses': 0, 'total_loss': 0, }
        for i, batch in enumerate(data_loader):
            sample_imgs, sample_past_trs, targets = (batch["image"], batch["past_tr"], batch["label"])
            if sample_imgs.device != self.device:
                sample_imgs = sample_imgs.to(self.device)
            if sample_past_trs is not None:
                sample_past_trs = sample_past_trs.to(self.device)

            prev_step_info = {"first_step": True,
                              "indices": [],
                              "out": None,
                              "bifur_list": None}
            losses = []
            batch_metrics = {'scaled_losses': [], 'unscaled_losses': []}
            prev_step_targets = None
            for step in range(1, self.args.seq_len):
                curr_step_targets = self.get_single_step_targets(targets, step)
                curr_step_targets = [utils.nested_dict_to_device(t, self.device) for t in curr_step_targets]

                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.device)
                    outputs, prev_step_info = model(sample_imgs, sample_past_trs, prev_step_info, norm_step,
                                                    curr_step_targets, prev_step_targets)
                    loss_dict, prev_step_info['indices'] = criterion(outputs, curr_step_targets, prev_step_targets, prev_step_info)
                    weight_dict = criterion.weight_dict
                    losses.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {
                    f'{k}_unscaled': v for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
                loss_dict_reduced_scaled = {
                    k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
                batch_metrics['scaled_losses'].append(losses_reduced_scaled.item())
                batch_metrics['unscaled_losses'].append(losses_reduced_unscaled.item())

                for item in loss_dict_reduced:
                    if item in batch_metrics:
                        batch_metrics[item].append(loss_dict_reduced[item].item())
                    else:
                        batch_metrics.update({item: [loss_dict_reduced[item].item()]})
                prev_step_targets = curr_step_targets

            for item in batch_metrics:
                if item in epoch_metrics:
                    epoch_metrics[item] += sum(batch_metrics[item]) / (self.args.seq_len - 1)
                else:
                    epoch_metrics.update({item: sum(batch_metrics[item]) / (self.args.seq_len - 1)})

            # sum the losses for all steps
            total_losses = sum(losses)
            epoch_metrics['total_loss'] += total_losses.item()
            epoch_metrics['scaled_losses'] /= self.args.dec_layers
            epoch_metrics['unscaled_losses'] /= self.args.dec_layers

            if self.args.amp:
                optimizer.zero_grad()
                scaler.scale(total_losses).backward()
                if self.args.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_max_norm)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_step = (scale > scaler.get_scale())
            else:
                optimizer.zero_grad()
                total_losses.backward()
                if self.args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_max_norm)
                optimizer.step()

            if skip_lr_step:
                self.logger.info("Skipping LR Scheduler step.")
            else:
                lr_scheduler.step()

        for item in epoch_metrics:
            epoch_metrics[item] = epoch_metrics[item] / len(data_loader)
        epoch_metrics['lr'] = lr_scheduler.get_last_lr()[0]

        return epoch_metrics

    def create_node_batch_nx_sv(self, targets):
        """
        Create a list of nodes from the targets
        """
        node_batch = [nx.DiGraph() for _ in range(len(targets))]
        for i, target in enumerate(targets):
            node_pos = np.array([self.args.sub_vol_size / 2, self.args.sub_vol_size / 2,
                                 self.args.sub_vol_size / 2]) + np.array(target['rel_positions'][0][0]) * self.args.sub_vol_size / 2
            node_batch[i].add_node('0-0',
                                   position=node_pos.tolist(),
                                   radius=target['radii'][0][0],
                                   label=target['labels'][0][0],
                                   level=0, rel_pos=[0, 0, 0],
                                   query_index=-1, step=-1, score=1.0)
        return node_batch

    @staticmethod
    def get_global_pred_root_node_nx(pred_tree, target_tree):
        """
        Create a Node for the root point
        """
        in_degrees = dict(target_tree.in_degree())
        root_node_id = [node for node, in_degree in in_degrees.items() if in_degree == 0][0]
        root_node_info = get_node(root_node_id, target_tree)

        root_pos = root_node_info['position']
        rel_pos = np.array(root_pos) - np.array(root_pos).astype(int)
        radius = root_node_info['radius']
        pred_tree.add_node(root_node_id, position=root_pos,
                           rel_pos=rel_pos, radius=radius, level=0, query_index=-1, score=1.0, bifur_parent='None')

        return root_node_info

    def get_classes_radii_positions(self, outputs):
        """
        Obtain the classes, radii, and positions for all the queries
        """
        query_classes = outputs['class_logits']
        query_classes = torch.argmax(query_classes, dim=2)
        query_radii = outputs['radius_logits']
        query_radii = query_radii * (self.args.sub_vol_size // 2)
        query_positions = outputs['direc_logits']
        query_positions = query_positions * (self.args.sub_vol_size // 2)

        return query_classes, query_radii, query_positions

    @staticmethod
    def get_cont_next_node_ar2_nx(query_index, step_node, pred_tree, curr_node_root_pos, query_positions,
                                  query_radii, level, step=None, query_classes_scores=None, global_occ_pos=None):
        """
        Get the continuing node
        """
        branch_id = int(step_node.split("-")[0])
        point_id = int(step_node.split("-")[1]) + 1
        query_index_t = torch.tensor(query_index).to(query_positions.device)
        rel_pos = query_positions.index_select(1, query_index_t).squeeze()
        radius = query_radii.index_select(1, query_index_t).squeeze()
        if query_classes_scores is not None:
            score = query_classes_scores.index_select(1, query_index_t).squeeze()[0]
        else:
            score = torch.tensor(0)
        position = curr_node_root_pos + rel_pos

        if global_occ_pos is not None:
            global_occ_pos[tuple(position.to(dtype=int).tolist())] = str(branch_id) + "-" + str(point_id)

        next_node = str(branch_id) + "-" + str(point_id)
        pred_tree.add_node(next_node,
                           position=position.tolist(),
                           rel_pos=rel_pos.tolist(),
                           radius=radius.item(),
                           query_index=query_index,
                           level=level,
                           step=step,
                           score=score.item(),
                           bifur_parent=get_node(step_node, pred_tree)['bifur_parent'])

        parent_pos = torch.tensor(get_node(step_node, pred_tree)['position']).to(query_positions.device)
        child_pos = torch.tensor(get_node(next_node, pred_tree)['position']).to(query_positions.device)
        length = torch.norm(child_pos - parent_pos).item()
        pred_tree.add_edge(step_node, next_node, length=length)

        return next_node

    @staticmethod
    def get_new_next_nodes_ar2_nx(new_branches, parent_node, pred_tree, query_positions, query_radii,
                                  curr_node_root_pos, global_branch_id, level, special_indices, step=None, query_classes_scores=None):
        """
        Get the new nodes originating at a bifurcation point
        """
        new_branch_node_list = []
        for query_index in new_branches:
            query_index_t = torch.tensor(query_index).to(query_positions.device)
            rel_pos = query_positions.index_select(1, query_index_t).squeeze()
            radius = query_radii.index_select(1, query_index_t).squeeze()
            if query_classes_scores is not None:
                score = query_classes_scores.index_select(1, query_index_t).squeeze()[0]
            else:
                score = torch.tensor(0)
            position = curr_node_root_pos + rel_pos

            if special_indices and query_index in special_indices[0]:
                branch_id = special_indices[1][special_indices[0].index(query_index)]
                if (len(special_indices[0]) == 1 and len(new_branches) == 1
                        and get_node(parent_node, pred_tree)['query_index'] == -1):
                    point_id = int(parent_node.split("-")[1]) + 1
                else:
                    point_id = 0

            else:
                global_branch_id += 1
                branch_id = global_branch_id
                point_id = 0

            new_node = str(branch_id) + "-" + str(point_id)
            pred_tree.add_node(new_node,
                               position=position.tolist(),
                               rel_pos=rel_pos.tolist(),
                               radius=radius.item(),
                               query_index=query_index,
                               level=level,
                               step=step,
                               score=score.item(),
                               bifur_parent=parent_node)
            pred_tree.add_edge(parent_node, new_node)
            parent_pos = torch.tensor(get_node(parent_node, pred_tree)['position']).to(query_positions.device)
            child_pos = torch.tensor(get_node(new_node, pred_tree)['position']).to(query_positions.device)
            length = torch.norm(child_pos - parent_pos).item()
            pred_tree.add_edge(parent_node, new_node, length=length)
            new_branch_node_list.append(new_node)

        return new_branch_node_list, global_branch_id

    @staticmethod
    def get_updated_indices_nx(indices, curr_step, pred_tree):
        """
        Combine the indices from the previous step with the current step
        """
        if len(indices) == 0 or sum(len(lst) for lst in indices) == 0:
            branch_indices = [int(node.split("-")[0]) for node in curr_step]
            query_indices = [get_node(node, pred_tree)['query_index'] for node in curr_step]
            indices = [[torch.tensor(query_indices), torch.tensor(branch_indices)]]
        else:
            indices_list = [indices[0][0].tolist(), indices[0][1].tolist()]
            query_indices = [get_node(node_i, pred_tree)['query_index'] for node_i in curr_step
                             if get_node(node_i, pred_tree)['query_index'] not in indices_list[0]]
            branch_indices = [int(node_i.split("-")[0]) for node_i in curr_step
                              if (int(node_i.split("-")[0]) not in indices_list[1])]
            indices = [[torch.tensor(indices_list[0] + query_indices),
                        torch.tensor(indices_list[1] + branch_indices)]]

        for sample in indices:
            sorted_targets, sort_perm = torch.sort(sample[1])
            if (sample[1] != sorted_targets).any():
                for i in range(2):
                    sample[i] = sample[i][sort_perm]

        if indices[0][0].size(0) == 0:
            indices[0] = []

        return indices

    def log_progress(self, tree_id):
        if self.args.eval_only:
            self.logger.info('Tree ID: %d' % tree_id)

    def check_finished(self, targets, tree_id, curr_level, level):
        break_loop = False
        if level >= self.args.max_inference_levels and self.args.eval_limit_levels:
            self.logger.info(f'Max inference levels reached! Sample: {str(targets[0]["index"])}, Tree ID: {tree_id:d}')
            break_loop = True

        if len(curr_level) > self.args.max_nodes_per_level and self.args.eval_limit_nodes_per_level:
            self.logger.info('Max nodes in a level reached! Skipping further evaluation.  '
                  'Sample: %s, Tree ID: %d' % (targets[0]['index'], tree_id))
            break_loop = True
        return break_loop

    def log_progress_level(self, curr_level, level):
        self.logger.info('Level: %d \t | \t Nodes: %d \t ' % (level, len(curr_level)))

    def get_sub_vol_and_past_tr(self, node_batch, samples, pred_tree, crop_pad):
        """
        Get the sub-volume and past trajectory for the nodes in the node_batch
        """
        sub_vol_batch = []
        past_traj_pos_batch = []
        for enu_i, node in enumerate(node_batch):
            image_size = torch.tensor(samples.squeeze().size())
            node_pos = torch.tensor(get_node(node, pred_tree)['position'])
            if torch.logical_and(torch.all(node_pos < (image_size - 1)), torch.all(node_pos >= 0)):
                sub_vol = crop_pad(samples.squeeze(dim=0), node_pos.tolist(), -1.0)
                sub_vol = sub_vol.unsqueeze(0)
            else:
                node_batch.pop(enu_i)
                continue

            # A quickfix
            if len(sub_vol.shape) == 6:
                sub_vol = sub_vol.squeeze(0)

            past_traj_pos = generate_past_trajectory_pp_nx(pred_tree, node, self.args.num_prev_pos,
                                                           self.args.sub_vol_size)
            past_traj_pos = past_traj_pos.to(self.device).unsqueeze(0)
            sub_vol_batch.append(sub_vol)
            past_traj_pos_batch.append(past_traj_pos)
        return sub_vol_batch, past_traj_pos_batch

    def update_perma_finished_branches_bifur(self, node_perma_finished_branches, query_classes, selected_queries):
        """
        Update the permanently finished branches list
        """
        bifur_end_mask = ((query_classes == self.args.class_dict['bifurcation']) |
                          (query_classes == self.args.class_dict['end']))
        bifur_end_query_idxs = torch.nonzero(bifur_end_mask, as_tuple=True)[1]
        perma_finished_queries = [query_idx for query_idx in selected_queries
                                  if query_idx in bifur_end_query_idxs.tolist()]
        if len(perma_finished_queries):
            node_perma_finished_branches += perma_finished_queries

    @staticmethod
    def get_old_to_new_idx_dict(prev_step_info):
        """
        Get the mapping of old indices to new indices
        """
        old_indices = prev_step_info['old_indices']
        new_indices = prev_step_info['indices']
        if len(old_indices) == 0 or sum(len(lst) for lst in old_indices) == 0:
            return None
        else:
            same_indices = (torch.cat([s_idx[0] for s_idx in old_indices if len(s_idx)], dim=0) ==
                            torch.cat([s_idx[0] for s_idx in new_indices if len(s_idx)], dim=0)).all()
            if same_indices:
                return None
            else:
                index_mapping = []
                for old_sample_idxs, new_sample_idx in zip(old_indices, new_indices):
                    if not len(old_sample_idxs):
                        index_mapping.append(None)
                    else:
                        index_mapping.append({old_idx: new_idx for old_idx, new_idx
                                              in zip(old_sample_idxs[0].tolist(), new_sample_idx[0].tolist())})
                return index_mapping

    def map_old_to_new_indices(self, pred_tree, curr_step_batch, node_finished_branches_batch,
                               finished_branches_end_nodes_batch, node_perma_finished_branches_batch, prev_step_info):
        """
        Map new indices to old indices
        """
        index_mapping = self.get_old_to_new_idx_dict(prev_step_info)
        if index_mapping:
            for sample_id, sample_mapping in enumerate(index_mapping):
                if sample_mapping:
                    for node in curr_step_batch[sample_id]:
                        pred_tree.nodes[node]['query_index'] = sample_mapping[pred_tree.nodes[node]['query_index']]

                    if len(node_finished_branches_batch[sample_id]):
                        node_finished_branches_batch[sample_id] = [sample_mapping[node] if node != -1 else -1
                                                                   for node in node_finished_branches_batch[sample_id]]
                        for node in finished_branches_end_nodes_batch[sample_id]:
                            node_query_index = pred_tree.nodes[node]['query_index']
                            pred_tree.nodes[node]['query_index'] = sample_mapping[node_query_index] if node_query_index != -1 else -1

                    if len(node_perma_finished_branches_batch[sample_id]):
                        node_perma_finished_branches_batch[sample_id] = [sample_mapping[node]
                                                                         for node in node_perma_finished_branches_batch[sample_id]]

    def map_old_to_new_indices_sv(self, pred_tree_batch, curr_step_batch, node_finished_branches_batch,
                                  finished_branches_end_nodes_batch, node_perma_finished_branches_batch, prev_step_info):
        """
        Map new indices to old indices for sub-volume evaluation
        """
        index_mapping = self.get_old_to_new_idx_dict(prev_step_info)
        if index_mapping:
            for sample_id, sample_mapping in enumerate(index_mapping):
                if sample_mapping:
                    pred_tree = pred_tree_batch[sample_id]
                    for node in curr_step_batch[sample_id]:
                        pred_tree.nodes[node]['query_index'] = sample_mapping[pred_tree.nodes[node]['query_index']]

                    if len(node_finished_branches_batch[sample_id]):
                        node_finished_branches_batch[sample_id] = [sample_mapping[node] if node != -1 else -1
                                                                   for node in node_finished_branches_batch[sample_id]]
                        for node in finished_branches_end_nodes_batch[sample_id]:
                            node_query_index = pred_tree.nodes[node]['query_index']
                            pred_tree.nodes[node]['query_index'] = sample_mapping[node_query_index] if node_query_index != -1 else -1

                    if len(node_perma_finished_branches_batch[sample_id]):
                        node_perma_finished_branches_batch[sample_id] = [sample_mapping[node]
                                                                         for node in node_perma_finished_branches_batch[sample_id]]

    def add_new_bifur_branches_points(self, bifur_dict, pred_tree, new_branches, curr_step, finished_branches_end_nodes,
                                      global_branch_id, query_positions, query_radii, curr_node_root_pos, level, step,
                                      query_classes_scores, global_occ_pos, next_step, node_perma_finished_branches, query_classes, selected_queries):
        """
        Add new bifurcation branches points
        """
        special_indices = None
        for bifur_id in bifur_dict:
            allocated_bifur_queries = bifur_dict[bifur_id][1]
            if len(allocated_bifur_queries):
                unused_buffer_queries = [idx for idx in allocated_bifur_queries if idx not in new_branches]
                used_bifur_queries = [idx for idx in allocated_bifur_queries if idx not in unused_buffer_queries]
                parent_query_index = bifur_dict[bifur_id][0]
                parent_node = [step_node for step_node in curr_step + finished_branches_end_nodes if
                               get_node(step_node, pred_tree)['query_index'] == parent_query_index][0]
                new_branch_node_list, global_branch_id[0] = self.get_new_next_nodes_ar2_nx(used_bifur_queries,
                                                                                           parent_node, pred_tree, query_positions, query_radii,
                                                                                           curr_node_root_pos, global_branch_id[0], level,
                                                                                           special_indices, step - 1, query_classes_scores)
                for next_step_node in new_branch_node_list:
                    node_pos = np.array(get_node(next_step_node, pred_tree)['position']).astype(int).tolist()
                    if global_occ_pos is not None:
                        global_occ_pos[tuple(node_pos)] = next_step_node
                next_step += new_branch_node_list
                self.update_perma_finished_branches_bifur(node_perma_finished_branches, query_classes, selected_queries)

    def add_new_bifur_branches_points_beg(self, global_branch_id, selected_queries, curr_step, pred_tree, query_positions, query_radii,
                                          curr_node_root_pos, level, step, query_classes_scores, global_occ_pos, next_step, node_finished_branches,
                                          finished_branches_end_nodes, node_perma_finished_branches, query_classes):
        """
        Add new bifurcation branches points for the first step
        """
        special_indices = None
        new_branch_node_list, global_branch_id[0] = self.get_new_next_nodes_ar2_nx(selected_queries,
                                                                                   curr_step[0], pred_tree, query_positions, query_radii,
                                                                                   curr_node_root_pos, global_branch_id[0], level,
                                                                                   special_indices, step - 1, query_classes_scores)
        for next_step_node in new_branch_node_list:
            node_pos = np.array(get_node(next_step_node, pred_tree)['position']).astype(int).tolist()
            if global_occ_pos is not None:
                global_occ_pos[tuple(node_pos)] = next_step_node
        next_step += new_branch_node_list
        finished_branches = [-1]
        node_finished_branches += finished_branches
        finished_branches_end_nodes += [end_node for end_node in curr_step
                                        if get_node(end_node, pred_tree)['query_index'] in finished_branches]
        self.update_perma_finished_branches_bifur(node_perma_finished_branches, query_classes, selected_queries)

    def add_cont_branches_points(self, continuing_branches, previous_selected_queries, curr_step,
                                 pred_tree, curr_node_root_pos, query_positions, query_radii, level,
                                 step, query_classes_scores, global_occ_pos, next_step):
        """
        Add continuing branches points
        """
        for query_index in continuing_branches:
            node_index = previous_selected_queries.index(query_index)
            step_node = curr_step[node_index]
            next_step_node = self.get_cont_next_node_ar2_nx(query_index, step_node,
                                                            pred_tree, curr_node_root_pos, query_positions, query_radii,
                                                            level, step - 1, query_classes_scores, global_occ_pos)
            if next_step_node is not None:
                next_step.append(next_step_node)

    def add_cont_branch_point_beg(self, selected_queries, curr_step, pred_tree, curr_node_root_pos, query_positions,
                                  query_radii, level, step, query_classes_scores, global_occ_pos, next_step):
        """
        Add continuing branches points for the first step
        """
        next_step_node = self.get_cont_next_node_ar2_nx(selected_queries[0],
                                                        curr_step[0], pred_tree, curr_node_root_pos, query_positions,
                                                        query_radii, level, step - 1, query_classes_scores, global_occ_pos)
        if next_step_node is not None:
            next_step.append(next_step_node)

    @staticmethod
    def init_req_lists(self, indices, pred_tree, curr_step, selected_queries,
                       node_finished_branches, finished_branches_end_nodes):
        """
        Initialize the required lists
        """
        used_queries = indices[0][0].tolist() if len(indices[0]) else []
        previous_selected_queries = [get_node(node_i, pred_tree)['query_index'] for node_i in curr_step]
        new_branches = (list(set(selected_queries) - set(used_queries)))
        continuing_branches = [x for x in selected_queries if x in used_queries]
        finished_branches = list(set(previous_selected_queries) - set(selected_queries))
        node_finished_branches += finished_branches
        finished_branches_end_nodes += [end_node for end_node in curr_step
                                        if get_node(end_node, pred_tree)['query_index'] in finished_branches]

        return continuing_branches, new_branches

    @torch.no_grad()
    def evaluate_val(self, model, criterion, data_loader):
        """
        Validation function for full volume evaluation
        """
        model.eval()
        criterion.eval()
        all_masks = []
        all_samples = []
        all_samples_ids = []
        all_preds = []
        all_targets = []
        elapsed_time = []
        crop_pad = CropAndPad(self.args.sub_vol_size)

        # Only batch_size 1 supported for now
        for i, batch in enumerate(data_loader):
            samples, samples_min, targets, masks = (batch["image"], batch["image_min"], batch["label"], batch["mask"])
            if self.args.eval_only:
                self.logger.info(f'Sample: {targets[0]["index"]:d}')
            for tree_id in range(len(targets[0]['networkx'])):
                st = time.time()
                self.log_progress(tree_id)
                finished = False
                samples = samples.to(self.args.device)
                global_occ_pos = {}
                global_branch_id = [0]
                level = 0
                target_tree = targets[0]['networkx'][tree_id]
                pred_tree = nx.DiGraph()
                root_node = self.get_global_pred_root_node_nx(pred_tree, target_tree)
                global_occ_pos[tuple(np.array(root_node['position']).astype(int).tolist())] = "0-0"
                curr_level = [root_node['id']]
                while not finished:
                    if self.check_finished(targets, tree_id, curr_level, level):
                        break
                    if self.args.eval_only:
                        self.log_progress_level(curr_level, level)
                    next_level = []
                    for level_node_i in range(0, len(curr_level), self.args.batch_size_per_sample):
                        node_batch = curr_level[level_node_i: level_node_i + self.args.batch_size_per_sample]
                        sub_vol_batch, past_traj_pos_batch = self.get_sub_vol_and_past_tr(node_batch, samples,
                                                                                          pred_tree, samples_min, crop_pad)
                        if not len(sub_vol_batch):
                            continue

                        prev_step_info = {"first_step": True,
                                          "indices": [[] for _ in range(len(node_batch))],
                                          "out": None,
                                          "bifur_list": None,
                                          "starting_node_id": [int(node.split("-")[0]) for node in node_batch],
                                          "global_id": global_branch_id}
                        finish_flag = [False for i in range(len(node_batch))]
                        sub_vol = torch.cat(sub_vol_batch, dim=0)
                        past_traj_pos = torch.cat(past_traj_pos_batch, dim=0)
                        curr_step_batch = [[node] for node in node_batch]
                        curr_node_root_pos_batch = [torch.tensor(get_node(node, pred_tree)['position'],
                                                                 dtype=int).to(self.args.device) for node in node_batch]
                        node_finished_branches_batch = [[] for i in range(len(node_batch))]
                        finished_branches_end_nodes_batch = [[] for i in range(len(node_batch))]
                        node_perma_finished_branches_batch = [[] for i in range(len(node_batch))]

                        for step in range(1, self.args.seq_len):
                            num_all_nodes = sum([len(sublist) for sublist in curr_step_batch])
                            if not num_all_nodes:
                                break
                            next_step_batch = [[] for i in range(len(node_batch))]

                            norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                            outputs, prev_step_info = model(sub_vol, past_traj_pos, prev_step_info, norm_step)
                            self.map_old_to_new_indices(pred_tree,
                                                        curr_step_batch,
                                                        node_finished_branches_batch,
                                                        finished_branches_end_nodes_batch,
                                                        node_perma_finished_branches_batch,
                                                        prev_step_info)

                            query_classes_batch, query_radii_batch, query_positions_batch = self.get_classes_radii_positions(outputs)
                            query_classes_scores_batch = outputs['class_logits'].detach()
                            query_classes_scores_batch = torch.softmax(query_classes_scores_batch, dim=-1)
                            query_classes_scores_batch = torch.stack([query_classes_scores_batch[:, :, :3].sum(dim=2),
                                                                      query_classes_scores_batch[:, :, 3]], dim=2)
                            filter_class_ids = torch.tensor([self.args.class_dict['background']]).to(query_classes_batch.device)
                            query_classes_batch_filtered = torch.logical_not(torch.isin(query_classes_batch, filter_class_ids))
                            selected_queries_batch = [query_classes_batch_filtered[i].nonzero().flatten().tolist()
                                                      for i in range(query_classes_batch_filtered.shape[0])]

                            if step == 1:
                                first_step_queries = list(range(self.args.num_bifur_queries))
                                selected_queries_batch = [[query for query in selected_queries if query in first_step_queries]
                                                          for selected_queries in selected_queries_batch]

                            selected_queries_batch = [[elem for elem in sublist1 if elem not in sublist2]
                                                      for sublist1, sublist2 in zip(selected_queries_batch, node_perma_finished_branches_batch)]
                            for sq_i, selected_queries in enumerate(selected_queries_batch):
                                if len(selected_queries) == 0 and not finish_flag[sq_i]:
                                    finish_flag[sq_i] = True
                                if finish_flag[sq_i]:
                                    curr_step_batch[sq_i] = []
                                    continue

                                indices = [prev_step_info['indices'][sq_i]]
                                next_step = next_step_batch[sq_i]
                                curr_step = curr_step_batch[sq_i]
                                curr_node_root_pos = curr_node_root_pos_batch[sq_i]
                                node_finished_branches = node_finished_branches_batch[sq_i]
                                finished_branches_end_nodes = finished_branches_end_nodes_batch[sq_i]
                                node_perma_finished_branches = node_perma_finished_branches_batch[sq_i]
                                query_radii = query_radii_batch[sq_i:sq_i + 1]
                                query_positions = query_positions_batch[sq_i:sq_i + 1]
                                query_classes = query_classes_batch[sq_i:sq_i + 1]
                                query_classes_scores = query_classes_scores_batch[sq_i:sq_i + 1]
                                previous_selected_queries = [get_node(node_i, pred_tree)['query_index'] for node_i in curr_step]

                                if previous_selected_queries == [-1]:
                                    if len(selected_queries) == 1:
                                        self.add_cont_branch_point_beg(selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                                       query_positions, query_radii, level, step, query_classes_scores, global_occ_pos,
                                                                       next_step)

                                    else:
                                        self.add_new_bifur_branches_points_beg(global_branch_id, selected_queries, curr_step, pred_tree,
                                                                               query_positions, query_radii, curr_node_root_pos, level, step, query_classes_scores,
                                                                               global_occ_pos, next_step, node_finished_branches, finished_branches_end_nodes,
                                                                               node_perma_finished_branches, query_classes)

                                else:
                                    continuing_branches, new_branches = self.init_req_lists(indices, pred_tree,
                                                                                            curr_step, selected_queries,
                                                                                            node_finished_branches, finished_branches_end_nodes)

                                    self.add_cont_branches_points(continuing_branches, previous_selected_queries, curr_step,
                                                                  pred_tree, curr_node_root_pos, query_positions, query_radii, level, step,
                                                                  query_classes_scores, global_occ_pos, next_step)

                                    if prev_step_info['bifur_list'] and len(prev_step_info['bifur_list'][sq_i]):
                                        bifur_dict = prev_step_info['bifur_list'][sq_i]
                                        self.add_new_bifur_branches_points(bifur_dict, pred_tree, new_branches, curr_step, finished_branches_end_nodes,
                                                                           global_branch_id, query_positions, query_radii, curr_node_root_pos, level, step,
                                                                           query_classes_scores, global_occ_pos, next_step, node_perma_finished_branches, query_classes)

                                curr_step = next_step
                                indices = self.get_updated_indices_nx(indices, curr_step, pred_tree)
                                prev_step_info['indices'][sq_i] = indices[0]
                                curr_step_batch[sq_i] = curr_step

                    for node in next_level:
                        node_info = get_node(node, pred_tree)
                        node_pos = node_info['position']
                        node_info['query_index'] = -1
                        node_info['rel_pos'] = np.array(node_pos) - np.array(node_pos).astype(int)

                    curr_level = next_level
                    if len(curr_level) == 0:
                        finished = True
                    level += 1

                all_preds.append(pred_tree)
                all_targets.append(targets[0]['networkx'][tree_id])
                et = time.time()
                elapsed_time.append(et - st)
                all_samples_ids.append(targets[0]['index'])
            if self.args.mask:
                all_masks.append(masks)
            all_samples.append(samples)

        return all_preds, all_targets, all_samples_ids, all_samples, all_masks, elapsed_time

    @torch.no_grad()
    def evaluate_sinsam(self, model, criterion, sample_id):
        model.eval()
        criterion.eval()
        crop_pad = CropAndPad(self.args.sub_vol_size)

        image_reader = LoadImage(image_only=True)
        annot_reader = LoadAnnotPickle()
        norm = ScaleIntensity(minv=-1.0)

        annot_dir = os.path.join(self.args.data_dir, 'annots_cont_sep_test')
        msk_dir = os.path.join(self.args.data_dir, 'masks_sep_test')
        img_dir = os.path.join(self.args.data_dir, 'images_sep_test')

        img_path = os.path.join(img_dir, sample_id + '.nii.gz')
        msk_path = os.path.join(msk_dir, sample_id + '.nii.gz')
        annot_path = os.path.join(annot_dir, sample_id + '.pickle')

        samples = norm(image_reader(img_path)).unsqueeze(0).unsqueeze(0).to(self.args.device)
        targets = [annot_reader(annot_path)]
        targets[0]['index'] = sample_id
        masks = image_reader(msk_path).unsqueeze(0).unsqueeze(0)

        if self.args.eval_only:
            self.logger.info(f'Sample: {str(targets[0]["index"])}')

        for tree_id in range(len(targets[0]['networkx'])):
            st = time.time()
            self.log_progress(tree_id)
            finished = False
            samples = samples.to(self.args.device)
            global_occ_pos = {}
            global_branch_id = [0]
            level = 0
            target_tree = targets[0]['networkx'][tree_id]
            pred_tree = nx.DiGraph()
            root_node = self.get_global_pred_root_node_nx(pred_tree, target_tree)
            global_occ_pos[tuple(np.array(root_node['position']).astype(int).tolist())] = "0-0"
            curr_level = [root_node['id']]
            while not finished:
                if self.check_finished(targets, tree_id, curr_level, level):
                    break
                if self.args.eval_only:
                    self.log_progress_level(curr_level, level)

                next_level = []
                for level_node_i in range(0, len(curr_level), self.args.batch_size_per_sample):
                    node_batch = curr_level[level_node_i: level_node_i + self.args.batch_size_per_sample]
                    sub_vol_batch, past_traj_pos_batch = self.get_sub_vol_and_past_tr(node_batch, samples,
                                                                                      pred_tree, crop_pad)
                    if not len(sub_vol_batch):
                        continue

                    prev_step_info = {"first_step": True,
                                      "indices": [[] for _ in range(len(node_batch))],
                                      "out": None,
                                      "bifur_list": None,
                                      "starting_node_id": [int(node.split("-")[0]) for node in node_batch],
                                      "global_id": global_branch_id}

                    finish_flag = [False for i in range(len(node_batch))]

                    sub_vol = torch.cat(sub_vol_batch, dim=0)
                    past_traj_pos = torch.cat(past_traj_pos_batch, dim=0)
                    curr_step_batch = [[node] for node in node_batch]
                    curr_node_root_pos_batch = [torch.tensor(get_node(node, pred_tree)['position'],
                                                             dtype=int).to(self.args.device) for node in node_batch]
                    node_finished_branches_batch = [[] for i in range(len(node_batch))]
                    finished_branches_end_nodes_batch = [[] for i in range(len(node_batch))]
                    node_perma_finished_branches_batch = [[] for i in range(len(node_batch))]

                    for step in range(1, self.args.seq_len):
                        num_all_nodes = sum([len(sublist) for sublist in curr_step_batch])
                        if not num_all_nodes:
                            break
                        next_step_batch = [[] for i in range(len(node_batch))]

                        norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                        outputs, prev_step_info = model(sub_vol, past_traj_pos, prev_step_info, norm_step)
                        self.map_old_to_new_indices(pred_tree,
                                                    curr_step_batch,
                                                    node_finished_branches_batch,
                                                    finished_branches_end_nodes_batch,
                                                    node_perma_finished_branches_batch,
                                                    prev_step_info)

                        query_classes_batch, query_radii_batch, query_positions_batch = self.get_classes_radii_positions(outputs)
                        query_classes_scores_batch = outputs['class_logits'].detach()
                        query_classes_scores_batch = torch.softmax(query_classes_scores_batch, dim=-1)
                        query_classes_scores_batch = torch.stack([query_classes_scores_batch[:, :, :3].sum(dim=2),
                                                                  query_classes_scores_batch[:, :, 3]], dim=2)
                        filter_class_ids = torch.tensor([self.args.class_dict['background']]).to(query_classes_batch.device)
                        query_classes_batch_filtered = torch.logical_not(torch.isin(query_classes_batch, filter_class_ids))
                        selected_queries_batch = [query_classes_batch_filtered[i].nonzero().flatten().tolist()
                                                  for i in range(query_classes_batch_filtered.shape[0])]

                        if step == 1:
                            first_step_queries = list(range(self.args.num_bifur_queries))
                            selected_queries_batch = [[query for query in selected_queries if query in first_step_queries]
                                                      for selected_queries in selected_queries_batch]

                        selected_queries_batch = [[elem for elem in sublist1 if elem not in sublist2]
                                                  for sublist1, sublist2 in zip(selected_queries_batch, node_perma_finished_branches_batch)]

                        for sq_i, selected_queries in enumerate(selected_queries_batch):
                            if len(selected_queries) == 0 and not finish_flag[sq_i]:
                                finish_flag[sq_i] = True
                            if finish_flag[sq_i]:
                                curr_step_batch[sq_i] = []
                                continue

                            indices = [prev_step_info['indices'][sq_i]]
                            next_step = next_step_batch[sq_i]
                            curr_step = curr_step_batch[sq_i]
                            curr_node_root_pos = curr_node_root_pos_batch[sq_i]
                            node_finished_branches = node_finished_branches_batch[sq_i]
                            finished_branches_end_nodes = finished_branches_end_nodes_batch[sq_i]
                            node_perma_finished_branches = node_perma_finished_branches_batch[sq_i]
                            query_radii = query_radii_batch[sq_i:sq_i + 1]
                            query_positions = query_positions_batch[sq_i:sq_i + 1]
                            query_classes = query_classes_batch[sq_i:sq_i + 1]
                            query_classes_scores = query_classes_scores_batch[sq_i:sq_i + 1]
                            previous_selected_queries = [get_node(node_i, pred_tree)['query_index'] for node_i in curr_step]

                            if previous_selected_queries == [-1]:
                                if len(selected_queries) == 1:
                                    self.add_cont_branch_point_beg(selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                                   query_positions, query_radii, level, step, query_classes_scores, global_occ_pos,
                                                                   next_step)
                                else:
                                    self.add_new_bifur_branches_points_beg(global_branch_id, selected_queries, curr_step, pred_tree,
                                                                           query_positions, query_radii, curr_node_root_pos, level, step, query_classes_scores,
                                                                           global_occ_pos, next_step, node_finished_branches, finished_branches_end_nodes,
                                                                           node_perma_finished_branches, query_classes)

                            else:
                                continuing_branches, new_branches = self.init_req_lists(indices, pred_tree,
                                                                                        curr_step, selected_queries,
                                                                                        node_finished_branches, finished_branches_end_nodes)
                                self.add_cont_branches_points(continuing_branches, previous_selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                              query_positions, query_radii, level, step, query_classes_scores, global_occ_pos, next_step)

                                if prev_step_info['bifur_list'] and len(prev_step_info['bifur_list'][sq_i]):
                                    bifur_dict = prev_step_info['bifur_list'][sq_i]
                                    self.add_new_bifur_branches_points(bifur_dict, pred_tree, new_branches, curr_step, finished_branches_end_nodes,
                                                                       global_branch_id, query_positions, query_radii, curr_node_root_pos, level, step,
                                                                       query_classes_scores, global_occ_pos, next_step, node_perma_finished_branches, query_classes)

                            curr_step = next_step
                            indices = self.get_updated_indices_nx(indices, curr_step, pred_tree)
                            prev_step_info['indices'][sq_i] = indices[0]
                            curr_step_batch[sq_i] = curr_step

                for node in next_level:
                    node_info = get_node(node, pred_tree)
                    node_pos = node_info['position']
                    node_info['query_index'] = -1
                    node_info['rel_pos'] = np.array(node_pos) - np.array(node_pos).astype(int)

                curr_level = next_level
                if len(curr_level) == 0:
                    finished = True
                level += 1

        return [pred_tree], [targets[0]['networkx'][tree_id]], [targets[0]['index']], [samples], [masks], [time.time() - st]

    @torch.no_grad()
    def evaluate_sv(self, model, criterion, data_loader):
        model.eval()
        criterion.eval()
        all_masks = []
        all_samples = []
        all_samples_ids = []
        all_preds = []
        all_targets = []
        elapsed_time = []
        level = 0
        for i, batch in tqdm.tqdm(enumerate(data_loader), desc="Batch", leave=False, total=len(data_loader)):
            start = time.time()
            sample_imgs, sample_past_trs, targets = (batch["image"], batch["past_tr"], batch["label"])
            if self.args.mask:
                masks = batch["mask"]
            if sample_imgs.device != self.args.device:
                sample_imgs = sample_imgs.to(self.args.device)
            if sample_past_trs is not None:
                sample_past_trs = sample_past_trs.to(self.args.device)

            pred_tree_batch = self.create_node_batch_nx_sv(targets)
            node_batch = ['0-0' for _ in range(len(pred_tree_batch))]
            global_branch_id = [0]
            prev_step_info = {"first_step": True,
                              "indices": [[] for _ in range(len(node_batch))],
                              "out": None,
                              "bifur_list": None,
                              "starting_node_id": [int(node.split("-")[0]) for node in node_batch],
                              "global_id": global_branch_id}
            finish_flag = [False for i in range(len(node_batch))]
            sub_vol = sample_imgs
            past_traj_pos = sample_past_trs
            curr_step_batch = [[node] for node in node_batch]
            curr_node_root_pos_batch = [torch.tensor(get_node('0-0', pred_tree)['position'], dtype=int).to(self.args.device)
                                        for pred_tree in pred_tree_batch]
            node_finished_branches_batch = [[] for i in range(len(node_batch))]
            finished_branches_end_nodes_batch = [[] for i in range(len(node_batch))]
            node_perma_finished_branches_batch = [[] for i in range(len(node_batch))]
            for s_i, step in enumerate(range(1, self.args.seq_len)):
                next_step_batch = [[] for i in range(len(node_batch))]
                norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                outputs, prev_step_info = model(sub_vol, past_traj_pos, prev_step_info, norm_step)
                self.map_old_to_new_indices_sv(pred_tree_batch,
                                               curr_step_batch,
                                               node_finished_branches_batch,
                                               finished_branches_end_nodes_batch,
                                               node_perma_finished_branches_batch,
                                               prev_step_info)

                query_classes_batch, query_radii_batch, query_positions_batch = self.get_classes_radii_positions(outputs)
                query_classes_scores_batch = outputs['class_logits'].detach()
                query_classes_scores_batch = torch.softmax(query_classes_scores_batch, dim=-1)
                query_classes_scores_batch = torch.stack([query_classes_scores_batch[:, :, :3].sum(dim=2),
                                                          query_classes_scores_batch[:, :, 3]], dim=2)
                filter_class_ids = torch.tensor([self.args.class_dict['background']]).to(query_classes_batch.device)
                query_classes_batch_filtered = torch.logical_not(torch.isin(query_classes_batch, filter_class_ids))
                selected_queries_batch = [query_classes_batch_filtered[i].nonzero().flatten().tolist()
                                          for i in range(query_classes_batch_filtered.shape[0])]

                if step == 1:
                    first_step_queries = list(range(self.args.num_bifur_queries))
                    selected_queries_batch = [[query for query in selected_queries if query in first_step_queries]
                                              for selected_queries in selected_queries_batch]

                selected_queries_batch = [[elem for elem in sublist1 if elem not in sublist2]
                                          for sublist1, sublist2 in zip(selected_queries_batch, node_perma_finished_branches_batch)]
                for sq_i, selected_queries in enumerate(selected_queries_batch):
                    if len(selected_queries) == 0 and not finish_flag[sq_i]:
                        finish_flag[sq_i] = True
                    if finish_flag[sq_i]:
                        curr_step_batch[sq_i] = []
                        continue
                    pred_tree = pred_tree_batch[sq_i]
                    indices = [prev_step_info['indices'][sq_i]]
                    next_step = next_step_batch[sq_i]
                    curr_step = curr_step_batch[sq_i]
                    curr_node_root_pos = curr_node_root_pos_batch[sq_i]
                    node_finished_branches = node_finished_branches_batch[sq_i]
                    finished_branches_end_nodes = finished_branches_end_nodes_batch[sq_i]
                    node_perma_finished_branches = node_perma_finished_branches_batch[sq_i]
                    query_radii = query_radii_batch[sq_i:sq_i + 1]
                    query_positions = query_positions_batch[sq_i:sq_i + 1]
                    query_classes = query_classes_batch[sq_i:sq_i + 1]
                    query_classes_scores = query_classes_scores_batch[sq_i:sq_i + 1]
                    previous_selected_queries = [get_node(node_i, pred_tree)['query_index'] for node_i in curr_step]
                    if previous_selected_queries == [-1]:
                        if len(selected_queries) == 1:
                            self.add_cont_branch_point_beg(selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                           query_positions, query_radii, level, step, query_classes_scores, None,
                                                           next_step)
                        else:
                            self.add_new_bifur_branches_points_beg(global_branch_id, selected_queries, curr_step, pred_tree,
                                                                   query_positions, query_radii, curr_node_root_pos, level, step, query_classes_scores,
                                                                   None, next_step, node_finished_branches, finished_branches_end_nodes,
                                                                   node_perma_finished_branches, query_classes)
                    else:
                        continuing_branches, new_branches = self.init_req_lists(indices, pred_tree,
                                                                                curr_step, selected_queries,
                                                                                node_finished_branches, finished_branches_end_nodes)
                        self.add_cont_branches_points(continuing_branches, previous_selected_queries, curr_step, pred_tree,
                                                      curr_node_root_pos, query_positions, query_radii, level, step, query_classes_scores, None, next_step)

                        if prev_step_info['bifur_list'] and len(prev_step_info['bifur_list'][sq_i]):
                            bifur_dict = prev_step_info['bifur_list'][sq_i]
                            self.add_new_bifur_branches_points(bifur_dict, pred_tree, new_branches, curr_step, finished_branches_end_nodes,
                                                               global_branch_id, query_positions, query_radii, curr_node_root_pos, level, step,
                                                               query_classes_scores, None, next_step, node_perma_finished_branches, query_classes)

                    curr_step = next_step
                    indices = self.get_updated_indices_nx(indices, curr_step, pred_tree)
                    prev_step_info['indices'][sq_i] = indices[0]
                    curr_step_batch[sq_i] = curr_step

            end = time.time()
            elp_time = (end - start) / len(targets)
            elapsed_time += [elp_time for _ in range(len(targets))]
            all_preds += pred_tree_batch
            for sam_i in range(len(targets)):
                all_targets.append(targets[sam_i]['selected_node'])
                all_samples.append(sample_imgs[sam_i].cpu().detach())
                all_samples_ids.append(targets[sam_i]['index'])
                if self.args.mask:
                    all_masks.append(masks[sam_i].cpu().detach())
        return all_preds, all_targets, all_samples_ids, all_samples, all_masks, elapsed_time
