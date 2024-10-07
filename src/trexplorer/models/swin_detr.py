# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import copy
import warnings

from ..util.misc import (accuracy, get_world_size,
                         is_dist_avail_and_initialized,
                         sigmoid_focal_loss)


class SwinDETR(nn.Module):
    """ This is the SwinDETR module that performs centerline detection. """

    def __init__(self, transformer, num_classes=1, num_queries=10, num_prev_pos=1,
                 num_dec_layers=6, class_dict=None, num_bifur_queries=25):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture with SwinUNETR encoder. See transformer.py
            num_classes: number of object classes,
            num_queries: number of object queries. This is max number of branches that can be detected.
            num_prev_pos: number of previous positions to be used in the past trajectory embedding.
            num_dec_layers: number of decoder layers in the transformer.
            class_dict: dictionary containing the class labels.
            num_bifur_queries: number object queries allocated to a bifurcation point.
        """
        super().__init__()

        self.num_queries = num_queries
        self.num_max_queries = num_queries
        self.transformer = transformer
        self.class_dict = class_dict
        self.num_bifur_queries = num_bifur_queries

        _class_embed = nn.Linear(self.hidden_dim, num_classes + 1)  # branch or not branch
        class_embed_layerlist = [_class_embed for i in range(num_dec_layers)]
        self.class_embed = nn.ModuleList(class_embed_layerlist)

        _radius_embed = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        _direction_embed = MLP(self.hidden_dim, self.hidden_dim, 3, 3)
        share_direc_embed_num = 5
        radius_embed_layerlist = [_radius_embed for i in range(share_direc_embed_num)]
        direction_embed_layerlist = [_direction_embed for i in range(share_direc_embed_num)]
        num_layers_diff = num_dec_layers - share_direc_embed_num
        if num_layers_diff > 0:
            radius_embed_layerlist = [copy.deepcopy(_radius_embed) for i in range(num_layers_diff)] + radius_embed_layerlist
            direction_embed_layerlist = [copy.deepcopy(_direction_embed) for i in range(num_layers_diff)] + direction_embed_layerlist
        self.radius_embed = nn.ModuleList(radius_embed_layerlist)
        self.direction_embed = nn.ModuleList(direction_embed_layerlist)

        self.prev_pos_embed = MLP(num_prev_pos * 4, self.hidden_dim, self.hidden_dim, 3)
        self.step_embed = nn.Linear(1, self.hidden_dim)
        self.num_extra_tokens = 2

        self.bifur_offset_embed = nn.Embedding(num_bifur_queries, self.hidden_dim)
        self.query_embed = nn.Embedding(1, self.hidden_dim)
        self.extra_token_embed = nn.Embedding(self.num_extra_tokens, self.hidden_dim)
        self.query_embed_start = nn.Embedding(num_bifur_queries, self.hidden_dim)
        self.filter_background = torch.tensor([class_dict['background']])

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @staticmethod
    def find_element_index(element, my_list):
        try:
            index = my_list.index(element)
            return index
        except ValueError:
            return -1

    @staticmethod
    def safe_slice(lst, start, stop, eval=False):
        if start > len(lst) or stop > len(lst):
            if eval:
                warnings.warn("Slice indices are out of range!")
            else:
                raise IndexError("Slice indices are out of range!")
        return lst[start:stop]

    def create_bifur_list_from_targets(self, indices, targets, prev_targets):
        bifur_list = None
        prev_num_targets = sum([len(x['labels']) for x in prev_targets])
        num_targets = sum([len(x['labels']) for x in targets])
        if num_targets != prev_num_targets:
            bifur_list = []
            for sample_i, sample in enumerate(targets):
                bifur_idxs = {}
                if len(sample['labels']) == len(prev_targets[sample_i]['labels']):
                    bifur_list.append(bifur_idxs)
                    continue
                unique_idxs, idxs_count = torch.unique(sample['parent_idxs'], return_counts=True)
                used_queries = indices[sample_i][0].tolist()
                for count_i, count in enumerate(idxs_count):
                    if count > 1:
                        free_query_idxs = [i for i in range(self.num_queries) if i not in used_queries]
                        new_target_idxs = ((sample['parent_idxs'] == unique_idxs[count_i]).nonzero(as_tuple=True)[0][1:]).tolist()
                        target_arg_in_indices = (indices[sample_i][1] == unique_idxs[count_i].cpu()).nonzero(as_tuple=True)[0].item()
                        corres_query_idx = indices[sample_i][0][target_arg_in_indices].item()
                        curr_group_queries = self.safe_slice(free_query_idxs, 0, self.num_bifur_queries)
                        bifur_idxs[unique_idxs[count_i].item()] = [corres_query_idx, curr_group_queries, new_target_idxs]
                        used_queries += curr_group_queries
                bifur_list.append(bifur_idxs)
        return bifur_list

    def create_bifur_list_from_pred(self, prev_step_info):
        bifur_list = None
        query_classes = prev_step_info['out']['class_logits']
        query_classes = torch.argmax(query_classes, dim=2)
        bifur_queries = query_classes == self.class_dict['bifurcation']
        if bifur_queries.any():
            batch_size = query_classes.size(0)
            bifur_list = []
            for sample_i in range(batch_size):
                sample_bifur_queries_idxs = torch.nonzero(bifur_queries[sample_i], as_tuple=True)[0].tolist()
                bifur_idxs = {}
                indices = prev_step_info['old_indices']
                mapped_indices = prev_step_info['indices']
                if len(indices) and len(indices[sample_i]):
                    indices_queries = indices[sample_i][0].tolist()
                    mapped_indices_queries = mapped_indices[sample_i][0].tolist()
                    old_to_new_idxs_dict = {indices_queries[i]: mapped_indices_queries[i] for i in range(len(indices_queries))}

                    used_queries = copy.deepcopy(mapped_indices_queries)
                    for bifur_query_id in sample_bifur_queries_idxs:
                        free_query_idxs = [i for i in range(self.num_queries) if i not in used_queries]
                        curr_group_queries = self.safe_slice(free_query_idxs, 0, self.num_bifur_queries, eval=True)
                        bifur_query_idx_in_indices = self.find_element_index(bifur_query_id, indices_queries)

                        if bifur_query_idx_in_indices == -1:
                            continue
                        else:
                            target_id = mapped_indices[sample_i][1][bifur_query_idx_in_indices].item()

                        mapped_bifur_query_id = old_to_new_idxs_dict[bifur_query_id]
                        bifur_idxs[target_id] = [mapped_bifur_query_id, curr_group_queries, None]
                        used_queries += curr_group_queries
                bifur_list.append(bifur_idxs)

        return bifur_list

    def get_mapped_indices(self, indices):
        """
        Map indices to the beginning empty object queries.
        """
        if len(indices) and sum(len(lst) for lst in indices):
            mapped_indices = []
            for i, sample_indices in enumerate(indices):
                if len(sample_indices) and len(sample_indices[0]):
                    mapped_queries = torch.arange(len(sample_indices[0]), dtype=torch.int64)
                    mapped_indices.append([mapped_queries, sample_indices[1]])
                else:
                    mapped_indices.append(sample_indices)
            return mapped_indices
        else:
            return copy.deepcopy(indices)

    def forward(self, samples, past_trs, prev_step_info, step, targets=None, prev_targets=None):
        batch_size = samples.shape[0]
        memory = prev_step_info['memory'] if 'memory' in prev_step_info else None
        prev_hs = prev_step_info['hs_without_norm'] if 'hs_without_norm' in prev_step_info else None
        pos = prev_step_info['pos'] if 'pos' in prev_step_info else None
        indices = prev_step_info['indices']

        if len(indices) and sum(len(lst) for lst in indices):
            for i, sample_indices in enumerate(indices):
                if len(sample_indices):
                    sample_indices[0] = sample_indices[0].cpu()
                    sample_indices[1] = sample_indices[1].cpu()

        mapped_indices = self.get_mapped_indices(indices)

        prev_step_info['old_indices'] = indices
        prev_step_info['indices'] = mapped_indices

        # query_embed is the positional encoding of the object queries
        query_embed = torch.zeros([self.num_queries + self.num_extra_tokens, self.hidden_dim]).to(samples.device)
        query_embed[:self.num_queries, :] = self.query_embed.weight
        query_embed[self.num_queries:, :] = self.extra_token_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        # Copy the previous hidden states to the new queries
        if not prev_step_info['first_step']:
            queries_mask = torch.ones((batch_size, self.num_queries + self.num_extra_tokens),
                                      dtype=torch.bool).to(prev_hs.device)
            queries_mask[:, self.num_queries:] = False

            if targets is None:
                bifur_list = self.create_bifur_list_from_pred(prev_step_info)
            else:
                bifur_list = self.create_bifur_list_from_targets(mapped_indices, targets, prev_targets)

            zeros_hs = torch.zeros_like(query_embed)
            prev_hs = prev_hs.transpose(0, 1)
            for i, sample_indices in enumerate(indices):
                if len(sample_indices):
                    zeros_hs[mapped_indices[i][0], i, :] = prev_hs[sample_indices[0], i, :]
                    queries_mask[i, mapped_indices[i][0]] = False

            if bifur_list is not None:
                for i, sample_bifur_list in enumerate(bifur_list):
                    for bifur_group in sample_bifur_list:
                        old_query_idxs = indices[i][0].tolist()
                        new_query_idxs = mapped_indices[i][0].tolist()
                        new_to_old_idxs_dict = {new_query_idxs[i]: old_query_idxs[i] for i in range(len(old_query_idxs))}
                        old_query_index = new_to_old_idxs_dict[sample_bifur_list[bifur_group][0]]
                        zeros_hs[sample_bifur_list[bifur_group][1], i, :] = prev_hs[old_query_index, i, :]
                        queries_mask[i, sample_bifur_list[bifur_group][1]] = False

                        selected_indices = sample_bifur_list[bifur_group][1]
                        if len(selected_indices) < self.num_bifur_queries:
                            warnings.warn("Insufficient object queries available, required: {:d}, "
                                          "available: {:d}".format(self.num_bifur_queries, len(selected_indices)))
                        query_embed[selected_indices, i, :] = query_embed[selected_indices, i, :] + self.bifur_offset_embed.weight[:len(selected_indices)]
            prev_hs = zeros_hs
        else:
            bifur_list = None
            prev_hs = torch.zeros_like(query_embed)
            query_embed[:self.num_bifur_queries, :, :] += self.query_embed_start.weight.unsqueeze(1)
            used_indices = torch.cat([torch.arange(self.num_bifur_queries),
                                      torch.arange(self.num_queries, self.num_queries + self.num_extra_tokens)])
            queries_mask = torch.ones((batch_size, self.num_queries + self.num_extra_tokens),
                                      dtype=torch.bool).to(prev_hs.device)
            queries_mask[:, used_indices] = False

        past_trs_embed = self.prev_pos_embed(past_trs).unsqueeze(0)
        prev_hs[self.num_queries] = past_trs_embed

        step_embedding = self.step_embed(step)
        step_embedding = step_embedding.unsqueeze(0)
        step_embedding = step_embedding.expand(-1, batch_size, -1)
        prev_hs[-1] = step_embedding

        used_queries = torch.nonzero(~queries_mask[:, :self.num_queries], as_tuple=True)
        max_used_queries = 0
        for i in range(batch_size):
            sample_used_queries_idx = used_queries[0] == i
            sample_used_queries = used_queries[1][sample_used_queries_idx]
            assert (sample_used_queries == torch.arange(len(sample_used_queries)).to(sample_used_queries.device)).all()
            if len(sample_used_queries) > max_used_queries:
                max_used_queries = len(sample_used_queries)

        prev_hs = torch.cat([prev_hs[:max_used_queries],
                             prev_hs[-self.num_extra_tokens:]], dim=0).contiguous()
        query_embed = torch.cat([query_embed[:max_used_queries],
                                 query_embed[-self.num_extra_tokens:]], dim=0).contiguous()
        queries_mask = torch.cat([queries_mask[:, :max_used_queries],
                                 queries_mask[:, -self.num_extra_tokens:]], dim=1).contiguous()

        hs, hs_without_norm, memory, pos = self.transformer(prev_hs, query_embed,
                                                            pos, src=samples, memory=memory, tgt_mask=queries_mask)
        num_decoders, _, _, _ = hs.size()

        hs = hs[:, :, :max_used_queries, :]
        hs_without_norm = hs_without_norm[:, :, :max_used_queries, :]

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        outputs_direc = torch.stack([layer_direc_embed(layer_hs) for layer_direc_embed, layer_hs in zip(self.direction_embed, hs)])
        outputs_radius = torch.stack([layer_radius_embed(layer_hs) for layer_radius_embed, layer_hs in zip(self.radius_embed, hs)])
        outputs_radius = outputs_radius.sigmoid()
        outputs_direc = outputs_direc.tanh()

        out = {'class_logits': outputs_class[-1],
               'direc_logits': outputs_direc[-1],
               'radius_logits': outputs_radius[-1]}

        out.update({'query_mask': queries_mask[:, :max_used_queries]})
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_direc, outputs_radius)

        prev_step_info['bifur_list'] = bifur_list
        prev_step_info['hs_without_norm'] = hs_without_norm.detach()[-1]

        if prev_step_info['first_step']:
            prev_step_info['memory'] = memory
            prev_step_info['pos'] = pos
            prev_step_info['first_step'] = False

        prev_step_info['out'] = out

        return out, prev_step_info

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_direc, outputs_radius):
        aux_dict = [{'class_logits': a, 'direc_logits': b, 'radius_logits': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_direc[:-1], outputs_radius[:-1])]
        return aux_dict


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, position and radius)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, class_dict,
                 custom_focal_weights_value, num_bifur_queries=16):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.class_dict = class_dict
        self.num_bifur_queries = num_bifur_queries
        self.focal_weights = torch.tensor(custom_focal_weights_value)[None, None, :].to('cuda')

    def loss_labels_focal(self, outputs, targets, indices, num_vessels, log=True):
        assert 'class_logits' in outputs
        src_logits = outputs['class_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        query_mask = ~outputs['query_mask'] if 'query_mask' in outputs else None
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, query_mask, self.focal_weights)

        losses = {'loss_class': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_vessels):
        pred_logits = outputs['class_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)

        query_mask = outputs['query_mask'] if 'query_mask' in outputs else None
        card_pred = pred_logits.argmax(-1)
        if query_mask is not None:
            card_pred[query_mask] = pred_logits.shape[-1] - 1
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_direction(self, outputs, targets, indices, num_vessels):
        assert 'direc_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_direc = outputs['direc_logits'][idx]
        target_direc = torch.cat([t['directions'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_direc = F.l1_loss(src_direc, target_direc, reduction='none')
        losses = {}
        losses['loss_direction'] = loss_direc.sum() / num_vessels
        return losses

    def loss_radius(self, outputs, targets, indices, num_vessels):
        assert 'radius_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_radii = outputs['radius_logits'][idx]
        target_radii = torch.cat([t['radii'][i] for t, (_, i) in zip(targets, indices)], dim=0).unsqueeze(1)

        loss_radii = F.l1_loss(src_radii, target_radii, reduction='none')
        losses = {}
        losses['loss_radius'] = loss_radii.sum() / num_vessels
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_vessels, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal,
            'cardinality': self.loss_cardinality,
            'direction': self.loss_direction,
            'radius': self.loss_radius
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_vessels, **kwargs)

    def filter_finished_indices(self, indices, targets):
        """
        Filter out the indices of the finished queries.
        """
        target_classes_o = [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        background_mask = [(t == self.class_dict['background']).cpu() for t in target_classes_o]
        if torch.cat(background_mask).any():
            filtered_indices = []
            for sample_i, sample in enumerate(indices):
                filtered_indices.append([])
                filtered_indices[sample_i] = [indices[sample_i][0][~background_mask[sample_i]],
                                              indices[sample_i][1][~background_mask[sample_i]]]
            return filtered_indices
        else:
            return indices

    def forward(self, outputs, targets, prev_targets, prev_step_info):
        """
        This performs the loss computation
        """
        prev_indices = prev_step_info['indices']
        bifur_list = prev_step_info['bifur_list']

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets, prev_targets, prev_indices, bifur_list)
        filtered_indices = self.filter_finished_indices(indices, targets)

        num_vessels = sum([len(v[1]) for v in filtered_indices])
        num_vessels = torch.as_tensor([num_vessels], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_vessels)
            num_vessels = torch.clamp(num_vessels / get_world_size(), min=1).item()
        else:
            num_vessels = torch.clamp(num_vessels, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, filtered_indices, num_vessels))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    if 'query_mask' in outputs:
                        aux_outputs['query_mask'] = outputs['query_mask']
                    l_dict = self.get_loss(loss, aux_outputs, targets, filtered_indices, num_vessels, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x
