# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .swin_detr import SwinDETR, SetCriterion
from .vessel_matcher import build_matcher
from .transformer import build_transformer


def build_model(args):
    num_classes = 4
    class_dict = {'end': 0,
                  'intermediate': 1,
                  'bifurcation': 2,
                  'background': 3}
    args.class_dict = class_dict

    device = torch.device(args.device)
    transformer = build_transformer(args)
    matcher = build_matcher(args)

    detr_kwargs = {
        'transformer': transformer,
        'num_classes': num_classes,
        'num_queries': args.num_queries,
        'num_prev_pos': args.num_prev_pos,
        'num_dec_layers': args.dec_layers,
        'class_dict': args.class_dict,
        'num_bifur_queries': args.num_bifur_queries,
    }

    model = SwinDETR(**detr_kwargs)

    weight_dict = {'loss_class': args.cls_loss_coef,
                   'loss_direction': args.dir_loss_coef,
                   'loss_radius': args.rad_loss_coef, }

    aux_weight_dict = {}
    for i in range(args.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    losses = ['labels', 'direction', 'radius', 'cardinality']

    criterion = SetCriterion(
        num_classes - 1,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        class_dict=args.class_dict,
        custom_focal_weights_value=args.custom_focal_weights_value,
        num_bifur_queries=args.num_bifur_queries,)

    model.to(device)
    criterion.to(device)

    return model, criterion
