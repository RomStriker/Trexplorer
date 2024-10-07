import os
import pickle
import glob
import sys
import torch
from argparse import Namespace
import monai
import numpy as np
sys.setrecursionlimit(10000)

from monai.data import CacheDataset, SmartCacheDataset, partition_dataset, Dataset
from src.trexplorer.util.misc import is_main_process, get_world_size, get_rank, init_distributed_mode, get_sha
from monai.data import ThreadDataLoader, DataLoader
from src.trexplorer.datasets.transforms import (LoadAnnotPickled,
                                                CropAndPadd,
                                                ExtRandomSubTreed,
                                                ConvertTreeToTargetsd,
                                                LoadImageCropsAndTreesd)

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    ScaleIntensityd,)


def load_datalist(dataset_dir, data_key):
    annots_type = "annots_cont_"
    if data_key == "training":
        annots_dir = os.path.join(dataset_dir, annots_type + 'train')
        images_dir = os.path.join(dataset_dir, 'images_train')
        masks_dir = os.path.join(dataset_dir, 'masks_train')
    elif data_key == "validation":
        annots_dir = os.path.join(dataset_dir, annots_type + 'val')
        images_dir = os.path.join(dataset_dir, 'images_val')
        masks_dir = os.path.join(dataset_dir, 'masks_val')
    elif data_key == "validation_sv":
        with open(os.path.join(dataset_dir, "annots_val_sub_vol.pickle"), "rb") as f:
            data_list = pickle.load(f)
        data_list = [{"label": sample} for sample in data_list]
        return data_list
    else:
        raise NotImplementedError

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    annot_paths = sorted(glob.glob(os.path.join(annots_dir, "*.pickle")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.nii.gz")))

    datalist = []
    for (image, label, mask) in zip(image_paths, annot_paths, mask_paths):
        datalist.append({"image": image, "label": label, "mask": mask})

    return datalist


# Transforms
def build_training_transforms(cfg):
    transforms = [LoadImaged(keys=["image"], image_only=True),
                  EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                  ScaleIntensityd(keys=['image'], minv=-1.0),
                  LoadAnnotPickled(keys=["label"]),]

    if cfg.mask:
        transforms += [LoadImaged(keys=["mask"], image_only=True),
                       EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel")]

    transforms += [ExtRandomSubTreed(["label"], cfg.root_prob, cfg.bifur_prob,
                                        cfg.end_prob, cfg.seq_len, cfg.num_prev_pos),
                   ConvertTreeToTargetsd(["label"], cfg.seq_len, cfg.num_prev_pos,
                                         cfg.sub_vol_size, cfg.class_dict),
                   CropAndPadd(["image"], cfg.sub_vol_size),
                   ToTensord(keys=["image"], track_meta=False)]

    if cfg.mask:
        transforms += [CropAndPadd(["mask"], cfg.sub_vol_size),
                       ToTensord(keys=["mask"], track_meta=False)]

    if is_main_process():
        for i, t in enumerate(transforms):
            print("Training transform {}: {}".format(i, t))

    return monai.transforms.Compose(transforms)


def build_validation_sv_transforms(cfg):
    annots_dir = os.path.join(cfg.data_dir, 'annots_cont_val_sub_vol')
    images_dir = os.path.join(cfg.data_dir, 'images_val_sub_vol')
    masks_dir = os.path.join(cfg.data_dir, 'masks_val_sub_vol')
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    annot_paths = sorted(glob.glob(os.path.join(annots_dir, "*.pickle")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.nii.gz")))
    paths = list(zip(image_paths, annot_paths, mask_paths))
    transforms = [LoadImageCropsAndTreesd(["label"], cfg.seq_len, cfg.num_prev_pos,
                                          cfg.sub_vol_size, cfg.class_dict, cfg.mask, paths)]

    return monai.transforms.Compose(transforms)


def build_validation_transforms(cfg):
    transforms = [LoadImaged(keys=["image"], image_only=True),
                  EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                  ScaleIntensityd(keys=['image'], minv=-1.0),
                  ToTensord(keys=["image"], track_meta=False),
                  LoadAnnotPickled(keys=["label"])]

    if cfg.mask:
        transforms += [LoadImaged(keys=["mask"], image_only=True),
                       EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel"),
                       ToTensord(keys=["mask"], track_meta=False)]

    if is_main_process():
        for i, t in enumerate(transforms):
            print("Training transform {}: {}".format(i, t))

    return monai.transforms.Compose(transforms)


def build_training_datasets_dist(cfg, split, train_transform):
    files = load_datalist(cfg.data_dir, split)
    if is_main_process():
        print(f"Number of files in full {split} dataset: {len(files)}")

    partition = partition_dataset(data=files,
                                  num_partitions=get_world_size(),
                                  shuffle=False,
                                  even_divisible=True)[get_rank()]
    print(f"Number of files in training dataset partition for rank {get_rank()}:{len(partition)}", force=True)

    dataset_train = SmartCacheDataset(
        data=partition,
        transform=train_transform,
        cache_num=cfg.cache_num/get_world_size(),
        replace_rate=cfg.replace_rate,
        num_init_workers=cfg.num_init_workers,
        num_replace_workers=cfg.num_replace_workers,
        copy_cache=False,
    )

    print(f"Number of files in training dataset for rank {get_rank()}:{len(dataset_train)}", force=True)
    return dataset_train


def build_training_datasets(cfg, split, transforms):
    files = load_datalist(cfg.data_dir, split)
    print("Number of files in full training dataset: {}".format(len(files)))

    dataset = SmartCacheDataset(
        data=files,
        transform=transforms,
        cache_num=cfg.cache_num,
        replace_rate=cfg.replace_rate,
        num_init_workers=cfg.num_init_workers,
        num_replace_workers=cfg.num_replace_workers,
        copy_cache=False,
    )

    return dataset


def build_validation_datasets_dist(cfg, split, transforms):
    files = load_datalist(cfg.data_dir, split)
    if is_main_process():
        print(f"Number of files in full {split} dataset: {len(files)}")

    partition = partition_dataset(data=files,
                                  num_partitions=get_world_size(),
                                  shuffle=False,
                                  even_divisible=True)[get_rank()]
    print(f"Number of files in validation dataset partition for rank {get_rank()}:{len(partition)}", force=True)

    dataset_train = CacheDataset(
        data=partition,
        transform=transforms,
        cache_rate=cfg.cache_rate_train,
        num_workers=cfg.n_workers_train,
        copy_cache=False
    )

    print(f"Number of files in validation dataset for rank {get_rank()}:{len(dataset_train)}", force=True)
    return dataset_train


def build_validation_datasets(cfg, split, transforms):
    files = load_datalist(cfg.data_dir, split)
    print(f"Number of files in full validation dataset: {len(files)}")

    dataset = CacheDataset(
        data=files,
        transform=transforms,
        cache_rate=cfg.cache_rate_train,  # 1.0
        num_workers=cfg.n_workers_train,  # 8
        copy_cache=False,
    )

    return dataset


def build_dataset(cfg, split):
    if split == "training":
        transforms = build_training_transforms(cfg)
    elif split == "validation":
        transforms = build_validation_transforms(cfg)
    elif split == "validation_sv":
        transforms = build_validation_sv_transforms(cfg)
    else:
        raise NotImplementedError

    if split == "training":
        build_dataset_fn = build_training_datasets_dist if cfg.distributed else build_training_datasets
        dataset = build_dataset_fn(cfg, split, transforms)
    elif split in ["validation", "validation_sv"]:
        build_dataset_fn = build_validation_datasets_dist if cfg.distributed else build_validation_datasets
        dataset = build_dataset_fn(cfg, split, transforms)

    return dataset


def train_collate_fn(batch):
    images = []
    targets = []
    past_trs = []
    masks = []

    for sample in batch:
        images.append(torch.unsqueeze(sample['image'], 0))

        label = sample['label']

        targets.append(label)
        past_trs.append(torch.unsqueeze(sample['label']['past_tr'], 0))
        if isinstance(sample['mask'], torch.Tensor):
            masks.append(torch.unsqueeze(sample['mask'], 0))

    images_batch = torch.cat(images, dim=0)
    images_batch = images_batch.contiguous()

    if isinstance(sample['mask'], torch.Tensor):
        masks_batch = torch.cat(masks, dim=0)
        masks_batch = masks_batch.type(torch.BoolTensor)
        masks_batch = masks_batch.contiguous()
    else:
        masks_batch = None

    past_trs_batch = torch.cat(past_trs, dim=0)
    past_trs_batch = past_trs_batch.contiguous()

    batch_output = {"image": images_batch, "label": targets,
                    "past_tr": past_trs_batch, "mask": masks_batch}

    return batch_output


def val_collate_fn(batch):
    images = []
    targets = []
    masks = []

    for sample in batch:
        images.append(torch.unsqueeze(sample['image'], 0))
        label = {'seq_tree': sample['label']['branches'],
                 'index': sample['label']['index'],
                 'bifur_ids': sample['label']['bifur_ids']}
        targets.append(label)
        if isinstance(sample['mask'], torch.Tensor):
            masks.append(torch.unsqueeze(sample['mask'], 0))

    images_batch = torch.cat(images, dim=0)
    images_batch = images_batch.contiguous()

    if isinstance(sample['mask'], torch.Tensor):
        masks_batch = torch.cat(masks, dim=0)
        masks_batch = masks_batch.type(torch.BoolTensor)
        masks_batch = masks_batch.contiguous()
    else:
        masks_batch = None

    batch_output = {"image": images_batch, "label": targets, "mask": masks_batch}

    return batch_output


def compute_label_fracs(args):
    """
    Compute the fractions of the different labels in the training dataset
    """
    assert args.dataset == 'training'
    count_bg = 0
    count_end = 0
    count_inter = 0
    count_bifur = 0
    count_all = 0
    bifur_detected = 0
    end_detected = 0
    prev_inter = 0
    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        for i, batch in enumerate(dataloader):
            inputs, labels, past_tr, masks = (batch["image"], batch["label"], batch["past_tr"], batch['mask'])
            for sample in labels:
                for step in range(1, args.seq_len):
                    curr_bifur = len([x for x in sample['labels'][step] if x == args.class_dict['bifurcation']])
                    curr_all = len([x for x in sample['labels'][step] if x != args.class_dict['background']])
                    curr_end = len([x for x in sample['labels'][step] if x == args.class_dict['end']])
                    count_inter += len([x for x in sample['labels'][step] if x == args.class_dict['intermediate']])
                    count_all += curr_all
                    count_end += curr_end
                    count_bifur += curr_bifur

                    if step == 1:
                        count_bg += (args.num_bifur_queries - curr_all)
                    if bifur_detected:
                        count_bg += (args.num_bifur_queries*bifur_detected - (curr_all - prev_inter))
                        bifur_detected = 0
                    if end_detected:
                        count_bg += end_detected
                        end_detected = 0
                    if curr_bifur:
                        bifur_detected = curr_bifur
                        prev_inter = curr_all - curr_bifur - curr_end
                    if curr_end:
                        end_detected = curr_end

    print("Total count all: ", count_all)
    print("Total count end: ", count_end)
    print("Total count inter: ", count_inter)
    print("Total count bifur: ", count_bifur)
    print("Total count bg: ", count_bg)

    counts = np.array([count_end, count_inter, count_bifur, count_bg])
    sum_counts = np.sum(counts)
    norm_counts = counts/sum_counts
    norm_counts_inv = 1 - norm_counts

    print("Counts: ", counts)
    print("Norm counts: ", [f"{val:.10f}" for val in norm_counts])
    print("Norm counts inv: ", [f"{val:.10f}" for val in norm_counts_inv])


def check_samples():
    for i, batch in enumerate(dataloader):
        inputs = batch["image"]
        print("Sample: ", i)
        print("Min: ", torch.min(inputs))
        print("Max: ", torch.max(inputs))
        print("Mean: ", torch.mean(inputs))


if __name__ == '__main__':
    args = {'data_dir': '/data/synthetic',
            'seq_len': 10,
            'sub_vol_size': 64,
            'num_prev_pos': 5,
            'root_prob': 0.2,
            'bifur_prob': 0.5,
            'end_prob': 0.3,
            'enable_dist': True,
            'mask': True,
            'dataset': 'training',  # training, validation, validation_sv
            'distributed': False,
            'dist_url': 'env://',
            'world_size': 1,
            'cache_rate_train': 1.0,
            'cache_num': 32,
            'replace_rate': 0.125,
            'num_init_workers': 4,
            'num_replace_workers': 2,
            'n_workers_train': 1,
            'batch_size': 1,
            'determinism': True,
            'seed': 37,
            'epochs': 500,
            'class_dict':  {'end': 0,
                            'intermediate': 1,
                            'bifurcation': 2,
                            'background': 3},
            'num_bifur_queries': 26,
            'num_queries': 196,
            }

    args = Namespace(**args)
    if args.distributed:
        init_distributed_mode(args)
        print("git:\n  {}\n".format(get_sha()))

    if args.determinism:
        seed = args.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        monai.utils.set_determinism(seed=seed, additional_settings=None)

    dataset = build_dataset(args, args.dataset)

    if args.dataset in ['training', 'validation_sv']:
        collate_func = train_collate_fn
    elif args.dataset == 'validation':
        collate_func = val_collate_fn
    else:
        raise NotImplementedError

    dataloader = ThreadDataLoader(dataset, batch_size=args.batch_size,
                                  collate_fn=collate_func, num_workers=0)

    # check sample values
    check_samples()
