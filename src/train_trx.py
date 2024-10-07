# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pickle
import random
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
import sacred
import torch
import yaml
import monai
from monai.data import ThreadDataLoader, set_track_meta
from trexplorer.engine_trx import Trexplorer
import trexplorer.util.eval_utils as eval_utils
import trexplorer.util.misc as utils
from trexplorer.models import build_model
from trexplorer.util.misc import nested_dict_to_namespace, restore_config
from trexplorer.datasets.cache_dataset import build_dataset, train_collate_fn, val_collate_fn
from trexplorer.util.eval_utils import get_score_nx

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

ex = sacred.Experiment('train', save_git_info=False)
ex.add_config('./cfgs/train.yaml')
ex.add_named_config('eval', './cfgs/eval.yaml')


def train(args: Namespace) -> None:
    if args.resume:
        args = restore_config(args)

    logger = utils.setup_logger(args)

    if args.distributed:
        utils.init_distributed_mode(args)
        logger.info(f"git:\n  {utils.get_sha()}\n")

    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(
            vars(args),
            open(output_dir / 'config.yaml', 'w'), allow_unicode=True)

    device = torch.device(args.device)

    if args.determinism:
        seed = args.seed + utils.get_rank()
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        monai.utils.set_determinism(seed=seed, additional_settings=None)

    model, criterion = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info(f'Num trainable model params: {n_parameters}')

    param_dicts = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad], "lr": args.lr}
    ]

    if not len(args.test_sample):
        set_track_meta(False)
        if not args.eval_only:
            dataset_train = build_dataset(args, 'training')
            data_loader_train = ThreadDataLoader(dataset_train, batch_size=args.batch_size,
                                                 collate_fn=train_collate_fn, num_workers=0, shuffle=True)
        if args.volume_eval:
            dataset_val = build_dataset(args, 'validation')
            data_loader_val = ThreadDataLoader(dataset_val, batch_size=args.batch_size_val,
                                               collate_fn=val_collate_fn, num_workers=0)
        if args.sub_volume_eval:
            dataset_val_sv = build_dataset(args, 'validation_sv')
            data_loader_val_sv = ThreadDataLoader(dataset_val_sv, batch_size=args.batch_size,
                                                  collate_fn=train_collate_fn, num_workers=0)

    if not args.eval_only:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = utils.get_lr_scheduler(args, optimizer, len(data_loader_train))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']
            epoch_offset = 0
            best_f1_nd_dist = checkpoint['best_f1_nd_dist'] if 'best_f1_nd_dist' in checkpoint else 0
            new_best_nd = False
    else:
        epoch_offset = 0
        start_epoch = 0
        best_f1_nd_dist = 0
        new_best_nd = True

    torch.set_float32_matmul_precision('high')
    engine_trx = Trexplorer(args, logger, device)

    if args.eval_only:
        if args.distributed:
            torch.distributed.barrier()
        resume_dir = Path(args.resume).parent

        if len(args.test_sample):
            sav_dir = Path(args.resume).parent / 'test_result'
            sav_dir.mkdir(parents=True, exist_ok=True)
            set_track_meta(False)
            args.batch_size_per_sample = args.batch_size
            logger.info(f'batch_size_per_sample: {args.batch_size_per_sample}')
            test_sample_list = args.test_sample

            for sample in test_sample_list:
                preds, targets, sample_ids, samples, masks, elapsed_time = engine_trx.evaluate_sinsam(model,
                                                                                                      criterion,
                                                                                                      sample)
                stats_reduced = get_score_nx(preds, targets, elapsed_time, dist=args.distributed)
                # Save results
                pred_dict = {'preds': preds, 'targets': targets, 'elapsed_time': elapsed_time,
                             'stats_reduced': stats_reduced, 'samples': samples, 'masks': masks}

                with open(sav_dir / (sample + '.pkl'), 'wb') as fp:
                    pickle.dump(pred_dict, fp)

                message = eval_utils.get_stats_message(stats_reduced)
                logger.info(message)
            return

        if args.sub_volume_eval:
            preds, targets, sample_ids, samples, masks, elapsed_time = engine_trx.evaluate_sv(model,
                                                                                              criterion,
                                                                                              data_loader_val_sv)
            stats_reduced_sv = get_score_nx(preds, targets, elapsed_time, args.distributed)

            pred_dict = {'preds': preds, 'targets': targets, 'samples': samples,
                         'masks': masks, 'elapsed_time': elapsed_time, 'stats_reduced': stats_reduced_sv}

            with open(resume_dir / 'pred_dict_sv.pkl', 'wb') as fp:
                pickle.dump(pred_dict, fp)

            message = eval_utils.get_stats_message(stats_reduced_sv)
            logger.info(message)

        if args.volume_eval:
            args.batch_size_per_sample = args.batch_size
            logger.info(f'batch_size_per_sample: {args.batch_size_per_sample}')

            preds, targets, sample_ids, samples, masks, elapsed_time = engine_trx.evaluate_val(model,
                                                                                               criterion,
                                                                                               data_loader_val)
            stats_reduced = get_score_nx(preds, targets, elapsed_time, args.distributed)

            pred_dict = {'preds': preds, 'targets': targets, 'samples': samples,
                         'masks': masks, 'elapsed_time': elapsed_time, 'stats_reduced': stats_reduced}
            with open(resume_dir / 'pred_dict.pkl', 'wb') as fp:
                pickle.dump(pred_dict, fp)

            message = eval_utils.get_stats_message(stats_reduced)
            logger.info(message)
        return

    if utils.get_rank() == 0:
        logger.info("Start training")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            torch.distributed.barrier()

        if not (args.resume and epoch == start_epoch):
            st = time.time()
            metrics = engine_trx.train_one_epoch(model, criterion, data_loader_train,
                                                 optimizer, lr_scheduler, scaler, logger)
            et = time.time()
            elapsed_time = et - st

            if utils.get_rank() == 0:
                logger.info(f'Epoch: {epoch} \t | \t loss: {metrics["scaled_losses"]:.5f} '
                            f'\t\t | \t time taken: {elapsed_time:.5f}')

            if args.distributed:
                torch.distributed.barrier()

            if utils.get_rank() == 0 and args.output_dir and not epoch % args.save_checkpoint:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

                if args.save_model_interval and not epoch % args.save_model_interval:
                    checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch + epoch_offset}.pth")

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'best_f1_nd_dist': best_f1_nd_dist,
                    }, checkpoint_path)

        if args.sub_volume_eval and (not epoch % args.val_interval_sv):
            preds, targets, sample_ids, samples, masks, elapsed_time = engine_trx.evaluate_sv(model,
                                                                                              criterion,
                                                                                              data_loader_val_sv)
            stats_reduced_sv = get_score_nx(preds, targets, elapsed_time, args.distributed)

            if utils.get_rank() == 0:
                logger.info("Sub-volume Eval:")
                stats_message = eval_utils.get_stats_message(stats_reduced_sv)
                logger.info(stats_message)

        if args.volume_eval and (not epoch % args.val_interval):
            if not epoch:
                args.batch_size_per_sample = args.batch_size
                logger.info(f'sub_vol batch_size_per_sample: {args.batch_size_per_sample}')

            preds, targets, sample_ids, samples, masks, elapsed_time = engine_trx.evaluate_val(model,
                                                                                               criterion,
                                                                                               data_loader_val)
            stats_reduced = get_score_nx(preds, targets, elapsed_time, args.distributed)

            if utils.get_rank() == 0:
                stats_message = eval_utils.get_stats_message(stats_reduced)
                logger.info(stats_message)

        if utils.get_rank() == 0 and args.output_dir:
            checkpoint_paths = []
            if args.volume_eval:
                if new_best_nd:
                    checkpoint_paths += [output_dir / f"checkpoint_best_f1_nd.pth"]
                    new_best_nd = False

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_f1_nd_dist': best_f1_nd_dist
                }, checkpoint_path)

    logger.info("Training Finished!")


@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    train(args)
