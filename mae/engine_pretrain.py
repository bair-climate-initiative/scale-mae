# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import util.lr_sched as lr_sched
import util.misc as misc
from wandb_log import wandb_dump_input_output, wandb_log_metadata


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    scheduler=None,
    source_size_scheduler=None,
    fix_resolution_scheduler=None,
):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, ((samples, res, targets, target_res), metadata) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            target_size = scheduler.get_target_size(epoch)
            source_size = source_size_scheduler.get_target_size(epoch)[0]
            fix_decoding_size = fix_resolution_scheduler.get_target_size(epoch)
            model.module.set_target_size(target_size)
            model.module.set_fix_decoding_size(fix_decoding_size)
            loss, y, mask, mean, var, pos_emb, pos_emb_decoder, samples = model(
                samples,
                input_res=res,
                targets=targets,
                target_res=target_res,
                mask_ratio=args.mask_ratio,
                source_size=source_size,
            )

        if data_iter_step % print_freq == 0:
            y = [
                model.module.unpatchify(y_i)[0].permute(1, 2, 0).detach().cpu()
                for y_i in y
            ]
            x = torch.einsum("nchw->nhwc", samples[:1]).detach().cpu()
            wandb_dump_input_output(
                x[0],
                y,
                epoch,
                f"target-size:{target_size}-output_size:{fix_decoding_size}",
            )
            if metadata:
                wandb_log_metadata(metadata)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
