"""
Training script for the lensless reconstruction pipeline using Mamba as the refinement network.
The pipeline remains: FFT-based inversion -> unpixel shuffle -> Mamba -> pixel shuffle.
"""

from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
import os, sys, warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from models.get_model import get_inversion_and_channels
from models.mamba import Mamba
from loss import GLoss
from config_diffusercam import initialise
from metrics import PSNR

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.typing_alias import *

from utils.train_helper import (
    reduce_loss_dict,
    get_optimisers,
    load_models,
    save_weights,
    ExpLoss_with_dict,
    AvgLoss_with_dict,
    pprint_args,
)
from utils.ops import rggb_2_rgb, unpixel_shuffle
from utils.tupperware import tupperware

ex = Experiment("TrainMamba")
ex = initialise(ex)

if "LOCAL_RANK" in os.environ:
    is_local_rank_0 = int(os.environ["LOCAL_RANK"]) == 0
else:
    is_local_rank_0 = True
if not is_local_rank_0:
    sys.stdout = open(os.devnull, "w")

torch.multiprocessing.set_sharing_strategy("file_system")
torch.autograd.set_detect_anomaly(True)

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    dir_init(args, is_local_rank_0=is_local_rank_0)
    if not is_local_rank_0:
        warnings.filterwarnings("ignore")

    if args.distdataparallel:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        rank = args.device
        world_size = 1

    is_admm = "admm" in args.exp_name
    interm_name = "FFT" if not is_admm else "ADMM"

    data = get_dataloaders(args, is_local_rank_0=is_local_rank_0)

    Inversion, in_c = get_inversion_and_channels(args)
    G = Mamba(args, in_c=in_c).to(rank)
    FFT = Inversion(args).to(rank)

    (g_optimizer, fft_optimizer), (g_lr_scheduler, fft_lr_scheduler) = get_optimisers(G, FFT, args)

    (G, FFT), (g_optimizer, fft_optimizer), global_step, start_epoch, loss = load_models(
        G,
        FFT,
        g_optimizer,
        fft_optimizer,
        args,
        is_local_rank_0=is_local_rank_0,
    )

    if args.distdataparallel:
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[rank], output_device=rank)
        FFT = torch.nn.parallel.DistributedDataParallel(FFT, device_ids=[rank], output_device=rank)

    writer = SummaryWriter(log_dir=str(args.run_dir))
    writer.add_text("Args", pprint_args(args))

    if is_local_rank_0:
        world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        logging.info("Using {} GPUs".format(world_size))
        writer.add_text("Args", pprint_args(args))

        train_pbar = tqdm(range(len(data.train_loader) * args.batch_size), dynamic_ncols=True)
        val_pbar = (
            tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)
            if data.val_loader
            else None
        )

    g_loss = GLoss(args).to(rank)

    if not global_step:
        global_step = start_epoch * len(data.train_loader) * args.batch_size

    start_epoch = global_step // len(data.train_loader.dataset)

    loss_dict = {
        "g_loss": 0.0,
        "perception_loss": 0.0,
        "contextual_loss": 0.0,
        "image_loss": 0.0,
        "train_PSNR": 0.0,
    }

    metric_dict = {"PSNR": 0.0, "g_loss": 0.0}
    avg_metrics = AvgLoss_with_dict(loss_dict=metric_dict, args=args)
    exp_loss = ExpLoss_with_dict(loss_dict=loss_dict, args=args)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            G.train()
            FFT.train()

            if is_local_rank_0:
                train_pbar.reset()

            if args.distdataparallel:
                data.train_loader.sampler.set_epoch(epoch)

            for i, batch in enumerate(data.train_loader):
                if ((global_step + 1) % (len(data.train_loader) * args.batch_size) == 0) and (epoch == start_epoch):
                    break

                loss_dict = defaultdict(float)
                source, target, filename = batch
                source, target = (source.to(rank), target.to(rank))

                n, c, h, w = target.shape
                target_unpixel_shuffled = unpixel_shuffle(target, args.pixelshuffle_ratio).reshape(
                    n * args.pixelshuffle_ratio ** 2,
                    c,
                    h // args.pixelshuffle_ratio,
                    w // args.pixelshuffle_ratio,
                )

                G.zero_grad()
                FFT.zero_grad()

                fft_output = FFT(source)

                if is_admm:
                    fft_output = F.interpolate(fft_output, scale_factor=4, mode="nearest")

                fft_unpixel_shuffled = unpixel_shuffle(fft_output, args.pixelshuffle_ratio)
                output_unpixel_shuffled = G(fft_unpixel_shuffled)

                n, c, h, w = output_unpixel_shuffled.shape
                output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                    n * args.pixelshuffle_ratio ** 2,
                    c // args.pixelshuffle_ratio ** 2,
                    h,
                    w,
                )

                g_loss(output=output_unpixel_shuffled, target=target_unpixel_shuffled)
                g_loss.total_loss.backward()

                n, c, h, w = output_unpixel_shuffled.shape
                output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                    n // args.pixelshuffle_ratio ** 2,
                    c * args.pixelshuffle_ratio ** 2,
                    h,
                    w,
                )
                output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)
                loss_dict["train_PSNR"] += PSNR(output, target)

                g_optimizer.step()
                g_lr_scheduler.step(epoch + i / len(data.train_loader))

                if epoch >= args.fft_epochs:
                    fft_optimizer.step()
                    fft_lr_scheduler.step(epoch - args.fft_epochs + i / len(data.train_loader))

                loss_dict["g_loss"] += g_loss.total_loss
                loss_dict["perception_loss"] += g_loss.perception_loss
                loss_dict["contextual_loss"] += g_loss.contextual_loss
                loss_dict["image_loss"] += g_loss.image_loss

                exp_loss += reduce_loss_dict(loss_dict, world_size=1)
                global_step += args.batch_size * world_size

                if is_local_rank_0:
                    train_pbar.update(args.batch_size)
                    train_pbar.set_description(
                        f"Epoch: {epoch + 1} | Gen loss: {exp_loss.loss_dict['g_loss']:.3f} "
                    )

                    if i % args.log_interval == 0:
                        gen_lr = g_optimizer.param_groups[0]["lr"]
                        writer.add_scalar("lr/gen", gen_lr, global_step)
                        if epoch >= args.fft_epochs:
                            fft_lr = fft_optimizer.param_groups[0]["lr"]
                            writer.add_scalar(f"lr/{interm_name.lower()}", fft_lr, global_step)

                        for metric in exp_loss.loss_dict:
                            writer.add_scalar(
                                f"Train_Metrics/{metric.replace('fft', interm_name.lower())}",
                                exp_loss.loss_dict[metric],
                                global_step,
                            )

                        n_vis = np.min([3, args.batch_size])
                        for e in range(n_vis):
                            if not is_admm:
                                source_vis = rggb_2_rgb(source[e]).mul(0.5).add(0.5)
                                fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                            else:
                                source_vis = source[e].mul(0.5).add(0.5)
                                fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                            fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (fft_output_vis.max() - fft_output_vis.min())
                            target_vis = target[e].mul(0.5).add(0.5)
                            output_vis = output[e].mul(0.5).add(0.5)

                            writer.add_image(f"Source/Train_{e + 1}", source_vis.cpu().detach(), global_step)
                            writer.add_image(f"{interm_name}/Train_{e + 1}", fft_output_vis.cpu().detach(), global_step)
                            writer.add_image(f"Target/Train_{e + 1}", target_vis.cpu().detach(), global_step)
                            writer.add_image(f"Output/Train_{e + 1}", output_vis.cpu().detach(), global_step)
                            writer.add_text(f"Filename/Train_{e + 1}", filename[e], global_step)

                if is_local_rank_0 and (i % args.save_ckpt_interval == 0):
                    logging.info(f"Saving weights at epoch {epoch + 1} global step {global_step}")
                    save_weights(
                        epoch=epoch,
                        global_step=global_step,
                        G=G,
                        FFT=FFT,
                        g_optimizer=g_optimizer,
                        fft_optimizer=fft_optimizer,
                        loss=loss,
                        tag="latest",
                        args=args,
                        is_local_rank_0=True,
                    )

            with torch.no_grad():
                G.eval()
                FFT.eval()
                filename_static = []
                avg_metrics.reset()

                if is_local_rank_0:
                    val_pbar.reset()

                for i, batch in enumerate(data.val_loader):
                    metrics_dict = defaultdict(float)
                    source, target, filename = batch
                    source, target = (source.to(rank), target.to(rank))

                    n, c, h, w = target.shape
                    target_unpixel_shuffled = unpixel_shuffle(target, args.pixelshuffle_ratio).reshape(
                        n * args.pixelshuffle_ratio ** 2,
                        c,
                        h // args.pixelshuffle_ratio,
                        w // args.pixelshuffle_ratio,
                    )

                    fft_output = FFT(source)
                    if is_admm:
                        fft_output = F.interpolate(fft_output, scale_factor=4, mode="nearest")

                    fft_unpixel_shuffled = unpixel_shuffle(fft_output, args.pixelshuffle_ratio)
                    output_unpixel_shuffled = G(fft_unpixel_shuffled)

                    n, c, h, w = output_unpixel_shuffled.shape
                    output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                        n * args.pixelshuffle_ratio ** 2,
                        c // args.pixelshuffle_ratio ** 2,
                        h,
                        w,
                    )
                    output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)

                    g_loss(output=output_unpixel_shuffled, target=target_unpixel_shuffled)
                    metrics_dict["g_loss"] += g_loss.total_loss

                    if args.static_val_image in filename:
                        filename_static = filename
                        source_static = source
                        fft_output_static = fft_output
                        target_static = target
                        output_static = output

                    metrics_dict["PSNR"] += PSNR(output, target)
                    avg_metrics += reduce_loss_dict(metrics_dict, world_size=world_size)

                    if is_local_rank_0:
                        val_pbar.update(args.batch_size)
                        val_pbar.set_description(
                            f"Val Epoch : {epoch + 1} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f}| Total Loss: {avg_metrics.loss_dict['g_loss']:.3f}"
                        )

                if is_local_rank_0:
                    for metric in avg_metrics.loss_dict:
                        writer.add_scalar(f"Val_Metrics/{metric}", avg_metrics.loss_dict[metric], global_step)

                    n_vis = np.min([3, args.batch_size])
                    for e in range(n_vis):
                        if not is_admm:
                            source_vis = rggb_2_rgb(source[e]).mul(0.5).add(0.5)
                            fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                        else:
                            source_vis = source[e].mul(0.5).add(0.5)
                            fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                        fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (fft_output_vis.max() - fft_output_vis.min())
                        target_vis = target[e].mul(0.5).add(0.5)
                        output_vis = output[e].mul(0.5).add(0.5)

                        writer.add_image(f"{interm_name}/Val_{e+1}", fft_output_vis.cpu().detach(), global_step)
                        writer.add_image(f"Source/Val_{e+1}", source_vis.cpu().detach(), global_step)
                        writer.add_image(f"Target/Val_{e+1}", target_vis.cpu().detach(), global_step)
                        writer.add_image(f"Output/Val_{e+1}", output_vis.cpu().detach(), global_step)
                        writer.add_text(f"Filename/Val_{e + 1}", filename[e], global_step)

                    for e, filename in enumerate(filename_static):
                        if filename == args.static_val_image:
                            if not is_admm:
                                source_vis = rggb_2_rgb(source_static[e]).mul(0.5).add(0.5)
                                fft_output_vis = rggb_2_rgb(fft_output_static[e]).mul(0.5).add(0.5)
                            else:
                                source_vis = source_static[e].mul(0.5).add(0.5)
                                fft_output_vis = fft_output_static[e].mul(0.5).add(0.5)

                            fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (fft_output_vis.max() - fft_output_vis.min())
                            target_vis = target_static[e].mul(0.5).add(0.5)
                            output_vis = output_static[e].mul(0.5).add(0.5)

                            writer.add_image(f"{interm_name}/Val_Static", fft_output_vis.cpu().detach(), global_step)
                            writer.add_image(f"Source/Val_Static", source_vis.cpu().detach(), global_step)
                            writer.add_image(f"Target/Val_Static", target_vis.cpu().detach(), global_step)
                            writer.add_image(f"Output/Val_Static", output_vis.cpu().detach(), global_step)
                            writer.add_text(f"Filename/Val_Static", filename[e], global_step)
                            break

                    logging.info(
                        f"Saving weights at END OF epoch {epoch + 1} global step {global_step}, the PSNR is {avg_metrics.loss_dict['PSNR']:.3f}"
                    )

                    if avg_metrics.loss_dict["g_loss"] < loss:
                        is_min = True
                        loss = avg_metrics.loss_dict["g_loss"]
                    else:
                        is_min = False

                    save_weights(
                        epoch=epoch,
                        global_step=global_step,
                        G=G,
                        FFT=FFT,
                        g_optimizer=g_optimizer,
                        fft_optimizer=fft_optimizer,
                        loss=loss,
                        is_min=is_min,
                        args=args,
                        tag="best",
                    )

    except KeyboardInterrupt:
        if is_local_rank_0:
            logging.info("-" * 89)
            logging.info("Exiting from training early. Saving models")
            save_weights(
                epoch=epoch,
                global_step=global_step,
                G=G,
                FFT=FFT,
                g_optimizer=g_optimizer,
                fft_optimizer=fft_optimizer,
                loss=loss,
                is_min=True,
                args=args,
            )
