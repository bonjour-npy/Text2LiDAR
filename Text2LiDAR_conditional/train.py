import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path

from models.CLIP.clip import clip
import datasets as ds
import einops
import matplotlib.cm as cm
import torch
import torch._dynamo
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import utils.render
import utils.training
from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.Transdiff import Transdiff
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles

warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True


def train(cfg):
    torch.backends.cudnn.benchmark = True
    project_dir = (
        Path(cfg.output_dir) / cfg.dataset / cfg.lidar_projection
    )  # 拼接 path 定义投影路径

    # =================================================================================
    # Initialize accelerator
    # =================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )
    device = accelerator.device

    # =================================================================================
    # Setup models
    # =================================================================================

    channels = [
        1 if cfg.train_depth else 0,
        1 if cfg.train_reflectance else 0,
    ]  # 两个通道，分别为深度和反射强度

    # 定义降噪网络 denoiser model
    if cfg.model_name == "Transdiff":
        trans = Transdiff(
            in_channels=sum(channels),
            resolution=cfg.resolution,
            base_channels=cfg.model_base_channels,
            temb_channels=cfg.model_temb_channels,
            channel_multiplier=cfg.model_channel_multiplier,
            num_residual_blocks=cfg.model_num_residual_blocks,
            gn_num_groups=cfg.model_gn_num_groups,
            gn_eps=cfg.model_gn_eps,
            attn_num_heads=cfg.model_attn_num_heads,
            coords_embedding=cfg.model_coords_embedding,
            ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model_name}")

    # 两种投影的方式：spherical 和 unfolding
    if "spherical" in cfg.lidar_projection:
        accelerator.print("set HDL-64E linear ray angles")
        trans.coords = get_hdl64e_linear_ray_angles(*cfg.resolution)
    elif "unfolding" in cfg.lidar_projection:
        accelerator.print("set dataset ray angles")
        _coords = torch.load(f"data/{cfg.dataset}/unfolding_angles.pth")
        # 加载角度信息之后进行插值
        trans.coords = F.interpolate(_coords, size=cfg.resolution, mode="nearest-exact")
    else:
        raise ValueError(f"Unknown: {cfg.lidar_projection}")

    if accelerator.is_main_process:
        print(f"number of parameters: {utils.training.count_parameters(trans):,}")

    # 选择离散或连续时间步长的 DDPM（默认为连续的时间步长）
    if cfg.diffusion_timesteps_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            denoiser=trans,
            criterion=cfg.criterion,
            num_training_steps=cfg.diffusion_num_training_steps,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    elif cfg.diffusion_timesteps_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            denoiser=trans,
            criterion=cfg.criterion,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion_timesteps_type}")
    ddpm.train()
    ddpm.to(device)

    clip_model = clip.load("ViT-B/32", device=device)

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.ema_decay,
            update_every=cfg.ema_update_every,
            update_after_step=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)

    lidar_utils = LiDARUtility(
        resolution=cfg.resolution,
        image_format=cfg.image_format,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        ray_angles=ddpm.denoiser.coords,  # 投影之后的坐标信息
    )
    lidar_utils.to(device)

    # =================================================================================
    # Setup optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # 定义数据集
    dataset = ds.load_dataset(
        path=f"data/{cfg.dataset}",
        name=cfg.lidar_projection,
        split=ds.Split.TRAIN,
        num_proc=cfg.num_workers,
    ).with_format("torch")
    # cache_dir = '/project/r2dm-main/datacache/',

    if accelerator.is_main_process:
        print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # 带有 warmup 的余弦学习率
    lr_scheduler = utils.training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.num_steps * cfg.gradient_accumulation_steps,
    )

    # Comment out these codes during debugging
    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

    # =================================================================================
    # Utility
    # =================================================================================

    def preprocess(batch):
        x = []
        # batch 是以字典格式存储的输入数据
        if cfg.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.train_reflectance:
            x += [lidar_utils.convert_depth(batch["reflectance"])]
        x = torch.cat(x, dim=1)  # Equals to torch.cat((depth, reflectance), dim=1)
        x = lidar_utils.normalize(x)  # 从 [0, 1] scale 至 [-1, 1]
        # 重新插值，得到 [batch_size, 2, 32, 1024] 的 tensor
        x = F.interpolate(
            x.to(device),
            size=cfg.resolution,
            mode="nearest-exact",
        )
        text = batch["text"]
        return x, text

    def split_channels(image: torch.Tensor):
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        image = lidar_utils.denormalize(image)  # 从 [-1, 1] scale 至 [0, 1]
        out = dict()
        depth, rflct = split_channels(image)
        if depth.numel() > 0:
            out[f"{tag}/depth"] = utils.render.colorize(depth)
            metric = lidar_utils.revert_depth(depth)
            mask = (metric > lidar_utils.min_depth) & (metric < lidar_utils.max_depth)
            out[f"{tag}/depth/orig"] = utils.render.colorize(
                metric / lidar_utils.max_depth
            )
            xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
            normal = -utils.render.estimate_surface_normal(xyz)
            normal = lidar_utils.denormalize(normal)
            bev = utils.render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
                t=torch.tensor([0, 0, 1.0]).to(xyz),
            )
            out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()
        if rflct.numel() > 0:
            out[f"{tag}/reflectance"] = utils.render.colorize(rflct, cm.plasma)
        if mask.numel() > 0:
            out[f"{tag}/mask"] = utils.render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    # =================================================================================
    # Training loop
    # =================================================================================

    progress_bar = tqdm(
        range(cfg.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    global_step = 0
    while global_step < cfg.num_steps:
        ddpm.train()
        for batch in dataloader:
            x_0, text = preprocess(batch)
            # out = x_0[0:1, 0:1, :, :].cpu().detach().numpy()
            # out = out.squeeze()
            # plt.imshow(out, cmap='jet')
            # plt.savefig('/project/r2dm-main/debugfig/nuscenesdepth_processed_32.png', dpi=300, bbox_inches='tight', pad_inches=0)
            # out = x_0[0:1, 1:2, :, :].cpu().detach().numpy()
            # out = out.squeeze()
            # plt.imshow(out, cmap='jet')
            # plt.savefig('/project/r2dm-main/debugfig/nuscenesreflect_processed_32.png', dpi=300, bbox_inches='tight', pad_inches=0)
            # text embedding
            text_emb = clip.tokenize(text).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_emb)  # B, 512
            with accelerator.accumulate(ddpm):
                loss = ddpm(x_0=x_0, text=text_features)
                accelerator.backward(loss)
                # for name, param in ddpm.named_parameters():
                #     if param.grad is None:
                #         print(name)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trans.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()

                if global_step == 1:
                    log_images(x_0, "image", global_step)

                if global_step % cfg.save_image_steps == 0:
                    print(text)
                    ddpm_ema.ema_model.eval()
                    sample = ddpm_ema.ema_model.sample(
                        batch_size=cfg.batch_size_eval,
                        num_steps=cfg.diffusion_num_sampling_steps,
                        rng=torch.Generator(device=device).manual_seed(0),
                        text=text_features,
                    )
                    log_images(sample, "sample", global_step)

                if global_step % cfg.save_model_steps == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_dir / f"diffusion_{global_step:010d}.pth",
                    )

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.num_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(utils.training.TrainingConfig, dest="cfg")
    cfg: utils.training.TrainingConfig = parser.parse_args().cfg
    train(cfg)
