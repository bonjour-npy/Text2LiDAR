from pathlib import Path

import torch

from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.Transdiff import Transdiff
from utils.lidar import LiDARUtility

from .training import TrainingConfig, count_parameters


def setup_model(
    ckpt,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile_denoiser: bool = False,
):
    """
    在推理阶段（测试）设置并初始化模型

    该函数负责加载预训练的扩散模型和LiDAR工具类,并进行相应的配置。

    Args:
        ckpt: 模型检查点,可以是路径字符串、Path对象或已加载的字典
        device (torch.device | str, optional): 运行设备. 默认为 "cpu"
        ema (bool, optional): 是否使用EMA权重. 默认为 True
        show_info (bool, optional): 是否显示模型信息. 默认为 True
        compile_denoiser (bool, optional): 是否编译去噪器. 默认为 False

    Returns:
        tuple: 包含以下三个元素:
            - diffusion: 配置好的扩散模型
            - lidar_utils: LiDAR工具类实例
            - cfg: 训练配置对象

    功能分析:
        1. 加载模型配置并初始化输入通道
        2. 根据配置构建Transdiff模型架构
        3. 根据时间步类型创建对应的扩散模型
        4. 加载预训练权重并将模型迁移到指定设备
        5. 初始化LiDAR工具类
        6. 可选显示模型信息
    """
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = TrainingConfig(**ckpt["cfg"])

    in_channels = [0, 0]
    if cfg.train_depth:
        in_channels[0] = 1
    if cfg.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)

    if cfg.model_name == "Transdiff":
        trans = Transdiff(
            in_channels=in_channels,
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

    if cfg.diffusion_timesteps_type == "discrete":
        diffusion = DiscreteTimeGaussianDiffusion(
            denoiser=trans,
            criterion=cfg.criterion,
            num_training_steps=cfg.diffusion_num_training_steps,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    elif cfg.diffusion_timesteps_type == "continuous":
        diffusion = ContinuousTimeGaussianDiffusion(
            denoiser=trans,
            criterion=cfg.criterion,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion_timesteps_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]  # 加载 EMA 权重或原始权重
    diffusion.load_state_dict(state_dict)
    diffusion.eval()
    diffusion.to(device)

    if compile_denoiser:
        diffusion.denoiser = torch.compile(diffusion.denoiser)

    lidar_utils = LiDARUtility(
        resolution=cfg.resolution,
        image_format=cfg.image_format,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        ray_angles=diffusion.denoiser.coords,
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {trans.resolution}",
                f"denoiser: {trans.__class__.__name__}",
                f"diffusion: {diffusion.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {count_parameters(diffusion):,}",
            ],
            sep="\n",
        )

    return diffusion, lidar_utils, cfg


def setup_rng(seeds: list[int], device: torch.device | str):
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
