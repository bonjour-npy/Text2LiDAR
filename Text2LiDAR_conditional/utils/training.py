import dataclasses
import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclasses.dataclass
class TrainingConfig:
    dataset: Literal["kitti_raw", "kitti_360", "nuScenes", "nuLiDARText"] = (
        "nuLiDARText"  # 默认数据集是 nuScenes
    )
    image_format: str = "log_depth"
    lidar_projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    train_mask: bool = True
    resolution: tuple[int, int] = (32, 1024)  # (64, 1024) for 64-beam LiDAR
    min_depth = 1.45
    max_depth = 80.0
    batch_size_train: int = 8  # 16
    batch_size_eval: int = 2
    num_workers: int = 16
    num_steps: int = 400_000  # 800_000
    save_image_steps: int = 5_000  # 5_000
    save_model_steps: int = 100_000  # 10_000
    gradient_accumulation_steps: int = 1
    criterion: str = "l2"  # L2 范式损失函数
    # 一些训练参数
    lr: float = 1e-4
    lr_warmup_steps: int = 16_000  # 10_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 20  # 10
    output_dir: str = "logs/diffusion"
    seed: int = 0
    mixed_precision: str = "no"  # "fp16", "no"
    dynamo_backend: str = None  # "inductor", "no", None
    model_name: str = "Transdiff"
    model_base_channels: int = 64
    model_temb_channels: int = 384  # int | None = None
    model_channel_multiplier: tuple[int] | int = (1, 2, 4, 8)
    model_num_residual_blocks: tuple[int] | int = 3
    model_gn_num_groups: int = 32 // 4
    model_gn_eps: float = 1e-6
    model_attn_num_heads: int = 8
    model_coords_embedding: Literal[
        "spherical_harmonics", "polar_coordinates", "fourier_features", None
    ] = "fourier_features"  # 球谐函数、极坐标角度、傅里叶特征
    model_dropout: float = 0.0
    diffusion_num_training_steps: int = 1024
    diffusion_num_sampling_steps: int = 128
    diffusion_objective: Literal["eps", "v", "x_0"] = "eps"
    diffusion_beta_schedule: str = "cosine"  # Cosine Noise Schedule
    diffusion_timesteps_type: Literal["continuous", "discrete"] = (
        "continuous"  # 连续时间步长的扩散模型
    )


# 带有 warmup 的余弦学习率
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        # 首先定义 warmup 阶段的学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 在 warmup 阶段外计算训练进度
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # 根据 cosine 函数计算学习率
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
