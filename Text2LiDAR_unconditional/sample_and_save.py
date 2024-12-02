import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import utils.inference

warnings.filterwarnings("ignore", category=UserWarning)


def sample(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # keys_list = list(args.ckpt['state_dict'].keys())
    # for key in keys_list:
    #     if 'orig_mod.' in key:
    #         deal_key = key.replace('_orig_mod.', '')
    #         args.ckpt['state_dict'][deal_key] = args.ckpt['state_dict'][key]
    #         del args.ckpt['state_dict'][key]

    ddpm, lidar_utils, cfg = utils.inference.setup_model(args.ckpt)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        even_batches=False,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"{cfg=}")

    dataloader = DataLoader(
        TensorDataset(torch.arange(args.num_samples).long()),
        batch_size=args.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    ddpm.to(device)
    sample_fn = torch.compile(ddpm.sample)
    lidar_utils, dataloader = accelerator.prepare(lidar_utils, dataloader)

    save_dir = Path(args.output_dir)
    with accelerator.main_process_first():
        save_dir.mkdir(parents=True, exist_ok=True)

    # 后处理，将生成结果转换为 LiDAR 点云
    def postprocess(sample):
        sample = lidar_utils.denormalize(sample)
        depth, rflct = sample[:, [0]], sample[:, [1]]
        depth = lidar_utils.revert_depth(depth)
        xyz = lidar_utils.to_xyz(depth)
        return torch.cat([depth, xyz, rflct], dim=1)

    # 遍历一次 dataloader，取其中的数值作为随机种子生成样本
    for seeds in tqdm(
        dataloader,
        desc="saving...",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ):
        if seeds is None:
            break
        else:
            (seeds,) = seeds

        # with torch.cuda.amp.autocast(None is not None):
        # with torch.cuda.amp.autocast(cfg.mixed_precision is not None):
        #     samples = sample_fn(
        #         batch_size=len(seeds),
        #         num_steps=args.num_steps,
        #         rng=utils.inference.setup_rng(seeds.cpu().tolist(), device=device),
        #         progress=False,
        #     ).clamp(-1, 1)

        samples = ddpm.sample(
            batch_size=len(seeds),  # dataloader 是存放随机种子
            num_steps=args.num_steps,
            rng=utils.inference.setup_rng(seeds.cpu().tolist(), device=device),
            progress=False,
        ).clamp(-1, 1)

        samples = postprocess(samples)

        for i in range(len(samples)):
            sample = samples[i]
            torch.save(sample.clone(), save_dir / f"samples_{seeds[i]:010d}.pth")

        # accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--ckpt", type=str, default='/project/r2dm-transformer-5decoder-dwt/logs/diffusion/kitti_360/spherical-1024/dwt-convpos/models/diffusion_0000300000.pth')
    # parser.add_argument("--output_dir", type=str, default='/project/r2dm-transformer-5decoder-dwt/logs/diffusion/kitti_360/spherical-1024/dwt-convpos/results')
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/nipeiyang/codes/Text2LiDAR/Text2LiDAR_unconditional/logs/diffusion/kitti_360/spherical-1024/20241125T131019/models/diffusion_0000300000.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/nipeiyang/codes/Text2LiDAR/Text2LiDAR_unconditional/logs/diffusion/kitti_360/spherical-1024/20241125T131019/results",
    )
    parser.add_argument("--batch_size", type=int, default=35)  # 5 * 3090
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--num_steps", type=int, default=256)
    args = parser.parse_args()
    sample(args)
