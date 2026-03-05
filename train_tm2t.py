# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for SiT-TM2T: a two-stage generation model.

Architecture:
  - Pretrained SiT (frozen): generates from t=0 (noise) to t=m
  - TM2T model (trainable): generates from t=m to t=1 (clean image)

Training:
  - TM2T is trained with standard flow matching loss, but timesteps
    are restricted to [t_min, 1] via Transport(t_min=m).

Inference:
  - Stage 1: pretrained SiT ODE from t=0 → t=m
  - Stage 2: TM2T ODE from t=m → t=1 using stage 1 output as init
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math
import yaml

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
import wandb_utils

try:
    from FID import compute_fid
except ImportError:
    print('Warning: FID module not found. Validation will not be available.')
    compute_fid = None


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_sections = ['data', 'model', 'transport', 'tm2t', 'training',
                         'logging', 'validation', 'sampling', 'checkpoint']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file: {config_path}")

    return config


def config_to_args(config):
    args = argparse.Namespace()

    # Data
    args.data_path = config['data']['data_path']
    if args.data_path is None:
        raise ValueError("data_path is required in config file")
    args.val_data_path = config['data'].get('val_data_path', None)

    # Model
    args.model = config['model']['model']
    args.image_size = int(config['model']['image_size'])
    args.num_classes = int(config['model']['num_classes'])
    args.vae = config['model']['vae']

    # Transport
    args.path_type = config['transport']['path_type']
    args.prediction = config['transport']['prediction']
    args.pretrained_prediction = config['transport'].get('pretrained_prediction', args.prediction)
    args.loss_weight = config['transport'].get('loss_weight', None)
    args.sample_eps = config['transport'].get('sample_eps', None)
    args.train_eps = config['transport'].get('train_eps', None)

    # TM2T
    args.t_min = float(config['tm2t']['t_min'])
    args.pretrained_ckpt = config['tm2t']['pretrained_ckpt']
    if args.pretrained_ckpt is None:
        raise ValueError("tm2t.pretrained_ckpt is required in config file")

    # Training
    args.epochs = int(config['training']['epochs'])
    args.global_batch_size = int(config['training']['global_batch_size'])
    args.global_seed = int(config['training']['global_seed'])
    args.num_workers = int(config['training']['num_workers'])

    # Logging
    args.results_dir = config['logging']['results_dir']
    args.log_every = int(config['logging']['log_every'])
    args.ckpt_every = int(config['logging']['ckpt_every'])
    args.sample_every = int(config['logging']['sample_every'])
    args.wandb = bool(config['logging'].get('wandb', False))
    args.wandb_key = config['logging'].get('wandb_key', os.environ.get('WANDB_KEY', None))
    args.wandb_entity = config['logging'].get('wandb_entity', os.environ.get('ENTITY', 'default'))
    args.wandb_project = config['logging'].get('wandb_project', os.environ.get('PROJECT', 'SiT-TM2T'))

    # Validation
    args.val_num_samples = int(config['validation']['val_num_samples'])
    args.val_log_images = int(config['validation']['val_log_images'])

    # Sampling
    args.cfg_scale = float(config['sampling']['cfg_scale'])
    args.num_sampling_steps = int(config['sampling'].get('num_sampling_steps', 250))

    # Checkpoint
    args.ckpt = config['checkpoint'].get('ckpt', None)

    return args


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@torch.no_grad()
def two_stage_sample(
    pretrained_model,
    tm2t_model,
    pretrained_sampler,
    tm2t_sampler,
    z,
    model_kwargs,
    t_min,
    steps_stage1,
    steps_stage2,
):
    """
    Two-stage ODE sampling:
      Stage 1: pretrained SiT  t=0      → t=t_min  (noise → mid-point)
      Stage 2: TM2T            t=t_min  → t=1       (mid-point → image)

    Args:
        pretrained_model: frozen pretrained SiT (use .forward or .forward_with_cfg)
        tm2t_model: trained TM2T model (EMA preferred)
        pretrained_sampler: Sampler wrapping a standard Transport (t_min=0)
        tm2t_sampler: Sampler wrapping a Transport with t_min=args.t_min
        z: initial Gaussian noise [B, C, H, W]
        model_kwargs: dict passed to both models (y, cfg_scale, …)
        t_min: the handoff timestep m
        steps_stage1: ODE steps for stage 1
        steps_stage2: ODE steps for stage 2
    """
    use_cfg = 'cfg_scale' in model_kwargs and model_kwargs['cfg_scale'] > 1.0
    model_fn_pretrained = pretrained_model.forward_with_cfg if use_cfg else pretrained_model.forward
    model_fn_tm2t = tm2t_model.forward_with_cfg if use_cfg else tm2t_model.forward

    # Stage 1: noise → x_m
    stage1_fn = pretrained_sampler.sample_ode(
        num_steps=steps_stage1,
        t1_override=t_min,
    )
    x_m = stage1_fn(z, model_fn_pretrained, **model_kwargs)[-1]

    # Stage 2: x_m → x_1
    stage2_fn = tm2t_sampler.sample_ode(
        num_steps=steps_stage2,
        t0_override=t_min,
    )
    x_1 = stage2_fn(x_m, model_fn_tm2t, **model_kwargs)[-1]

    return x_1


@torch.no_grad()
def validate_fid_tm2t(
    ema_model,
    vae,
    val_loader,
    pretrained_model,
    pretrained_sampler,
    tm2t_sampler,
    args,
    device,
    rank,
    logger,
    steps_stage1,
    steps_stage2,
):
    """
    Compute FID score on the validation set using two-stage sampling.
    Mirrors validate_fid() from train.py, replacing single-model generation
    with two_stage_sample (pretrained SiT t=0→m, TM2T t=m→1).
    """
    if compute_fid is None:
        logger.info('FID computation not available. Skipping validation.')
        return None, None

    ema_model.eval()
    latent_size = args.image_size // 8
    use_cfg = args.cfg_scale > 1.0

    real_images_list = []
    labels_list = []
    for x, y in val_loader:
        real_images_list.append(x)
        labels_list.append(y)

    real_images = torch.cat(real_images_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)
    n = real_images.size(0)
    logger.info(f"Rank {rank}: Generating {n} samples for FID validation...")

    generated_images_list = []
    batch_size = args.global_batch_size // dist.get_world_size()

    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        curr_batch_size = end_idx - i

        ys = labels[i:end_idx]
        zs = torch.randn(curr_batch_size, 4, latent_size, latent_size, device=device)

        if use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([1000] * curr_batch_size, device=device)
            ys = torch.cat([ys, y_null], 0)
            sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        else:
            sample_model_kwargs = dict(y=ys)

        samples = two_stage_sample(
            pretrained_model=pretrained_model,
            tm2t_model=ema_model,
            pretrained_sampler=pretrained_sampler,
            tm2t_sampler=tm2t_sampler,
            z=zs,
            model_kwargs=sample_model_kwargs,
            t_min=args.t_min,
            steps_stage1=steps_stage1,
            steps_stage2=steps_stage2,
        )

        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        generated_images_list.append(samples.cpu())

    generated_images = torch.cat(generated_images_list, dim=0)

    # Gather across GPUs (handles uneven distribution with padding)
    local_size = torch.tensor([n], device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [int(s.item()) for s in all_sizes]
    max_size = max(all_sizes)

    if n < max_size:
        pad_size = max_size - n
        real_images = torch.cat([real_images, torch.zeros(pad_size, *real_images.shape[1:], device=device)], dim=0)
        generated_images = torch.cat([generated_images.to(device), torch.zeros(pad_size, *generated_images.shape[1:], device=device)], dim=0)
        labels = torch.cat([labels, torch.zeros(pad_size, dtype=labels.dtype, device=device)], dim=0)
    else:
        generated_images = generated_images.to(device)

    all_real_list = [torch.zeros_like(real_images) for _ in range(dist.get_world_size())]
    all_gen_list = [torch.zeros_like(generated_images) for _ in range(dist.get_world_size())]
    all_labels_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
    dist.all_gather(all_real_list, real_images)
    dist.all_gather(all_gen_list, generated_images)
    dist.all_gather(all_labels_list, labels)

    fid_score = None
    sample_images = None
    if rank == 0:
        all_real_images = torch.cat(all_real_list, dim=0)
        all_generated_images = torch.cat(all_gen_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        total_samples = sum(all_sizes)
        all_real_images = all_real_images[:total_samples]
        all_generated_images = all_generated_images[:total_samples]
        all_labels = all_labels[:total_samples]
        logger.info(f"Total validation samples gathered from all ranks: {total_samples}")

        sample_images = {
            'real': all_real_images.cpu(),
            'generated': all_generated_images.cpu(),
            'labels': all_labels.cpu(),
        }

        all_real_images = torch.clamp((all_real_images + 1.0) / 2.0, 0.0, 1.0)
        all_generated_images = torch.clamp((all_generated_images + 1.0) / 2.0, 0.0, 1.0)

        logger.info('Computing FID score...')
        fid_device = 'cuda' if isinstance(device, int) or (hasattr(device, 'type') and device.type == 'cuda') else 'cpu'
        fid_score = compute_fid(
            all_real_images.cpu(),
            all_generated_images.cpu(),
            batch_size=32,
            device=fid_device,
        )
        logger.info(f'Validation FID Score: {fid_score:.4f}')

    dist.barrier()
    return fid_score, sample_images


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = (
            f"{experiment_index:03d}-TM2T-{model_string_name}-"
            f"{args.path_type}-{args.prediction}-tmin{args.t_min}"
        )
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"TM2T t_min={args.t_min}  (pretrained handles [0, {args.t_min}], TM2T handles [{args.t_min}, 1])")

        if args.wandb:
            wandb_utils.initialize(args, args.wandb_entity, experiment_name, args.wandb_project, args.wandb_key)
    else:
        logger = create_logger(None)

    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8

    # -------------------------------------------------------------------------
    # Pretrained SiT (frozen) — used only during inference
    # -------------------------------------------------------------------------
    pretrained_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    assert os.path.isfile(args.pretrained_ckpt), \
        f"Pretrained checkpoint not found: {args.pretrained_ckpt}"
    pretrained_state = torch.load(
        args.pretrained_ckpt, map_location=lambda storage, loc: storage, weights_only=False
    )
    pretrained_model.load_state_dict(pretrained_state["ema"])
    pretrained_model.eval()
    requires_grad(pretrained_model, False)
    logger.info(f"Loaded pretrained SiT from {args.pretrained_ckpt}")

    # Sampler for pretrained model (full [0, 1] interval; t1_override=t_min at inference)
    pretrained_transport = create_transport(
        args.path_type,
        args.pretrained_prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        t_min=0.0,
    )
    pretrained_sampler = Sampler(pretrained_transport)

    # -------------------------------------------------------------------------
    # TM2T model (trainable)
    # -------------------------------------------------------------------------
    tm2t_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    ema = deepcopy(tm2t_model).to(device)

    # Resume TM2T from checkpoint if provided
    checkpoint_state = None
    if args.ckpt is not None:
        assert os.path.isfile(args.ckpt), f"TM2T checkpoint not found: {args.ckpt}"
        checkpoint_state = torch.load(
            args.ckpt, map_location=lambda storage, loc: storage, weights_only=False
        )
        tm2t_model.load_state_dict(checkpoint_state["model"])
        ema.load_state_dict(checkpoint_state["ema"])

    requires_grad(ema, False)
    tm2t_model = DDP(tm2t_model.to(device), device_ids=[device])

    # Transport for TM2T: restricts training timesteps to [t_min, 1]
    tm2t_transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        t_min=args.t_min,
    )
    tm2t_sampler = Sampler(tm2t_transport)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"TM2T Parameters: {sum(p.numel() for p in tm2t_model.parameters()):,}")

    opt = torch.optim.AdamW(tm2t_model.parameters(), lr=1e-4, weight_decay=0)
    if checkpoint_state is not None:
        opt.load_state_dict(checkpoint_state["opt"])

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=True, 
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset, 
        batch_size=local_batch_size, 
        shuffle=False, 
        sampler=sampler,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True, 
        prefetch_factor=12,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    val_loader = None
    if args.val_data_path is not None:
        val_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
        if args.val_num_samples > 0 and args.val_num_samples < len(val_dataset):
            indices = torch.randperm(len(val_dataset))[:args.val_num_samples].tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=4,
        )
        logger.info(f"Validation dataset contains {len(val_dataset):,} images ({args.val_data_path})")

    # -------------------------------------------------------------------------
    # Prepare for training
    # -------------------------------------------------------------------------
    if checkpoint_state is None:
        # Fresh training: sync EMA to model's initial weights
        update_ema(ema, tm2t_model.module, decay=0)
    tm2t_model.train()
    ema.eval()

    # Split sampling steps proportionally between the two stages
    steps_stage1 = max(1, round(args.num_sampling_steps * args.t_min))
    steps_stage2 = max(1, args.num_sampling_steps - steps_stage1)

    # Fixed noise/labels for periodic visualisation
    use_cfg = args.cfg_scale > 1.0
    vis_ys = torch.randint(1000, size=(local_batch_size,), device=device)
    vis_zs = torch.randn(local_batch_size, 4, latent_size, latent_size, device=device)
    if use_cfg:
        vis_zs = torch.cat([vis_zs, vis_zs], 0)
        y_null = torch.tensor([1000] * local_batch_size, device=device)
        vis_ys = torch.cat([vis_ys, y_null], 0)
        vis_model_kwargs = dict(y=vis_ys, cfg_scale=args.cfg_scale)
    else:
        vis_model_kwargs = dict(y=vis_ys)

    start_epoch = 0
    if checkpoint_state is not None and "train_steps" in checkpoint_state:
        train_steps = checkpoint_state["train_steps"]
        start_epoch = checkpoint_state.get("epoch", 0) + 1
        logger.info(f"Resuming from step {train_steps}, epoch {start_epoch}")
    else:
        train_steps = 0

    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"  Stage 1 steps (pretrained, t=0→{args.t_min}): {steps_stage1}")
    logger.info(f"  Stage 2 steps (TM2T,       t={args.t_min}→1): {steps_stage2}")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)
            # TM2T training loss: identical to SiT FM loss, but t ~ U[t_min, 1]
            loss_dict = tm2t_transport.training_losses(tm2t_model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, tm2t_model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if args.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                        step=train_steps,
                    )
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": tm2t_model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating two-stage EMA samples...")
                with torch.no_grad():
                    samples = two_stage_sample(
                        pretrained_model=pretrained_model,
                        tm2t_model=ema,
                        pretrained_sampler=pretrained_sampler,
                        tm2t_sampler=tm2t_sampler,
                        z=vis_zs,
                        model_kwargs=vis_model_kwargs,
                        t_min=args.t_min,
                        steps_stage1=steps_stage1,
                        steps_stage2=steps_stage2,
                    )
                    dist.barrier()
                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / 0.18215).sample
                    out_samples = torch.zeros(
                        (args.global_batch_size, 3, args.image_size, args.image_size), device=device
                    )
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logger.info("Generating two-stage EMA samples done.")

                if val_loader is not None:
                    logger.info("Running validation...")
                    fid_score, sample_images = validate_fid_tm2t(
                        ema_model=ema,
                        vae=vae,
                        val_loader=val_loader,
                        pretrained_model=pretrained_model,
                        pretrained_sampler=pretrained_sampler,
                        tm2t_sampler=tm2t_sampler,
                        args=args,
                        device=device,
                        rank=rank,
                        logger=logger,
                        steps_stage1=steps_stage1,
                        steps_stage2=steps_stage2,
                    )
                    if rank == 0 and fid_score is not None and args.wandb:
                        wandb_utils.log({"validation/FID": fid_score}, step=train_steps)
                    if rank == 0 and sample_images is not None and args.wandb:
                        wandb_utils.log_validation_images(
                            sample_images['real'],
                            sample_images['generated'],
                            sample_images['labels'],
                            step=train_steps,
                            num_samples=args.val_log_images,
                        )
                    logger.info("Validation done.")

    tm2t_model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM2T Training with YAML configuration")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    args = config_to_args(config)
    main(args)
