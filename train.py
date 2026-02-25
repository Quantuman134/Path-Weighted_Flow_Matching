# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
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
import yaml

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
import wandb_utils
import sys
import os

# Add parent directory to path for FID import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from FID import compute_fid
except ImportError:
    print('Warning: FID module not found. Validation will not be available.')
    compute_fid = None


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def load_config(config_path):
    """
    Load configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['data', 'model', 'transport', 'training', 'logging', 'validation', 'sampling', 'checkpoint']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file: {config_path}")
    
    return config


def config_to_args(config):
    """
    Convert config dict to argparse.Namespace for compatibility with existing code.
    """
    args = argparse.Namespace()
    
    # Data settings
    args.data_path = config['data']['data_path']
    if args.data_path is None:
        raise ValueError("data_path is required in config file")
    args.val_data_path = config['data'].get('val_data_path', None)
    
    # Model settings
    args.model = config['model']['model']
    args.image_size = int(config['model']['image_size'])
    args.num_classes = int(config['model']['num_classes'])
    args.vae = config['model']['vae']
    
    # Transport settings
    args.path_type = config['transport']['path_type']
    args.prediction = config['transport']['prediction']
    args.loss_weight = config['transport'].get('loss_weight', None)
    args.sample_eps = config['transport'].get('sample_eps', None)
    args.train_eps = config['transport'].get('train_eps', None)
    
    # Training settings
    args.epochs = int(config['training']['epochs'])
    args.global_batch_size = int(config['training']['global_batch_size'])
    args.global_seed = int(config['training']['global_seed'])
    args.num_workers = int(config['training']['num_workers'])
    
    # Logging settings
    args.results_dir = config['logging']['results_dir']
    args.log_every = int(config['logging']['log_every'])
    args.ckpt_every = int(config['logging']['ckpt_every'])
    args.sample_every = int(config['logging']['sample_every'])
    args.wandb = bool(config['logging'].get('wandb', False))
    args.wandb_key = config['logging'].get('wandb_key', os.environ.get('WANDB_KEY', None))
    args.wandb_entity = config['logging'].get('wandb_entity', os.environ.get('ENTITY', 'default'))
    args.wandb_project = config['logging'].get('wandb_project', os.environ.get('PROJECT', 'SiT'))
    
    # Validation settings
    args.val_num_samples = int(config['validation']['val_num_samples'])
    args.val_log_images = int(config['validation']['val_log_images'])
    
    # Sampling settings
    args.cfg_scale = float(config['sampling']['cfg_scale'])
    
    # Checkpoint settings
    args.ckpt = config['checkpoint'].get('ckpt', None)
    
    return args


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
def validate_fid(ema_model, vae, val_loader, transport_sampler, args, device, rank, logger):
    """
    Compute FID score on validation set.
    
    Args:
        ema_model: EMA model for generation
        vae: VAE for encoding/decoding
        val_loader: Validation data loader
        transport_sampler: Transport sampler for generation
        args: Training arguments
        device: Device to run on
        rank: DDP rank
        logger: Logger instance
    
    Returns:
        FID score (float), sample_images (dict or None)
    """
    if compute_fid is None:
        logger.info('FID computation not available. Skipping validation.')
        return None, None
    
    ema_model.eval()
    latent_size = args.image_size // 8
    use_cfg = args.cfg_scale > 1.0
    
    # Collect validation images and labels
    real_images_list = []
    labels_list = []
    
    for x, y in val_loader:
        real_images_list.append(x)
        labels_list.append(y)
    
    # Concatenate and move to device
    real_images = torch.cat(real_images_list, dim=0).to(device)  # (N, 3, H, W) in [-1, 1]
    labels = torch.cat(labels_list, dim=0).to(device)  # (N,)
    
    n = real_images.size(0)
    logger.info(f"Generating {n} samples for FID validation...")
    
    # Generate images with same class labels
    generated_images_list = []
    batch_size = args.global_batch_size // dist.get_world_size()
    
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        curr_batch_size = end_idx - i
        
        # Get labels for this batch
        ys = labels[i:end_idx]
        
        # Create noise
        zs = torch.randn(curr_batch_size, 4, latent_size, latent_size, device=device)
        
        # Setup for CFG if needed
        if use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([1000] * curr_batch_size, device=device)
            ys = torch.cat([ys, y_null], 0)
            sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
            model_fn = ema_model.forward_with_cfg
        else:
            sample_model_kwargs = dict(y=ys)
            model_fn = ema_model.forward
        
        # Generate samples
        sample_fn = transport_sampler.sample_ode()
        samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
        
        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)
        
        # Decode from latent space
        samples = vae.decode(samples / 0.18215).sample
        generated_images_list.append(samples.cpu())
    
    generated_images = torch.cat(generated_images_list, dim=0)  # (N, 3, H, W) in [-1, 1]
    
    # Gather all images and labels across all GPUs
    # Note: Use list-based gather to handle different sizes per rank (when dataset doesn't divide evenly)
    all_real_list = [torch.zeros_like(real_images) for _ in range(dist.get_world_size())]
    all_gen_list = [torch.zeros_like(generated_images) for _ in range(dist.get_world_size())]
    all_labels_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
    
    dist.all_gather(all_real_list, real_images)
    dist.all_gather(all_gen_list, generated_images.to(device))
    dist.all_gather(all_labels_list, labels)
    
    # Compute FID only on rank 0
    fid_score = None
    sample_images = None
    if rank == 0:
        # Concatenate all gathered tensors
        all_real_images = torch.cat(all_real_list, dim=0)
        all_generated_images = torch.cat(all_gen_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)
        
        # Store samples for visualization (keep in [-1, 1] range)
        sample_images = {
            'real': all_real_images.cpu(),
            'generated': all_generated_images.cpu(),
            'labels': all_labels.cpu()
        }
        
        # Convert from [-1, 1] to [0, 1] for FID computation
        all_real_images = (all_real_images + 1.0) / 2.0
        all_generated_images = (all_generated_images + 1.0) / 2.0
        
        # Clamp to [0, 1]
        all_real_images = torch.clamp(all_real_images, 0.0, 1.0)
        all_generated_images = torch.clamp(all_generated_images, 0.0, 1.0)
        
        logger.info('Computing FID score...')
        fid_score = compute_fid(all_real_images.cpu(), all_generated_images.cpu(), 
                               batch_size=32, device=str(device))
        logger.info(f'Validation FID Score: {fid_score:.4f}')
    
    dist.barrier()
    return fid_score, sample_images


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            wandb_utils.initialize(args, args.wandb_entity, experiment_name, args.wandb_project, args.wandb_key)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Load checkpoint if provided (before DDP wrapping)
    checkpoint_state = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        assert os.path.isfile(ckpt_path), f'Could not find SiT checkpoint at {ckpt_path}'
        checkpoint_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint_state["model"])
        ema.load_state_dict(checkpoint_state["ema"])
        # Note: We don't override args from checkpoint to allow config file to control all settings
        # The model weights and EMA weights are loaded, which is what's needed to resume training

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Load optimizer state if resuming from checkpoint
    if checkpoint_state is not None:
        opt.load_state_dict(checkpoint_state["opt"])

    # Setup data:
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
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Setup validation data if provided:
    val_loader = None
    if args.val_data_path is not None:
        val_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
        
        # Randomly sample val_num_samples from validation set
        if args.val_num_samples > 0 and args.val_num_samples < len(val_dataset):
            indices = torch.randperm(len(val_dataset))[:args.val_num_samples].tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
        
        # Use DistributedSampler for validation as well
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Validation dataset contains {len(val_dataset):,} images ({args.val_data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # Resume from checkpoint if available
    start_epoch = 0
    if checkpoint_state is not None and "train_steps" in checkpoint_state:
        train_steps = checkpoint_state["train_steps"]
        start_epoch = checkpoint_state.get("epoch", 0) + 1  # Start from next epoch
        logger.info(f"Resuming from step {train_steps}, epoch {start_epoch}")
    else:
        train_steps = 0
    
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / 0.18215).sample
                    out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logger.info("Generating EMA samples done.")
                
                # Run validation if validation data is provided
                if val_loader is not None:
                    logger.info("Running validation...")
                    fid_score, sample_images = validate_fid(ema, vae, val_loader, transport_sampler, args, device, rank, logger)
                    if rank == 0 and fid_score is not None and args.wandb:
                        wandb_utils.log({"validation/FID": fid_score}, step=train_steps)
                    if rank == 0 and sample_images is not None and args.wandb:
                        wandb_utils.log_validation_images(
                            sample_images['real'], 
                            sample_images['generated'],
                            sample_images['labels'],
                            step=train_steps,
                            num_samples=args.val_log_images
                        )
                    model.train()  # Set back to training mode
                    logger.info("Validation done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiT Training with YAML configuration")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    # Load config file and convert to args
    config = load_config(args.config)
    args = config_to_args(config)
    
    main(args)
