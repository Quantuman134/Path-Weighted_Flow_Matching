import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name, wandb_key=None):
    config_dict = namespace_to_dict(args)
    if wandb_key:
        wandb.login(key=wandb_key)
    elif "WANDB_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x


def log_validation_images(real_images, generated_images, labels, step=None, num_samples=16):
    """
    Log validation comparison images to wandb.
    
    Args:
        real_images: Real validation images tensor (N, 3, H, W) in [-1, 1]
        generated_images: Generated images tensor (N, 3, H, W) in [-1, 1]
        labels: Class labels tensor (N,)
        step: Training step number
        num_samples: Number of random samples to log
    """
    if not is_main_process():
        return
    
    # Randomly select indices
    n = min(num_samples, real_images.size(0))
    indices = torch.randperm(real_images.size(0))[:n]
    
    # Select random samples
    real_samples = real_images[indices]
    gen_samples = generated_images[indices]
    sample_labels = labels[indices]
    
    # Create grids
    real_grid = array2grid(real_samples)
    gen_grid = array2grid(gen_samples)
    
    # Log to wandb with labels
    wandb.log({
        "validation/real_images": wandb.Image(real_grid, caption=f"Real (labels: {sample_labels.tolist()})"),
        "validation/generated_images": wandb.Image(gen_grid, caption=f"Generated (labels: {sample_labels.tolist()})"),
    }, step=step)