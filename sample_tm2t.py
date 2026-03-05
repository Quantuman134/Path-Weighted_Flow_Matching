# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Two-stage image sampling with a pretrained SiT + TM2T model.

Stage 1 (pretrained SiT):  Gaussian noise  →  x_m  (t = 0  → t_min)
Stage 2 (TM2T):            x_m             →  image (t = t_min → 1)
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import os
from time import time


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8

    # -------------------------------------------------------------------------
    # Load pretrained (frozen) SiT — handles t = 0 → t_min
    # -------------------------------------------------------------------------
    pretrained_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    assert os.path.isfile(args.pretrained_ckpt), \
        f"Pretrained checkpoint not found: {args.pretrained_ckpt}"
    pretrained_state = torch.load(
        args.pretrained_ckpt, map_location="cpu", weights_only=False
    )
    pretrained_model.load_state_dict(pretrained_state["ema"])
    pretrained_model.eval()
    print(f"Loaded pretrained SiT from {args.pretrained_ckpt}")

    pretrained_transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        t_min=0.0,
    )
    pretrained_sampler = Sampler(pretrained_transport)

    # -------------------------------------------------------------------------
    # Load TM2T model — handles t = t_min → 1
    # -------------------------------------------------------------------------
    tm2t_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    assert os.path.isfile(args.tm2t_ckpt), \
        f"TM2T checkpoint not found: {args.tm2t_ckpt}"
    tm2t_state = torch.load(args.tm2t_ckpt, map_location="cpu", weights_only=False)
    tm2t_model.load_state_dict(tm2t_state["ema"])
    tm2t_model.eval()
    print(f"Loaded TM2T model from {args.tm2t_ckpt}")

    tm2t_transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        t_min=args.t_min,
    )
    tm2t_sampler = Sampler(tm2t_transport)

    # -------------------------------------------------------------------------
    # VAE
    # -------------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    # Split steps proportionally to the time intervals each stage covers
    steps_stage1 = max(1, round(args.num_sampling_steps * args.t_min))
    steps_stage2 = max(1, args.num_sampling_steps - steps_stage1)
    print(f"t_min={args.t_min}  |  stage-1 steps: {steps_stage1}  |  stage-2 steps: {steps_stage2}")

    class_labels = args.class_labels
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fn_pretrained = pretrained_model.forward_with_cfg
        model_fn_tm2t = tm2t_model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fn_pretrained = pretrained_model.forward
        model_fn_tm2t = tm2t_model.forward

    start_time = time()

    # Stage 1: noise → x_m
    stage1_fn = pretrained_sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=steps_stage1,
        atol=args.atol,
        rtol=args.rtol,
        t1_override=args.t_min,
    )
    x_m = stage1_fn(z, model_fn_pretrained, **model_kwargs)[-1]
    print(f"Stage 1 done ({time() - start_time:.2f}s)  x_m shape: {x_m.shape}")

    # Stage 2: x_m → x_1
    stage2_fn = tm2t_sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=steps_stage2,
        atol=args.atol,
        rtol=args.rtol,
        t0_override=args.t_min,
    )
    samples = stage2_fn(x_m, model_fn_tm2t, **model_kwargs)[-1]
    print(f"Stage 2 done ({time() - start_time:.2f}s total)")

    if use_cfg:
        samples, _ = samples.chunk(2, dim=0)

    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, args.output, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved {n} images to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage TM2T sampling")

    # Model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    # Checkpoints
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="Path to pretrained SiT checkpoint")
    parser.add_argument("--tm2t-ckpt", type=str, required=True,
                        help="Path to TM2T checkpoint")

    # TM2T
    parser.add_argument("--t-min", type=float, default=0.5,
                        help="Handoff timestep m: pretrained runs [0, t_min], TM2T runs [t_min, 1]")

    # Sampling
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampling-method", type=str, default="dopri5")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-labels", type=int, nargs="+",
                        default=[207, 360, 387, 974, 88, 979, 417, 279])
    parser.add_argument("--output", type=str, default="sample_tm2t.png")

    parse_transport_args(parser)

    args = parser.parse_args()
    main(args)
