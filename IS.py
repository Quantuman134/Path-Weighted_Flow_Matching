import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
from PIL import Image


def get_inception_probs(
    images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
    model=None,
) -> np.ndarray:
    """
    Extract softmax class probabilities from Inception v3.

    Args:
        images: Tensor of shape (N, 3, H, W), values in [0, 1]
        batch_size: How many images to process at once
        device: 'cuda' or 'cpu'
        model: Optional pre-loaded Inception v3 model (with fc layer intact).
               If None, a fresh model is loaded. Useful for sharing one model
               instance across multiple calls (e.g., in an alpha sweep).

    Returns:
        numpy array of shape (N, 1000) — softmax class probabilities
    """
    own_model = model is None
    if own_model:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        # NOTE: do NOT set model.fc = nn.Identity() — IS needs the full classifier
        model.eval().to(device)

    resize = transforms.Resize((299, 299), antialias=True)

    probs = []
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in loader:
            batch = resize(batch).to(device)
            logits = model(batch)          # (B, 1000)
            p = F.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())

    return np.concatenate(probs, axis=0)   # (N, 1000)


def compute_is_from_probs(
    probs: np.ndarray,
    splits: int = 10,
) -> tuple:
    """
    Compute Inception Score from pre-extracted softmax probabilities.

    IS = exp(E_x[ KL( p(y|x) || p(y) ) ])

    Splits the images into `splits` groups, computes a per-split IS,
    and returns the mean and std across splits.

    Args:
        probs: numpy array of shape (N, 1000) — softmax class probabilities
        splits: number of splits for mean/std estimation (default: 10)

    Returns:
        (mean_is, std_is) as floats (higher IS is better)
    """
    if len(probs) < splits:
        raise ValueError(
            f"Number of images ({len(probs)}) must be >= splits ({splits}). "
            "Reduce splits or generate more images."
        )

    chunks = np.array_split(probs, splits)
    split_scores = []

    for part in chunks:
        p_y = part.mean(axis=0)                            # marginal p(y)
        kl_divs = np.array([entropy(p_yx, p_y) for p_yx in part])  # KL per image
        split_scores.append(np.exp(kl_divs.mean()))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def compute_is(
    images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
    splits: int = 10,
) -> tuple:
    """
    Compute Inception Score end-to-end (loads its own Inception model).

    IS = exp(E_x[ KL( p(y|x) || p(y) ) ])

    Args:
        images: Tensor of shape (N, 3, H, W), values in [0, 1]
        batch_size: Batch size for Inception forward passes
        device: 'cuda' or 'cpu'
        splits: Number of splits for mean/std estimation

    Returns:
        (mean_is, std_is) as floats (higher IS is better)
    """
    probs = get_inception_probs(images, batch_size=batch_size, device=device)
    return compute_is_from_probs(probs, splits=splits)


def load_images_as_tensor(image_dir: str) -> torch.Tensor:
    """Load all images from a directory (recursively) as a (N, 3, H, W) tensor in [0, 1]."""
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    images = []
    to_tensor = transforms.ToTensor()
    for path in sorted(image_paths):
        img = Image.open(path).convert('RGB')
        images.append(to_tensor(img).unsqueeze(0))   # (1, 3, H, W)

    return torch.cat(images, dim=0)   # (N, 3, H, W)


def arg_parse():
    parser = argparse.ArgumentParser(description="Calculate Inception Score (IS)")
    parser.add_argument("--generated_images_dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Inception forward passes")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--splits", type=int, default=10,
                        help="Number of splits for IS mean/std estimation")
    return parser.parse_args()


def main(args):
    images = load_images_as_tensor(args.generated_images_dir)
    print(f"Loaded {len(images)} images from {args.generated_images_dir}")
    mean_is, std_is = compute_is(
        images, batch_size=args.batch_size, device=args.device, splits=args.splits
    )
    print(f"IS Score: mean={mean_is:.4f}, std={std_is:.4f}")


if __name__ == "__main__":
    args = arg_parse()
    main(args)
