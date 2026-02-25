import argparse
import os
import sys

# Add Generative_Rendering directory to path for utils import
current_dir = os.path.dirname(os.path.abspath(__file__))
generative_rendering_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if generative_rendering_dir not in sys.path:
    sys.path.insert(0, generative_rendering_dir)

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from utils import export_tensor_as_image, import_image_as_tensor

def arg_parse():
    parser = argparse.ArgumentParser(description="Calculate FID score")
    parser.add_argument("--generated_images_dir", type=str, help="Directory containing generated images")
    parser.add_argument("--real_images_dir", type=str, help="Directory containing real images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for Inception feature extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cuda' or 'cpu')")
    
    return parser.parse_args()

def get_inception_features(images: torch.Tensor, batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    """
    Extract 2048-d feature vectors from Inception v3's pool layer.
    
    Args:
        images: Tensor of shape (N, 3, H, W), values in [0, 1]
        batch_size: How many images to process at once
        device: 'cuda' or 'cpu'
    
    Returns:
        numpy array of shape (N, 2048)
    """
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()       # Remove classification head
    model.aux_logits = False       # Disable auxiliary output
    model.eval().to(device)

    # Inception v3 expects 299x299
    resize = transforms.Resize((299, 299), antialias=True)

    features = []
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in loader:
            batch = resize(batch).to(device)
            feat = model(batch)          # (B, 2048)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(real_images: torch.Tensor,
                fake_images: torch.Tensor,
                batch_size: int = 64,
                device: str = "cuda") -> float:
    """
    Compute the Fréchet Inception Distance (FID) between real and generated images.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·sqrt(Σ_r · Σ_g))

    Args:
        real_images: Tensor (N, 3, H, W), pixel values in [0, 1]
        fake_images: Tensor (M, 3, H, W), pixel values in [0, 1]
        batch_size:  Batch size for Inception forward passes
        device:      'cuda' or 'cpu'

    Returns:
        FID score as a float (lower is better)
    """
    real_feats = get_inception_features(real_images, batch_size, device)
    fake_feats = get_inception_features(fake_images, batch_size, device)

    # Means
    mu_r = real_feats.mean(axis=0)
    mu_g = fake_feats.mean(axis=0)

    # Covariance matrices (rowvar=False → each column is a variable)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(fake_feats, rowvar=False)

    # Matrix square root of (Σ_r · Σ_g)
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    # sqrtm can produce tiny imaginary parts due to numerical noise — discard them
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # FID formula
    diff = mu_r - mu_g
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)

    return float(fid)

def load_images_as_tensor(image_dir: str) -> torch.Tensor:
    # load all image from nested directories and convert to tensor
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    images = []
    for path in image_paths:
        img_tensor = import_image_as_tensor(path)  # (1, 3, H, W), values in [0, 1]
        images.append(img_tensor)

    return torch.cat(images, dim=0)  # (N, 3, H, W)

def main(args):
    # Load real and generated images as tensors (shape: N, 3, H, W, values in [0, 1])
    real_images = load_images_as_tensor(args.real_images_dir)
    fake_images = load_images_as_tensor(args.generated_images_dir)
    real_images = real_images.to(args.device)
    fake_images = fake_images.to(args.device)

    fid_score = compute_fid(real_images, fake_images, batch_size=args.batch_size, device=args.device)
    print(f"FID Score: {fid_score:.4f}")

if __name__ == "__main__":
    args = arg_parse()
    main(args)
