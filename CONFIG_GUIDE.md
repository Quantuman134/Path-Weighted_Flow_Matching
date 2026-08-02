# SiT Configuration File Usage Guide

## Overview

The SiT training script now supports YAML configuration files, making it easier to manage experiments and reproduce results. You can use config files exclusively or mix them with command-line arguments.

## Quick Start

### Using the Training Script

```bash
# Edit the config file with your settings
vim configs/sit_config.yaml

# Run training with the config file
./train.sh
```

Or directly with torchrun:

```bash
torchrun --nproc_per_node=8 train.py --config configs/sit_config.yaml
```

**Note:** A config file is required. All training parameters must be specified in the YAML configuration file.

## Configuration File Structure

All config files are located in the `configs/` directory:

```
configs/
├── sit_config.yaml           # Default template
└── imagenet256_example.yaml  # Example for ImageNet 256x256
```

### Configuration Sections

#### 1. Data Settings
```yaml
data:
  data_path: '/path/to/train'  # REQUIRED
  val_data_path: '/path/to/val'  # Optional for validation
  latent_scale: 1.0  # Optional: extra factor baked into pre-encoded latents
```

`latent_scale` (default `1.0`) describes latents that were stored pre-multiplied
by a constant — the variance subsets built by `build_variance_subset.py`. The
data on disk is used as-is; the factor is divided out before VAE decoding
(W&B samples, validation FID) and applied when raw images are encoded on the
fly. Pass the same value to `sample_ddp.py --latent-scale`.

#### 2. Model Settings
```yaml
model:
  model: 'SiT-XL/2'  # SiT-XL/2, SiT-L/2, SiT-B/2, SiT-S/2
  image_size: 256    # 256 or 512
  num_classes: 1000  # Number of classes
  vae: 'ema'         # 'ema' or 'mse'
```

Available models:
- `SiT-XL/2`: Extra-Large (675M params)
- `SiT-L/2`: Large (458M params)
- `SiT-B/2`: Base (130M params)
- `SiT-S/2`: Small (33M params)

#### 3. Transport Settings
```yaml
transport:
  path_type: 'Linear'      # 'Linear', 'GVP', or 'VP'
  prediction: 'velocity'   # 'velocity', 'score', or 'noise'
  loss_weight: null        # null, 'velocity', or 'likelihood'
  sample_eps: null         # Optional epsilon for sampling
  train_eps: null          # Optional epsilon for training
```

#### 4. Training Settings
```yaml
training:
  epochs: 1400              # Training epochs
  max_train_steps: null     # Optional: hard stop after N optimisation steps
  global_batch_size: 256    # Total batch across all GPUs
  global_seed: 0            # Random seed
  num_workers: 4            # Data loading workers
```

`max_train_steps` (default `null` = disabled) stops training once the step
count is reached, so a run can target a step budget instead of an epoch count.
`epochs` must still be large enough to reach it.

#### 5. Logging Settings
```yaml
logging:
  results_dir: 'results'    # Output directory
  log_every: 100            # Log frequency (steps)
  ckpt_every: 50000         # Checkpoint frequency (steps)
  sample_every: 10000       # Sample generation frequency (steps)
  wandb: false              # Enable W&B logging
  wandb_entity: 'default'   # W&B entity name (overrides ENTITY env var)
  wandb_project: 'SiT'      # W&B project name (overrides PROJECT env var)
```

#### 6. Validation Settings
```yaml
validation:
  val_num_samples: 5000     # Number of samples for FID
  val_log_images: 16        # Images to log to W&B
```

#### 7. Sampling Settings
```yaml
sampling:
  cfg_scale: 4.0            # Classifier-free guidance scale
```

#### 8. Checkpoint Settings
```yaml
checkpoint:
  ckpt: null                # Path to resume from
```

## Creating Custom Configurations

1. Copy an existing config:
   ```bash
   cp configs/sit_config.yaml configs/my_experiment.yaml
   ```

2. Edit your config file:
   ```yaml
   data:
     data_path: '/my/custom/dataset'
   
   model:
     model: 'SiT-L/2'
     image_size: 512
   
   training:
     epochs: 2000
     global_batch_size: 128
   ```

3. Update train.sh to use your config:
   ```bash
   config_file="./configs/my_experiment.yaml"
   ```

4. Run training:
   ```bash
   ./train.sh
   ```

## Examples

### Example 1: ImageNet 256x256 Training

```yaml
# configs/imagenet256.yaml
data:
  data_path: '/datasets/imagenet/train'
  val_data_path: '/datasets/imagenet/val'

model:
  model: 'SiT-XL/2'
  image_size: 256
  num_classes: 1000

training:
  epochs: 1400
  global_batch_size: 256
  global_seed: 0

logging:
  wandb: true
  wandb_entity: 'your-entity'
  wandb_project: 'SiT-ImageNet'
```

Run:
```bash
torchrun --nproc_per_node=8 train.py --config configs/imagenet256.yaml
```

### Example 2: Small Model for Quick Testing

```yaml
# configs/sit_small_test.yaml
data:
  data_path: '/datasets/imagenet/train'

model:
  model: 'SiT-S/2'
  image_size: 256
  num_classes: 1000

training:
  epochs: 100
  global_batch_size: 64

logging:
  log_every: 10
  ckpt_every: 5000
  sample_every: 1000
```

Run:
```bash
torchrun --nproc_per_node=4 train.py --config configs/sit_small_test.yaml
```

### Example 3: Resume from Checkpoint

```yaml
# configs/resume_training.yaml
data:
  data_path: '/datasets/imagenet/train'

model:
  model: 'SiT-XL/2'
  image_size: 256

checkpoint:
  ckpt: 'results/001-SiT-XL-2/checkpoints/0100000.pt'

training:
  epochs: 2000  # Continue to more epochs
```

Run:
```bash
torchrun --nproc_per_node=8 train.py --config configs/resume_training.yaml
```

## Tips

- **Start with a template**: Copy `configs/sit_config.yaml` or `configs/imagenet256_example.yaml`
- **Version control**: Save config files in git alongside your trained models
- **Reproducibility**: Config files ensure exact reproduction of training runs
- **Experimentation**: Create multiple configs for different hyperparameter settings
- **Quick changes**: Create a new config file for each experiment variation

## Troubleshooting

**Q: I get "data_path is required in config file"**
A: Make sure `data.data_path` is set in your config file and points to a valid directory.

**Q: Can I override config values from command line?**
A: No, all parameters must be specified in the config file. To change parameters, edit the YAML file or create a new config file.

**Q: Do I need to use a config file?**
A: Yes, the `--config` parameter is required. This ensures reproducibility and makes it easier to manage experiments.
