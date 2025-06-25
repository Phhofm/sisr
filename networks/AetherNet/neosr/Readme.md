# AetherNet neosr Integration

This directory contains the necessary files to integrate AetherNet models for training and evaluation within the `neosr` framework.

## What is `neosr`?

[`neosr`](https://github.com/xinntao/neosr) is a popular and actively developed PyTorch-based training framework specializing in image restoration tasks, including Single Image Super-Resolution (SISR), Denoising, Deblurring, and more. It provides a robust and flexible environment for conducting experiments, managing datasets, configuring models, and defining complex training pipelines.

## AetherNet and `neosr`: Empowering Your Training

Integrating AetherNet with `neosr` allows you to harness `neosr`'s comprehensive training capabilities. This includes its advanced data processing, diverse range of loss functions, state-of-the-art optimizers, efficient multi-GPU training, and systematic logging and validation. `neosr` streamlines the entire research and development workflow for AetherNet.

### Key Aspects of this Integration:

1. **Modular Architecture Definition and Registration:**

   * The `aether_arch.py` file within this directory meticulously defines the AetherNet architecture (`aether` class and its variants: `aether_small`, `aether_medium`, `aether_large`).

   * Crucially, these model classes are registered with `neosr`'s `ARCH_REGISTRY`. This means `neosr` automatically discovers them, allowing you to instantiate any AetherNet variant simply by referencing its name (e.g., `aether_medium`) in your configuration files, without needing direct import statements.

2. **Configuration-Driven Training:**

   * `neosr` relies on TOML or YAML configuration files to orchestrate every aspect of a training run.

   * You can specify the desired AetherNet variant, its architectural parameters (`embed_dim`, `depths`, `mlp_ratio`, `lk_kernel`, `sk_kernel`, `drop_rate`, `drop_path_rate`, `img_range`), and global training settings directly within these configuration files. This promotes reproducibility and ease of experimentation.

3. **Intelligent Upscale Factor Handling:**

   * A unique feature of this `neosr` integration is how it handles the `scale` parameter. The AetherNet architecture in `neosr/archs/aether_arch.py` is specifically designed to automatically retrieve the global `scale` factor defined at the top level of your `neosr` configuration file (e.g., `scale = 2`).

   * This ensures that the model's internal upsampling layers are correctly initialized to match the target super-resolution factor of your dataset, eliminating manual configuration mismatches.

4. **Support for Fused Initialization (for Inference/Loading):**

   * The AetherNet implementation includes a `fused_init` parameter. While typically set to `false` during training, this parameter allows you to load an already fused AetherNet model (created using `fuse_aether.py`) directly within `neosr` for validation, testing, or benchmark purposes. This is valuable for evaluating the performance of your deployment-ready models.

### Benefits of Using AetherNet with `neosr`:

* **Established Training Pipeline:** Leverage `neosr`'s battle-tested and highly optimized training loop, which includes features like multi-GPU training, automatic mixed precision (AMP), and distributed data parallel (DDP) support for efficient scaling.

* **Comprehensive Feature Set:** Gain access to `neosr`'s extensive collection of loss functions (e.g., L1, GAN, perceptual, DISTS), optimizers, learning rate schedulers, and advanced regularization techniques.

* **Reproducibility:** The configuration-driven approach makes it easy to reproduce experiments, track changes, and share your exact training setup with others.

* **Active Community & Tools:** Benefit from `neosr`'s active development, community support, and robust tooling for dataset preparation, result analysis, and model management.

## Installation / Setup

To integrate AetherNet into your `neosr` environment:

1. **Ensure `neosr` is Installed:**
   If you haven't already, follow the official installation instructions for `neosr` (typically via pip or by cloning the repository and installing dependencies).

2. **Place the AetherNet Architecture File:**
   Copy the `aether_arch.py` file from this directory (`AetherNet/neosr/archs/aether_arch.py`) into your `neosr` installation's architecture directory. The standard path is:
   `path/to/your/neosr-repo/neosr/archs/`

```

# Example command to copy (adjust paths as needed):

cp AetherNet/neosr/archs/aether\_arch.py  
/path/to/your/neosr-repo/neosr/archs/

```

## Configuring AetherNet in `neosr`

Once `aether_arch.py` is in place, you can define your AetherNet model within your `neosr` configuration TOML or YAML file.

Here's an example snippet for an `aether_medium` model with a 2x upscale factor, using a TOML configuration:

```

# In your neosr config.toml (e.g., options/train/your\_aethernet\_config.toml)

# Define the global upscale factor for datasets and model output

scale = 2

# ... other dataset and training configurations ...

[network\_g]

# Specify the AetherNet variant type. This matches the function names in aether\_arch.py

type = "aether\_medium"

# You can pass specific parameters to the aether\_medium constructor directly here

# For example, to explicitly set or override defaults:

# embed\_dim = 128

# depths = [6, 6, 6, 6, 6, 6]

# mlp\_ratio = 2.0

# lk\_kernel = 11

# sk\_kernel = 3

# drop\_rate = 0.0

# drop\_path\_rate = 0.1

# img\_range = 1.0

# fused\_init = false \# Set to true if loading an already fused model for testing

# ... rest of your training configuration (losses, optimizers, etc.) ...

```

**Important Note on `scale` Parameter:**
The AetherNet model's `__init__` method in `neosr/archs/aether_arch.py` uses `upscale = upscale_opt`. The `upscale_opt` variable is automatically populated by `neosr` from the `scale` value defined at the *top level* of your configuration file. This ensures proper alignment between your global dataset settings and the model's architecture. **You do NOT need to define `upscale` again inside the `network_g` dictionary as a parameter.**

## File Provided in this Directory

* `neosr/archs/aether_arch.py`:

  * Contains the PyTorch definition of the AetherNet network (`aether` class).

  * Includes definitions for `aether_small`, `aether_medium`, and `aether_large` variants.

  * Registers these variants with `neosr`'s `ARCH_REGISTRY` for auto-detection.

  * Integrates `neosr`'s `net_opt` utility to correctly retrieve the global `scale` factor.

By following these steps, you can effectively train and evaluate your AetherNet models using the `neosr` framework, leveraging its full suite of features for high-performance image restoration.