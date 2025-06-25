# AetherNet traiNNer-redux Integration

This directory contains the necessary files to integrate AetherNet models for training and evaluation within the `traiNNer-redux` framework.

## What is `traiNNer-redux`?

`traiNNer-redux` is a powerful and flexible PyTorch-based training framework primarily designed for image restoration tasks, including Single Image Super-Resolution (SISR). It provides a structured environment for defining datasets, models, losses, and training schedules, making it a popular choice for researchers and developers in the field.

## AetherNet and `traiNNer-redux`: Seamless Training

Integrating AetherNet with `traiNNer-redux` allows you to leverage the framework's robust training pipeline, extensive loss functions, optimization strategies, and logging capabilities to effectively train your AetherNet models.

### Key Aspects of this Integration:

1. **Modular Architecture Definition:**

   * The `aether_arch.py` file defines the AetherNet architecture (`aether` class and its variants like `aether_small`, `aether_medium`, `aether_large`).

   * These models are registered with `traiNNer-redux`'s `ARCH_REGISTRY`, allowing them to be instantiated simply by name in your configuration files.

2. **Configuration-Driven Training:**

   * `traiNNer-redux` uses YAML configuration files to define all aspects of a training run (datasets, network architecture, loss functions, optimizers, schedulers, validation, logging, etc.).

   * You can specify the AetherNet variant, `embed_dim`, `depths`, `mlp_ratio`, `lk_kernel`, `sk_kernel`, `drop_rate`, `drop_path_rate`, `img_range`, and crucially, the `upscale` factor directly within your `traiNNer-redux` YAML configuration.

3. **Automatic Upscale Handling:**

   * The AetherNet architecture provided for `traiNNer-redux` is designed to correctly accept the `scale` parameter from your training configuration. This ensures that the model's upsampling layers are correctly initialized for your target super-resolution factor (e.g., 2x, 3x, 4x).

4. **Support for Fused Initialization (for Inference/Loading):**

   * The AetherNet implementation includes a `fused_init` parameter. While typically `false` for training (as training happens on unfused models), it's important for loading already fused models for testing or inference within `traiNNer-redux`. This allows you to test the performance of your fused checkpoints directly using `traiNNer-redux`'s validation scripts.

### Benefits of Using AetherNet with `traiNNer-redux`:

* **Established Training Pipeline:** Leverage `traiNNer-redux`'s mature and optimized training loop, including multi-GPU support, mixed-precision training, and efficient data loading.

* **Rich Feature Set:** Access a wide array of built-in loss functions (e.g., L1, perceptual, GAN), optimizers, and scheduling strategies.

* **Reproducibility:** YAML configuration files make it easy to reproduce experiments and share training setups.

* **Community & Tools:** Benefit from `traiNNer-redux`'s community support and existing tooling for data preparation and result analysis.

## Installation / Setup

To integrate AetherNet into your `traiNNer-redux` environment:

1. **Ensure `traiNNer-redux` is Installed:**
   Follow the official installation instructions for `traiNNer-redux` if you haven't already.

2. **Place the AetherNet Architecture File:**
   Copy the `aether_arch.py` file from this directory into your `traiNNer-redux` installation's architecture directory. The typical path is:
   `path/to/your/traiNNer-redux-repo/traiNNer/archs/`

```

# Example command to copy (adjust paths as needed):

cp AetherNet/traiNNer-redux/traiNNer/archs/aether\_arch.py  
/path/to/your/traiNNer-redux/traiNNer/archs/

```

## Configuring AetherNet in `traiNNer-redux`

Once `aether_arch.py` is in place, you can define your AetherNet model within your `traiNNer-redux` configuration YAML file.

Here's an example snippet for an `aether_medium` model with a 2x upscale factor:

```

# In your traiNNer-redux config.yml (e.g., options/train/your\_config.yml)

# Define the global upscale factor for datasets and models

scale: 2

# ... other dataset and training configurations ...

network\_g:

# Specify the AetherNet variant type. This matches the function names in aether\_arch.py

type: aether\_medium

# You can pass specific parameters to the aether\_medium constructor directly here

# For example, to explicitly set or override defaults:

# embed\_dim: 128

# depths: [6, 6, 6, 6, 6, 6]

# mlp\_ratio: 2.0

# lk\_kernel: 11

# sk\_kernel: 3

# drop\_rate: 0.0

# drop\_path\_rate: 0.1

# img\_range: 1.0

# fused\_init: false \# Set to true if loading an already fused model for testing

```

**Note on `upscale` parameter:** The AetherNet model's `__init__` method in `traiNNer-redux/traiNNer/archs/aether_arch.py` is designed to automatically pick up the global `scale` defined at the top level of your `traiNNer-redux` configuration file. This ensures proper alignment between your dataset's scale and the model's internal upsampling mechanism.

## File Provided in this Directory

* `traiNNer/archs/aether_arch.py`:

  * Contains the PyTorch definition of the AetherNet network (`aether` class).

  * Includes definitions for `aether_small`, `aether_medium`, and `aether_large` variants.

  * Registers these variants with `traiNNer-redux`'s `ARCH_REGISTRY` for auto-detection.

  * Manages parameter passing from `traiNNer-redux` configurations, including the critical `upscale` factor and `fused_init` flag.

By following these steps, you can effectively train and evaluate your AetherNet models using the `traiNNer-redux` framework.