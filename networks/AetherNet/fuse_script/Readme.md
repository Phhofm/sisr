# `fuse_aether.py`: AetherNet Model Fusion Script

This script provides a standalone utility to take a trained, **unfused** AetherNet model checkpoint and convert it into a **fused** model checkpoint. Fusing the model streamlines its architecture by combining specific layers (ReparamLargeKernelConv) into single, highly efficient convolutions, resulting in faster inference speeds without compromising performance.

## Purpose

The primary purpose of `fuse_aether.py` is to prepare a trained AetherNet model for **deployment and release**. While AetherNet can be trained in an "unfused" state (where its `ReparamLargeKernelConv` blocks consist of parallel small and large kernels), the fused version offers significant speed benefits during inference due to reduced computational graph complexity.

## When to Use It

You should use this script **after you have successfully trained your AetherNet model** (e.g., using `neosr` or `traiNNer-redux`) and before you intend to deploy it or share it as a ready-for-inference checkpoint. It transforms the training-friendly architecture into an inference-optimized one.

## What It Does

1. **Loads Trained Model:** Reads an existing AetherNet `.pth` checkpoint file. This checkpoint should be from a model trained with the original (unfused) `ReparamLargeKernelConv` architecture.

2. **Fuses Layers:** Iterates through the model's `ReparamLargeKernelConv` modules and applies a "fusion" operation. This operation mathematically combines the weights and biases of the parallel large-kernel and small-kernel convolutions into a single, equivalent large-kernel convolution.

3. **Saves Fused Model:** Saves the modified (fused) model's state dictionary to a new `.pth` file. This new file represents the optimized, faster-for-inference version of your AetherNet model.

4. **Optional Verification:** After saving, it attempts to load the newly fused model to ensure its integrity and confirm that the fusion was successful.

## How to Use It

### Prerequisites

* Python 3.x

* PyTorch (installed)

### Command-Line Arguments

The script accepts the following command-line arguments:

* `--model_path` (required): Path to your **unfused** AetherNet model checkpoint (.pth file).

* `--output_path` (required): Path where the **fused** model checkpoint (.pth file) will be saved.

* `--model_type` (optional): The variant of the AetherNet model.

  * Choices: `aether_small`, `aether_medium`, `aether_large`.

  * Default: `aether_medium`.

  * **Important:** This must match the model variant you actually trained.

* `--upscale` (optional): The upscale factor the model was trained for.

  * Choices: `2`, `3`, `4`.

  * Default: `4`.

  * **Important:** This must match the upscale factor your model was trained for.

* `--device` (optional): The device to load the model on for fusion.

  * Choices: `cuda` (if GPU available), `cpu`.

  * Default: `cuda` (if available, otherwise `cpu`).

### Examples

**1. Fusing an `aether_medium` (4x upscale) model:**

```

python fuse\_aether.py  
\--model\_path /path/to/your\_trained\_unfused\_model.pth  
\--output\_path ./fused\_aether\_medium\_x4.pth  
\--model\_type aether\_medium  
\--upscale 4  
\--device cuda

```

**2. Fusing an `aether_small` (2x upscale) model on CPU:**

```

python fuse\_aether.py  
\--model\_path /path/to/your\_trained\_aether\_small\_x2.pth  
\--output\_path ./fused\_aether\_small\_x2\_cpu.pth  
\--model\_type aether\_small  
\--upscale 2  
\--device cpu

```

**3. Fusing an `aether_large` (3x upscale) model:**

```

python fuse\_aether.py  
\--model\_path /path/to/my\_aether\_large\_x3.pth  
\--output\_path ./release\_aether\_large\_x3\_fused.pth  
\--model\_type aether\_large  
\--upscale 3
\# Device defaults to cuda if available

```

After running the script, a new `.pth` file containing the fused AetherNet model will be saved at the specified `output_path`. This model is now optimized for faster inference!