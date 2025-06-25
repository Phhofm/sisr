# `aether2onnx.py`: AetherNet to ONNX Conversion Script

This script converts a trained and optionally **fused** AetherNet PyTorch model (`.pth`) into the ONNX format. This is a crucial step for deploying your AetherNet model with high-performance inference engines like NVIDIA TensorRT.

## Purpose

The primary purpose of `aether2onnx.py` is to create an intermediate ONNX representation of your AetherNet model. This ONNX file can then be used by various inference runtimes, most notably NVIDIA TensorRT, to achieve highly optimized and fast inference on NVIDIA GPUs.

## When to Use It

You should use this script:

1. **After your AetherNet model has been trained and optionally fused.** It's highly recommended to fuse your AetherNet model first using `fuse_aether.py` before converting to ONNX for TensorRT deployment, as this will result in a faster and more efficient TensorRT engine.

2. **Before creating a TensorRT engine.** The ONNX file serves as the input graph definition for `trtexec` or TensorRT's Python/C++ APIs.

## What It Does

The `aether2onnx.py` script performs the following key functions:

1. **Loads AetherNet Model:** It loads your AetherNet model from a PyTorch `.pth` checkpoint. It expects the model to be in a *fused* state if you intend to optimize it fully with TensorRT.

2. **Supports Dynamic Shapes (Recommended):** You can export the ONNX model with dynamic batch size, height, and width axes. This allows the ONNX model (and subsequently, the TensorRT engine) to handle variable input image sizes without recompilation, providing greater flexibility in deployment.

3. **FP16/FP32 Export:** You can specify whether to export the ONNX model in FP32 (full precision) or FP16 (half precision). While FP16 ONNX files are smaller, the final precision for TensorRT is primarily controlled during the TensorRT engine build.

4. **ONNX Graph Tracing:** It uses PyTorch's `torch.onnx.export` to trace the model's computation graph and save it in the ONNX format.

5. **Optional Verification:** It can optionally perform a verification step by comparing the outputs of the PyTorch model and the exported ONNX model (using ONNX Runtime). This helps ensure that the conversion process did not introduce significant numerical discrepancies.

## How to Use It

### Prerequisites

* Python 3.x

* PyTorch (installed)

* `onnxruntime` (for optional verification): `pip install onnxruntime`

### Command-Line Arguments

* `--input_model_path` (required): Path to your AetherNet PyTorch model checkpoint (`.pth`). This should ideally be a **fused** model.

* `--output_onnx_path` (required): Path where the exported ONNX model (`.onnx`) will be saved.

* `--model_type` (optional): The variant of the AetherNet model.

  * Choices: `aether_small`, `aether_medium`, `aether_large`.

  * Default: `aether_medium`.

  * **Important:** This must match the model variant used during training and fusion.

* `--scale` (optional): The upscale factor the model was trained for.

  * Choices: `1`, `2`, `3`, `4`.

  * Default: `4`.

  * **Important:** This must match the upscale factor your model was trained for.

* `--input_channels` (optional): Number of input image channels (e.g., `3` for RGB). Default: `3`.

* `--dynamic_shapes` (optional flag): If set, the ONNX model will be exported with dynamic batch size, height, and width. **Highly recommended for deployment flexibility.**

* `--opt_batch_size` (optional, with `--dynamic_shapes`): Optimal batch size for ONNX export tracing. Default: `1`.

* `--min_batch_size` (optional, with `--dynamic_shapes`): Minimum batch size for dynamic ONNX export. Default: `1`.

* `--max_batch_size` (optional, with `--dynamic_shapes`): Maximum batch size for dynamic ONNX export. Default: `4`.

* `--opt_height` (optional, with `--dynamic_shapes`): Optimal low-resolution height for ONNX tracing. Default: `64`.

* `--min_height` (optional, with `--dynamic_shapes`): Minimum low-resolution height for dynamic ONNX export. Default: `32`.

* `--max_height` (optional, with `--dynamic_shapes`): Maximum low-resolution height for dynamic ONNX export. Default: `256`.

* `--opt_width` (optional, with `--dynamic_shapes`): Optimal low-resolution width for ONNX tracing. Default: `64`.

* `--min_width` (optional, with `--dynamic_shapes`): Minimum low-resolution width for dynamic ONNX export. Default: `32`.

* `--max_width` (optional, with `--dynamic_shapes`): Maximum low-resolution width for dynamic ONNX export. Default: `256`.

* `--opset_version` (optional): The ONNX opset version to use for export. Default: `17`.

* `--fp_mode` (optional): Floating-point precision for the ONNX model.

  * Choices: `fp32`, `fp16`.

  * Default: `fp32`.

  * **Note:** While you can export an FP16 ONNX, TensorRT's `--fp16` flag during engine creation is the primary control for FP16 inference.

* `--verify_onnx` (optional flag): If set, compares PyTorch and ONNX Runtime outputs for numerical similarity. Requires `onnxruntime`.

* `--atol` (optional, with `--verify_onnx`): Absolute tolerance for verification. Default: `1e-3`.

* `--rtol` (optional, with `--verify_onnx`): Relative tolerance for verification. Default: `1e-5`.

* `--device` (optional): Device to perform conversion on. Default: `cuda` (if available), `cpu` (otherwise).

### Examples

**1. Exporting a Static ONNX Model (Fixed Input Size):**

```

python aether2onnx.py  
\--input\_model\_path /path/to/fused\_aether\_medium\_x2.pth  
\--output\_onnx\_path ./aether\_medium\_x2\_static.onnx  
\--model\_type aether\_medium  
\--scale 2  
\--opt\_height 128 --opt\_width 128  
\--fp\_mode fp32  
\--opset\_version 17  
\--verify\_onnx  
\--device cuda

```

**2. Exporting a Dynamic ONNX Model (Recommended):**

This example uses conservative values for a 6GB VRAM GPU (like GTX 1660 SUPER). Adjust `max_` values based on your GPU memory and typical high-resolution needs.

```

python aether2onnx.py  
\--input\_model\_path /path/to/fused\_aether\_medium\_x2.pth  
\--output\_onnx\_path ./aether\_medium\_x2\_dynamic.onnx  
\--model\_type aether\_medium  
\--scale 2  
\--dynamic\_shapes  
\--min\_batch\_size 1 --opt\_batch\_size 1 --max\_batch\_size 2  
\--min\_height 32 --opt\_height 128 --max\_height 256  
\--min\_width 32 --opt\_width 128 --max\_width 256  
\--fp\_mode fp32  
\--opset\_version 17  
\--verify\_onnx  
\--device cuda

```

## Next Step: Creating a TensorRT Engine

The `.onnx` file generated by this script is the input for creating a highly optimized NVIDIA TensorRT engine. The `trtexec` tool, provided with the TensorRT SDK, is commonly used for this.

**Crucial Note for Dynamic Shapes:** When creating a TensorRT engine from a dynamic ONNX model, you **MUST** specify the same (or compatible) dynamic input ranges using `trtexec`'s `--minShapes`, `--optShapes`, and `--maxShapes` flags.

**Example `trtexec` Command (for the dynamic ONNX created above):**

```

/usr/src/tensorrt/bin/trtexec  
\--onnx=./aether\_medium\_x2\_dynamic.onnx  
\--saveEngine=./aether\_medium\_x2\_dynamic\_fp32.engine  
\--fp32  
\--minShapes=input:1x3x32x32  
\--optShapes=input:1x3x128x128  
\--maxShapes=input:2x3x256x256  
\--workspace=4096  \# Allocate 4GB workspace (adjust based on GPU memory)

```

**Key `trtexec` Flags:**

* `--onnx=<path_to_onnx>`: Specifies your input ONNX model.

* `--saveEngine=<path_to_engine>`: Specifies where to save the generated TensorRT engine.

* `--fp32` / `--fp16` / `--int8`: Sets the precision mode for the TensorRT engine. Choose one. For best performance on modern NVIDIA GPUs, try `--fp16`.

* `--minShapes=<input_name>:<min_dims>`: Minimum dimensions for dynamic inputs.

* `--optShapes=<input_name>:<opt_dims>`: Optimal dimensions for dynamic inputs (TensorRT optimizes most for this).

* `--maxShapes=<input_name>:<max_dims>`: Maximum dimensions for dynamic inputs.

* `--workspace=<MB>`: Maximum GPU memory TensorRT can use during engine building. Increase if you encounter out-of-memory errors during build.

After this step, you will have a `.engine` file that can be loaded directly by TensorRT for highly optimized inference.