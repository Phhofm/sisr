# AetherNet Release Models (aether_small, 2x scale, QAT)

This directory contains the release-ready models for the `aether_small` variant of AetherNet, specifically trained for 2x super-resolution with Quantization-Aware Training (QAT). These models were generated from a checkpoint at 71,000 iterations of L1-loss-only training on a dataset downsampled with lambda 1.0 (DPID downsampling).

The models here are optimized for various deployment scenarios, including direct PyTorch inference, ONNX Runtime inference, and highly efficient NVIDIA TensorRT inference.

## 📂 Contents

This folder contains the following files:

* `aether_small_2x_fused.pth`
* `aether_small_2x_notfused.pth`
* `aether_small_2x_fp32_int8.onnx`
* `aether_small_2x_fp16.onnx`

### File Details and Purpose

1.  **`aether_small_2x_notfused.pth`**
    * **Description:** This is the *original* PyTorch checkpoint (`net_g_71000.pth`) directly from the training process, renamed for clarity. It represents the `aether_small` model after 71,000 iterations of Quantization-Aware Training (QAT), but *before* the structural re-parameterization fusion has been applied to its `ReparamLargeKernelConv` modules.
    * **Purpose:** Primarily included for reference, debugging, or if you prefer to manually handle the fusion process yourself. This model is QAT-trained, meaning its weights have adapted for 8-bit quantization.
    * **Model Type:** PyTorch (`torch.nn.Module`)

2.  **`aether_small_2x_fused.pth`**
    * **Description:** This is the PyTorch model after applying the **structural re-parameterization fusion** to all `ReparamLargeKernelConv` modules. This model is directly derived from `aether_small_2x_notfused.pth` but has been transformed for optimized inference.
    * **Properties:**
        * **Network:** `aether_small`
        * **Upscale Factor:** 2x
        * **Optimization:** Fused for speed. This model will run faster than the `_notfused.pth` version within PyTorch environments due to reduced computational graph complexity.
    * **Purpose:** Ready for high-performance inference directly within PyTorch-based frameworks like [Spandrel](https://github.com/Aaron-Zeng/spandrel) or [ChaiNNer](https://github.com/chaiNNer-org/chaiNNer). This is an excellent choice for CPU inference or if you prefer to stay within the PyTorch ecosystem while benefiting from significant speed gains.
    * **Model Type:** PyTorch (`torch.nn.Module`) `state_dict`

3.  **`aether_small_2x_fp32_int8.onnx`**
    * **Description:** This is the `aether_small` model exported to **ONNX format with FP32 precision**. Crucially, because the original model was QAT-trained, this ONNX file contains **Quantize-DeQuantize (QDQ) nodes** around the operations. These nodes embed the quantization parameters (scales and zero-points) that the model learned during QAT.
    * **Properties:**
        * **Network:** `aether_small`
        * **Upscale Factor:** 2x
        * **Precision:** FP32 (but INT8-ready via QDQ nodes)
        * **Opset Version:** 17
        * **Dynamic Shapes:**
            * Batch Size: `min=1`, `opt=1`, `max=16`
            * Input Height: `min=32`, `opt=256`, `max=1080`
            * Input Width: `min=32`, `opt=256`, `max=1920`
    * **Purpose:** This is the primary input file for building highly optimized **INT8 TensorRT engines**. The embedded QDQ nodes allow TensorRT to perform precise 8-bit quantization without needing a separate calibration dataset. It can also be used for inference with ONNX Runtime.
    * **Model Type:** ONNX

4.  **`aether_small_2x_fp16.onnx`**
    * **Description:** This is the `aether_small` model exported to **ONNX format with FP16 (half-precision) floating-point**.
    * **Properties:**
        * **Network:** `aether_small`
        * **Upscale Factor:** 2x
        * **Precision:** FP16
        * **Opset Version:** 17
        * **Dynamic Shapes:** Same dynamic shape ranges as the FP32 ONNX.
    * **Purpose:** Provides a balance between precision and speed, particularly effective on hardware with Tensor Cores. It can be used for inference with ONNX Runtime.
    * **Model Type:** ONNX

---

## 🚀 Building an INT8 TensorRT Engine for Peak Performance

The `aether_small_2x_fp32_int8.onnx` file is specifically designed to be consumed by NVIDIA's TensorRT for building a highly optimized INT8 engine. This leverages the benefits of Quantization-Aware Training (QAT) to achieve maximum speed with minimal quality loss.

### Why INT8 and QAT?

* **INT8 Performance:** Running inference with 8-bit integer precision can lead to significant speedups (often 2x-4x) and reduced memory bandwidth usage compared to FP32, especially on hardware accelerators that have dedicated INT8 ALUs (e.g., Tensor Cores on NVIDIA GPUs).
* **QAT Advantage:** AetherNet was trained with QAT. This means the model learned to compensate for the effects of quantization during its training phase. The quantization parameters (scales and zero-points) are already "baked into" the model's ONNX graph via **Quantize-DeQuantize (QDQ) nodes**. This is a major advantage because:
    * **No Calibration Dataset Needed:** Unlike post-training quantization (PTQ) where you typically need to run an inference pass on a representative dataset to determine quantization parameters (calibration), with QAT-enabled ONNX, TensorRT can directly extract these parameters from the QDQ nodes.
    * **Superior Quality:** The quantization parameters are derived from the training process itself, ensuring that the model maintains higher accuracy compared to PTQ methods, which can introduce significant quality degradation.

### Using `trtexec` to Create the INT8 Engine

You can use NVIDIA's `trtexec` tool (part of the TensorRT SDK) to build the highly optimized INT8 engine from the `aether_small_2x_fp32_int8.onnx` file. The command below is optimized for speed and quality, respecting the dynamic input shapes of the ONNX model:

```bash
# Navigate to the directory containing trtexec (e.g., /usr/src/tensorrt/bin/)
# Or ensure trtexec is in your system's PATH

trtexec --onnx=./aether_small_2x_fp32_int8.onnx \
        --saveEngine=aether_small_2x_int8_dynamic.engine \
        --int8 \
        --minShapes=input:1x3x32x32 \
        --optShapes=input:1x3x256x256 \
        --maxShapes=input:16x3x1080x1920 \
        --buildOnly \
        --verbose
````

**`trtexec` Argument Breakdown (Optimized for Speed & Quality):**

  * `--onnx=./aether_small_2x_fp32_int8.onnx`: Specifies the input ONNX model. **It is crucial to use the `_fp32_int8.onnx` file here, as it contains the QDQ nodes necessary for native INT8 conversion.**
  * `--saveEngine=aether_small_2x_int8_dynamic.engine`: Defines the output path and filename for your compiled TensorRT engine. This will be the executable inference engine.
  * `--int8`: **This is the core flag that tells TensorRT to build an INT8 precision engine.** TensorRT will automatically leverage the QDQ information from the ONNX graph for precise quantization.
  * `--minShapes=input:1x3x32x32`: Defines the minimum input dimensions that the TensorRT engine will support. This should match the `--min_batch_size`, `--min_height`, and `--min_width` used during the ONNX export.
  * `--optShapes=input:1x3x256x256`: Defines the *optimal* input dimensions for the engine. TensorRT will perform its most intensive optimizations for this specific shape. **Set this to the most commonly expected inference resolution (e.g., a typical patch size for tiling, or a frequent image resolution you will process).** This should match `--opt_batch_size`, `--opt_height`, and `--opt_width` from the ONNX export.
  * `--maxShapes=input:16x3x1080x1920`: Defines the maximum input dimensions the engine can handle. This ensures the engine can scale up to common full HD resolutions (1080p width, 1920p height) without requiring tiling, if sufficient VRAM is available. This should match `--max_batch_size`, `--max_height`, and `--max_width` from the ONNX export.
  * `--buildOnly`: Instructs `trtexec` to only build the engine and not run immediate inference. This is typically desired when creating the engine for later deployment.
  * `--verbose`: Provides detailed logging during the engine building process, which is very helpful for understanding the optimization steps and for debugging any potential issues.

This process will result in a highly efficient `.engine` file that offers top-tier performance on NVIDIA GPUs, making AetherNet exceptionally fast for real-time super-resolution tasks.