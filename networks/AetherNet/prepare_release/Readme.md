# `prepare_release/`: AetherNet Model Conversion and Release Tools

This `prepare_release/` directory contains the essential script for converting your trained AetherNet PyTorch models into highly optimized, deployment-ready formats. The primary goal is to facilitate the creation of high-performance TensorRT engines, particularly for INT8 precision, to achieve superior speed and quality.

## 📂 Contents

* [`convert_aethernet.py`](convert_aethernet.py): This Python script orchestrates the conversion process.

## 🎯 Script Goal and Outputs

The `convert_aethernet.py` script serves as the central hub for preparing AetherNet models for efficient deployment. Its core goals are:

1.  **Optimize for Speed:** Convert the trained PyTorch model into formats that offer significant inference speedups, especially on dedicated AI accelerators.
2.  **Enable INT8 Deployment:** Facilitate the creation of highly efficient INT8 TensorRT engines without the need for post-training calibration datasets, thanks to Quantization-Aware Training (QAT).
3.  **Provide Flexible Deployment Options:** Output a range of optimized model formats to suit various inference environments.

Upon execution, the script produces the following optimized model files in your specified output directory:

* **Fused PyTorch `.pth`:** An optimized PyTorch checkpoint where all `ReparamLargeKernelConv` modules have been structurally re-parameterized and "fused" into single, efficient convolutional layers. This model is ready for faster inference directly within PyTorch-based frameworks like [Spandrel](https://github.com/Aaron-Zeng/spandrel) or [ChaiNNer](https://github.com/chaiNNer-org/chaiNNer).
* **FP32 ONNX (with QDQ nodes):** The fused model exported to ONNX format with 32-bit floating-point precision. Crucially, if the input `.pth` model was trained with QAT, this ONNX file will automatically include **Quantize-DeQuantize (QDQ) nodes**. These nodes explicitly mark the quantization points and parameters, making the model "INT8-ready" for TensorRT.
* **FP16 ONNX (Optional):** The fused model exported to ONNX format with 16-bit floating-point precision. This offers a balance between precision and speed, particularly beneficial on hardware with Tensor Cores.

## ⚙️ Design and Optimization Principles

The `convert_aethernet.py` script's design is driven by AetherNet's core philosophy of delivering both quality and speed, with a strong emphasis on efficient deployment:

1.  **Leveraging Structural Re-parameterization (Fusion):** The script first loads your trained AetherNet model and then explicitly calls its `fuse_model()` method. This step collapses the complex multi-branch `ReparamLargeKernelConv` into a single, computationally simpler convolutional layer. This "fusion" is a **key optimization for speed**, as it reduces memory access and arithmetic operations during inference, making the PyTorch `.pth` model and subsequent ONNX exports inherently faster.

2.  **Quantization-Aware Training (QAT) for Native INT8 Support:**
    * AetherNet is designed to be trained with QAT (enabled via `enable_qat=true` in `neosr` configs). During QAT, the model learns to operate with quantized weights and activations, adapting its parameters to minimize quality degradation when converted to 8-bit integers.
    * The `convert_aethernet.py` script directly exports the QAT-trained PyTorch model to ONNX. Because PyTorch's QAT mechanism inserts `FakeQuantize` modules, the exported FP32 ONNX model will naturally contain **Quantize-DeQuantize (QDQ) nodes**.
    * **How TensorRT Profits:** TensorRT is specifically designed to recognize and interpret these QDQ nodes. When you build an INT8 TensorRT engine from such an ONNX model, TensorRT uses the quantization parameters (scales and zero-points) embedded within the QDQ nodes. This means **no separate calibration dataset is needed for quantization** during TensorRT engine building, as the quantization parameters are directly derived from the QAT process. This streamlines the deployment workflow and ensures that the INT8 engine achieves high accuracy because the model was *trained* to be quantized.

3.  **Dynamic ONNX Shapes for Flexibility:** The script supports exporting ONNX models with dynamic input shapes (batch size, height, width). This is highly recommended for real-world applications where input image sizes may vary, allowing a single ONNX model (and subsequent TensorRT engine) to handle different resolutions efficiently without needing to recompile for each size.

## 🚀 Command Examples

To convert your trained AetherNet model, navigate to the `prepare_release/` directory and run the `convert_aethernet.py` script.

### Basic Conversion (Fixed Shape)

```bash
python convert_aethernet.py \
    --input_pth_path "path/to/your/aether_medium_qat_trained.pth" \
    --output_dir "converted_models_fixed" \
    --scale 2 \
    --network aether_medium \
    --img_size 32 \
    --opset_version 17 \
    --fp_mode fp32 \
    --atol 1e-4 --rtol 1e-3
````

*This command will create models optimized for a fixed input resolution of 32x32. Using `img_size 32` is suitable as it represents the smallest practical base resolution for tracing the computational graph, fitting well with minimal image inputs.*

### Recommended Conversion (Dynamic Shapes)

For most deployment scenarios, creating a dynamic ONNX model is highly recommended for flexibility and performance. This allows the model to efficiently handle varying input resolutions.

```bash
python convert_aethernet.py \
    --input_pth_path "path/to/your/aether_medium_qat_trained.pth" \
    --output_dir "converted_models_dynamic" \
    --scale 2 \
    --network aether_medium \
    --img_size 32 \
    --dynamic_shapes \
    --opset_version 17 \
    --fp_mode fp32 \
    --min_batch_size 1 --opt_batch_size 1 --max_batch_size 16 \
    --min_height 32 --opt_height 256 --max_height 1080 \
    --min_width 32 --opt_width 256 --max_width 1920 \
    --atol 1e-4 --rtol 1e-3
```

**Argument Explanations:**

  * `--input_pth_path`: **(Required)** Path to your `QAT-trained` PyTorch checkpoint file (e.g., typically found in `neosr/experiments/<your_exp_name>/models/net_g_xxxx.pth`). This should be a model that has undergone QAT.
  * `--output_dir`: Directory where the converted models (fused `.pth`, FP32 ONNX, FP16 ONNX) will be saved.
  * `--scale`: Upscale factor of your model (e.g., 2, 3, 4).
  * `--network`: The specific AetherNet variant (`aether_small`, `aether_medium`, `aether_large`). This helps the script instantiate the correct model architecture and its specific parameters.
  * `--img_size`: A placeholder image size for the dummy input used during ONNX tracing. If `--dynamic_shapes` is *not* used, this will be the fixed height/width. When `dynamic_shapes` is enabled, this argument is less critical as `opt_height`/`opt_width` take precedence for tracing.
  * `--dynamic_shapes`: **(Flag)** If present, the ONNX model will be exported with dynamic batch size, height, and width. **This is highly recommended.**
  * `--opset_version`: ONNX Opset version (e.g., `17`). Ensure your ONNX Runtime and TensorRT versions support this opset for compatibility.
  * `--fp_mode`: Floating-point precision for ONNX export (`fp32` or `fp16`). For INT8 TensorRT, `fp32` ONNX (with QDQ) is the standard input.
  * `--min_batch_size`, `--opt_batch_size`, `--max_batch_size`: Define the minimum, optimal (used for tracing), and maximum batch sizes for dynamic ONNX.
  * `--min_height`, `--opt_height`, `--max_height`: Define the minimum, optimal (used for tracing), and maximum input heights for dynamic ONNX. It's crucial to select optimal values that represent common inference sizes for best performance.
  * `--min_width`, `--opt_width`, `--max_width`: Define the minimum, optimal (used for tracing), and maximum input widths for dynamic ONNX.
  * `--atol`, `--rtol`: Absolute and relative tolerances for validating ONNX outputs against PyTorch. Adjust these if validation fails, especially for FP16 exports.

## 📈 Building an INT8 TensorRT Engine

After running `convert_aethernet.py`, you will have an ONNX file (e.g., `aether_medium_2x_fp32_int8.onnx` if using default naming) that is specifically prepared for TensorRT's INT8 capabilities.

### Understanding the FP32 ONNX (with QDQ) for INT8

The FP32 ONNX model generated by this script from a QAT-trained PyTorch model is `INT8-ready`. This is because PyTorch's QAT inserts `FakeQuantize` nodes into the computational graph. When exported to ONNX, these become **Quantize-DeQuantize (QDQ) nodes**. These QDQ nodes explicitly carry the quantization parameters (scaling factors and zero-points) that the model learned during QAT.

TensorRT's builder is designed to read these QDQ nodes. Instead of performing a separate, potentially less accurate, post-training quantization calibration step (which typically requires a representative dataset), TensorRT can directly extract the learned INT8 quantization information from the QDQ nodes in the ONNX graph. This ensures:

  * **Optimal Quality:** The quantization parameters come directly from the QAT process, where the model adapted to these constraints during training, leading to minimal accuracy loss.
  * **Faster Workflow:** No need to provide an additional calibration dataset during TensorRT engine building, simplifying the deployment pipeline.

### `trtexec` Command Example for INT8 Engine

You can use NVIDIA's `trtexec` tool (part of the TensorRT SDK) to build the highly optimized INT8 engine. The following example demonstrates how to build an INT8 engine that corresponds well with the dynamic ONNX models generated by the script's default values:

```bash
# Example command for building an INT8 TensorRT engine
# Assuming your output_dir was "converted_models_dynamic"
trtexec --onnx=converted_models_dynamic/aether_medium_2x_fp32_int8.onnx \
        --saveEngine=aether_medium_2x_int8_dynamic.engine \
        --int8 \
        --minShapes=input:1x3x32x32 \
        --optShapes=input:1x3x256x256 \
        --maxShapes=input:16x3x1080x1920 \
        --verbose
```

**`trtexec` Argument Explanations for Speed and Quality:**

  * `--onnx=converted_models_dynamic/aether_medium_2x_fp32_int8.onnx`: Specifies the input ONNX model. Use the FP32 ONNX model that contains the QDQ nodes.
  * `--saveEngine=aether_medium_2x_int8_dynamic.engine`: Defines the output path and filename for your compiled TensorRT engine. Use a descriptive name.
  * `--int8`: **This is the critical flag for INT8 precision.** TensorRT will leverage the QDQ information within the ONNX graph to build an INT8 engine.
  * `--minShapes=input:1x3x32x32`: Defines the minimum input dimensions the engine will support. **Align these with the `--min_batch_size`, `--min_height`, `--min_width` used during ONNX export for optimal flexibility.**
  * `--optShapes=input:1x3x256x256`: Defines the optimal input dimensions for the engine. TensorRT will heavily optimize for this shape, so **set this to the most frequently expected inference size** (e.g., 256x256, 512x512, or common video resolutions). This should align with `--opt_batch_size`, `--opt_height`, `--opt_width` from the ONNX export.
  * `--maxShapes=input:16x3x1080x1920`: Defines the maximum input dimensions the engine will support. **Align these with the `--max_batch_size`, `--max_height`, `--max_width` from the ONNX export.**
  * `--buildOnly`: Instructs `trtexec` to only build the engine and not run immediate inference.
  * `--verbose`: Provides detailed output during the engine building process, useful for debugging.

This process will generate a `.engine` file, which is a highly optimized, hardware-specific INT8 TensorRT executable. This engine can be loaded and run with extremely high performance on NVIDIA GPUs, making AetherNet suitable for the most demanding real-time super-resolution applications.
