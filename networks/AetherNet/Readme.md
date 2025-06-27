Of course. Here is the full content of the updated root `README.md` file, including all the Markdown syntax, presented as a plain text block for easy copying.

```markdown
# AetherNet: High-Performance Quantization-Aware Super-Resolution
<p align="right" style="color: orange; font-weight: bold;">⚠️ Under Development: Code and features may change without prior notice.</p>

## ✨ Introduction

Welcome to **AetherNet**, a novel Single Image Super-Resolution (SISR) network designed from the ground up for exceptional visual quality and unparalleled inference speed, especially on hardware accelerators. Developed by Philip Hofmann with the assistance of advanced AI, AetherNet leverages cutting-edge architectural components and a robust deployment strategy, making it ideal for real-time applications, edge devices, and scenarios demanding top-tier performance.

Our core philosophy with AetherNet is **"Quality & Speed, Quantization-Ready."** We achieve this by integrating advanced re-parameterization techniques and explicitly supporting Quantization-Aware Training (QAT), paving the way for highly optimized INT8 TensorRT engines.

## 🌌 The AetherNet Design & Strategy

AetherNet's architecture is meticulously crafted around a few key principles to achieve its goals:

1.  **Structural Re-parameterization with Large Kernels (`ReparamLargeKernelConv`):**
    * **Innovation:** We've implemented a custom re-parameterizable large kernel convolution. During training, this module effectively learns complex, global contextual information using a large receptive field (e.g., 11x11 or 13x13 kernel) in parallel with a smaller kernel (e.g., 3x3).
    * **Inference Advantage:** For deployment, these parallel branches are "fused" into a single, equivalent large kernel convolution. This process eliminates redundant computations, significantly reducing inference latency without sacrificing the rich contextual understanding gained during training. This directly translates to faster processing than many traditional large-kernel designs.

2.  **Gated Feed-Forward Networks (`GatedFFN`):**
    * **Innovation:** AetherNet incorporates Gated FFNs within its core blocks. Unlike standard FFNs, the gating mechanism provides a dynamic control over feature flow, allowing the network to selectively emphasize or attenuate information. This enhances the network's non-linear transformation capabilities and improves feature mixing, contributing to superior reconstruction quality.

3.  **Robust AetherBlock Construction:**
    * Each `AetherBlock` combines the `ReparamLargeKernelConv` and `GatedFFN`, complemented by `LayerNormalization` for stable training and `Stochastic Depth (DropPath)` for effective regularization. This holistic design ensures both powerful feature extraction and improved generalization.

4.  **Optimized Upsampling:**
    * We utilize efficient `PixelShuffle` (sub-pixel convolution) for image reconstruction, a proven method that prevents common artifacts and efficiently increases resolution.

## 🎯 Our Goal: Superior Speed & Quality with INT8 TensorRT and Optimized PyTorch Deployment

The ultimate objective of AetherNet is to deliver exceptional super-resolution performance in real-world deployment scenarios. This is achieved through a streamlined release workflow focused on achieving highly optimized models for various environments:

1.  **Quantization-Aware Training (QAT):** AetherNet is designed for QAT. During training (e.g., using `neosr`), the model is instrumented with `FakeQuantize` modules. This allows the model's weights and activations to adapt to the constraints of 8-bit quantization *during training*, minimizing quality loss compared to conventional post-training quantization.

2.  **Model Fusion:** After QAT, all `ReparamLargeKernelConv` modules are fused into their efficient single-convolution counterparts.

3.  **Optimized PyTorch Model Export:** Beyond merely an intermediate step, the conversion process directly yields a fused PyTorch `.pth` file. This is an optimized PyTorch model that benefits from the speed gains of the fusion process and can be readily utilized in other PyTorch-based inference frameworks like Spandrel or ChaiNNer. This provides a direct path to high-performance inference within the PyTorch ecosystem without requiring further conversions for CPU or GPU inference (outside of TensorRT).

4.  **ONNX Export:** The fused, QAT-trained PyTorch model is exported to ONNX format (FP32 and optionally FP16). Thanks to QAT, these ONNX models already contain the necessary Quantize-DeQuantize (QDQ) nodes, making them ready for efficient INT8 conversion.

5.  **TensorRT Engine Creation:** The exported ONNX files are then used to build highly optimized INT8 TensorRT engines. TensorRT is NVIDIA's high-performance inference optimizer and runtime, which can deliver significant speedups by leveraging hardware-specific optimizations and INT8 precision.

This end-to-end workflow ensures that AetherNet models maintain their high perceptual quality while achieving remarkable speed gains across diverse deployment targets, from standard PyTorch environments to highly specialized TensorRT engines.

## 🤝 The "Single Source of Truth" Concept

AetherNet embraces a "single source of truth" development paradigm for maximum maintainability:

* **`core/aether_core.py`:** This file contains the pure, framework-agnostic definition of the AetherNet architecture. It is the central, canonical representation of the model.

* **Framework Wrappers:** Instead of duplicating the model code, training frameworks (like `neosr`, `traiNNer-redux`) and inference frameworks (like `spandrel`) will implement lightweight wrappers that import and utilize the `aether` class directly from `aether_core.py`.

This modular approach ensures that any architectural changes or bug fixes in the core AetherNet definition only need to be applied in one place (`aether_core.py`), automatically propagating to all integrated codebases. This vastly improves maintainability, reduces development overhead, and guarantees consistency across all deployments.

## 🚀 Comparison to Other Networks

While networks like **SPAN** and **RealPLKSR** have pushed the boundaries of SISR, AetherNet distinguishes itself through:

* **Explicit Fusion for Inference:** Many large-kernel models achieve impressive results but may rely on complex multi-branch inference or configurations that don't easily fuse into a single, highly efficient operation. AetherNet's `ReparamLargeKernelConv` is purpose-built for this fusion, providing a direct path to speed without compromising kernel size benefits.

* **Integrated QAT for INT8 Excellence:** Unlike models that primarily target FP32 inference and rely on post-training quantization (which often incurs significant quality degradation), AetherNet's native QAT support ensures that the model learns to operate effectively under 8-bit precision from the outset. This translates to superior INT8 TensorRT performance, both in terms of speed and maintained visual fidelity.

* **Maintainability-First Design:** The "single source of truth" and wrapper strategy is a significant architectural improvement for long-term project health and community contributions, distinguishing it from research codebases that might be harder to adapt or maintain across different tools.

AetherNet's existence is justified by its commitment to providing a balanced solution that excels in both visual quality and **practical deployability at scale**, particularly in resource-constrained environments or high-throughput systems that benefit from INT8 optimization.

## 📋 Network Options

AetherNet offers configurable parameters to tailor its performance and size:

* `in_chans`: Number of input image channels (e.g., `3` for RGB).

* `embed_dim`: Feature dimension across the network, controlling model "width."

* `depths`: Tuple of integers, specifying the number of `AetherBlocks` in each stage, controlling model "depth."

    * **`aether_small`**: `embed_dim=96`, `depths=(4, 4, 4, 4)`

    * **`aether_medium`**: `embed_dim=128`, `depths=(6, 6, 6, 6, 6, 6)`

    * **`aether_large`**: `embed_dim=180`, `depths=(8, 8, 8, 8, 8, 8, 8, 8)`, `mlp_ratio=2.5`, `lk_kernel=13`

* `mlp_ratio`: Multiplier for the hidden dimension of the `GatedFFN` relative to `embed_dim`.

* `drop_rate`: Dropout rate within `GatedFFN`.

* `drop_path_rate`: Total stochastic depth rate, linearly distributed across blocks.

* `lk_kernel`: Large kernel size for `ReparamLargeKernelConv` (e.g., `11`, `13`).

* `sk_kernel`: Small kernel size for `ReparamLargeKernelConv` (typically `3`).

* `scale`: The super-resolution upscale factor (e.g., `2`, `3`, `4`).

* `img_range`: The maximum pixel value range for normalization (e.g., `1.0` for [0,1] input).

* `fused_init`: Boolean flag; if `True`, initializes the network directly in its fused (inference-optimized) state. Useful when loading a pre-fused checkpoint.

## License

AetherNet is released under the MIT License. See the `LICENSE` file for more details.

---
**Created by Philip Hofmann with the help of AI.**
```
