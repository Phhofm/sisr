# AetherNet: A High-Performance & Efficient SISR Network

**⚠️ DISCLAIMER: Under active development and testing. This network is not yet in a frozen state and its architecture, training strategies, or best practices may change at any time. Use for experimental purposes and be aware of potential future breaking changes. ⚠️**

## 🌌 About AetherNet

AetherNet is a novel Single Image Super-Resolution (SISR) network designed to achieve state-of-the-art perceptual quality while prioritizing computational efficiency and deployability. The name "Aether" (from ancient Greek αἰθήρ, meaning 'pure, fresh air' or 'the upper regions of space') reflects its design goals: to produce clear, pure high-resolution images efficiently, reaching for the "higher regions" of performance without excessive computational "weight."

## 🎯 Design Goals & Approach

The primary goal behind AetherNet's design is to strike an optimal balance between **super-resolution quality**, **inference speed**, and **resource efficiency**. Many state-of-the-art SISR networks achieve impressive results but often come with high computational costs, making them challenging for real-time applications or deployment on resource-constrained hardware.

AetherNet's approach addresses this by:

1. **Optimizing Receptive Field:** Achieving a broad receptive field (essential for capturing global context) without resorting to computationally expensive large kernel convolutions during inference.

2. **Efficient Feature Transformation:** Employing lightweight yet powerful modules for feature mixing and non-linearity, avoiding the quadratic complexity of traditional attention mechanisms.

3. **Deployment-Readiness:** Incorporating techniques like structural re-parameterization for easy model fusion and ensuring compatibility with industry-standard inference acceleration tools like NVIDIA TensorRT.

## 🛠️ Architectural Design & Components

AetherNet is built upon a hybrid design that strategically combines several advanced techniques:

* **ReparamLargeKernelConv:**

  * **What it is:** This module introduces a large convolutional kernel during training for a wide receptive field, crucial for comprehensive context understanding in SR. Crucially, its weights can be structurally re-parameterized and fused into a single, smaller, and highly efficient convolution for inference.

  * **Why it's used:** It allows the network to learn global image context like large-kernel convolutions, which are beneficial for image restoration, but then collapse into efficient, standard convolutions for fast runtime inference, mitigating the computational overhead typically associated with large kernels. This is a key contributor to AetherNet's efficiency.

* **Gated Feed-Forward Network (GatedFFN):**

  * **What it is:** A custom feed-forward network that incorporates a "gating" mechanism (GELU activation applied to one branch, which then multiplies another).

  * **Why it's used:** This design enhances the network's capacity for non-linear feature transformation and feature interaction, akin to a lightweight, channel-wise attention mechanism. It offers improved expressiveness and adaptivity compared to standard FFNs, without the quadratic computational cost of full self-attention often found in Transformer-based models.

* **Residual Connections with Stochastic Depth (DropPath):**

  * **What it is:** Standard skip connections that add the input of a block to its output. Stochastic Depth (often called DropPath) is a regularization technique where paths (residual branches) are randomly dropped during training.

  * **Why it's used:** Residual connections are essential for training very deep neural networks by enabling gradients to flow more easily, preventing vanishing gradient problems, and facilitating the learning of residual mappings. Stochastic Depth further improves generalization by preventing overfitting and encouraging independent feature learning across paths.

* **Layer Normalization:**

  * **What it is:** A normalization technique applied across the features of an individual sample within a layer.

  * **Why it's used:** It helps stabilize and accelerate the training process, particularly in deep networks with complex feature transformations, by normalizing inputs to each sub-layer.

* **LeakyReLU Activation:**

  * **What it is:** A rectified linear unit (ReLU) variant that allows a small, non-zero gradient when the input is negative.

  * **Why it's used:** It helps prevent the "dying ReLU" problem (where neurons can become inactive for negative inputs), ensuring that gradients can still flow, contributing to more stable and robust training, especially in generative tasks like super-resolution.

* **PixelShuffle (Sub-pixel Convolution):**

  * **What it is:** A commonly used technique for upsampling that rearranges elements from a low-resolution feature map into a higher-resolution image, effectively increasing spatial resolution.

  * **Why it's used:** It's a highly efficient and effective upsampling method that avoids checkerboard artifacts often associated with transposed convolutions, producing cleaner and sharper edges.

These design choices collectively aim to deliver a network that is both powerful in image reconstruction and practical for deployment, striking a balance that many other SR solutions struggle with.

## 📊 Network Options / Variants

AetherNet comes in several variants to cater to different performance and resource requirements:

* **`aether_small`**:

  * **Parameters:** `embed_dim=96`, `depths=(4, 4, 4, 4)`

  * **Purpose:** Lighter model, suitable for resource-constrained environments or applications where speed is paramount and a slight quality trade-off is acceptable.

* **`aether_medium`**:

  * **Parameters:** `embed_dim=128`, `depths=(6, 6, 6, 6, 6, 6)`

  * **Purpose:** The standard recommended variant, offering an excellent balance of quality and efficiency.

* **`aether_large`**:

  * **Parameters:** `embed_dim=180`, `depths=(8, 8, 8, 8, 8, 8, 8, 8)`, `lk_kernel=13`

  * **Purpose:** The largest variant for maximum quality, with a slightly larger large-kernel size (`lk_kernel`) to capture even more global context. It naturally demands more computational resources but pushes the boundaries of quality.

These variants ensure that AetherNet can be adapted to a wide array of use cases, from real-time video upscaling to high-fidelity image restoration for professional applications.

## 💡 Novelty & Strengths

AetherNet distinguishes itself from other SISR networks through:

* **Novel Fusion of Concepts:** It uniquely combines the re-parameterization of large kernel convolutions with a specifically designed Gated FFN within a residual framework, tailored for efficient and high-quality image reconstruction. This particular synergy aims to achieve superior performance-to-cost ratio.

* **Inference Efficiency by Design:** Unlike many SOTA models that achieve high quality at the expense of massive FLOPs and slow inference, AetherNet is inherently designed for fast deployment via its fusable components. This focus on post-training efficiency makes it highly competitive for real-world applications.

* **Strong Balance of Performance and Resources:** AetherNet seeks to occupy a sweet spot in the SISR landscape, offering quality comparable to or exceeding many larger models, but with significantly reduced inference times and memory footprints after fusion.

* **Deployment-Oriented Development:** The entire workflow, from training architecture to release (fusion, ONNX export, TensorRT compatibility), is structured to facilitate seamless deployment.

AetherNet is needed and important because it addresses the critical demand for **deployable, high-performance SISR solutions** that can run efficiently on modern GPUs without sacrificing too much visual quality. It represents a step towards making advanced super-resolution more accessible and practical for a wider range of applications.

## 🚀 How to Use This Network

This repository provides the necessary components to train, fuse, convert, and integrate AetherNet into various popular deep learning frameworks and inference pipelines.

The general workflow is:

1. **Train:** Train an AetherNet model using `neosr` or `traiNNer-redux`.

2. **Fuse (Recommended):** Convert the trained model to its fused (inference-optimized) state using `fuse_script`.

3. **Convert to ONNX:** Convert the fused model to the ONNX format using the `onnx` conversion script.

4. **Create TensorRT Engine:** Use the ONNX model to build a highly optimized TensorRT engine for maximum performance on NVIDIA GPUs.

Detailed instructions for each step are provided in the respective sub-directories:

* **`neosr/`**: Contains the `aether_arch.py` file for integrating AetherNet with the `neosr` framework, along with example configuration files for training.

* **`traiNNer-redux/`**: Contains the `aether_arch.py` file for integrating AetherNet with the `traiNNer-redux` framework.

* **`fuse_script/`**: Contains `fuse_aether.py` and its `Readme.md` for converting unfused AetherNet checkpoints into fused, optimized ones for faster inference. **Highly recommended before ONNX conversion.**

* **`onnx/`**: Contains `aether2onnx.py` and its `Readme.md` for converting your AetherNet model to the ONNX format. This script also supports exporting models with dynamic input shapes.

* **`spandrel/`**: Contains the necessary files (`architectures/AetherNet/__init__.py` and `architectures/AetherNet/__arch/aether_core.py`) to enable **automatic detection and loading** of AetherNet models (both fused and unfused) by the Spandrel model introspection library. Spandrel will automatically deduce the model's scale, fusion state, and network options (small, medium, large) from the checkpoint.

## 📂 Repository Structure

```

AetherNet/
├── fuse\_script/              \# Script to fuse trained AetherNet models for inference speedup
│   ├── fuse\_aether.py        \#   - The fusion script
│   └── Readme.md             \#   - Usage instructions for the fusion script
├── neosr/                    \# Integration with the neosr training framework
│   ├── archs/                \#   - Architecture definitions for neosr
│   │   └── aether\_arch.py    \#     - AetherNet architecture for neosr
│   └── options/              \#   - Example training/testing configuration files
│       ├── 2xBHI\_aether\_medium\_l1pretrain.toml
│       └── test\_aether.toml
├── onnx/                     \# Scripts for ONNX export and TensorRT compatibility
│   ├── aether2onnx.py        \#   - Script to convert .pth to .onnx
│   └── Readme.md             \#   - Usage instructions for ONNX export and TRT conversion
├── spandrel/                 \# Integration with the Spandrel model introspection library
│   └── architectures/        \#   - Spandrel's architecture detection files
│       └── AetherNet/        \#     - AetherNet specific Spandrel integration
│           ├── \_arch/        \#       - Core AetherNet definition for Spandrel
│           │   └── aether\_core.py
│           └── **init**.py   \#       - Spandrel detection logic (fused/unfused, scale, etc.)
└── traiNNer-redux/           \# Integration with the traiNNer-redux training framework
└── traiNNer/
└── archs/            \#   - Architecture definitions for traiNNer-redux
└── aether\_arch.py\#     - AetherNet architecture for traiNNer-redux

```

## 🤝 Contribution & License

This project is developed by **Philip Hofmann** with the valuable assistance of AI models.

This code is released under the **MIT License**.