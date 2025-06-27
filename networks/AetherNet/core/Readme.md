# `core/`: AetherNet's Single Source of Truth

This `core/` directory houses the foundational and pure PyTorch implementation of the AetherNet super-resolution architecture. It contains one crucial file:

* [`aether_core.py`](aether_core.py): This Python file defines the `aether` PyTorch `nn.Module` class, along with all its essential, framework-agnostic sub-modules and utility functions.

## 🎯 Design Purpose and Single Source of Truth

The `aether_core.py` file is meticulously designed to serve as the **single, canonical source of truth** for the entire AetherNet model architecture. This strategic design choice is fundamental to AetherNet's maintainability and consistency across various deployments:

1.  **Framework Agnostic Purity:** `aether_core.py` is intentionally kept free of any framework-specific imports (e.g., `neosr`, `traiNNer-redux`, `spandrel`). This ensures that the core model definition remains pristine and independent, preventing coupling issues and making it inherently portable.
2.  **Consistency Across Integrations:** By having a single, authoritative definition, the exact same AetherNet model (including its internal components like `ReparamLargeKernelConv`, `GatedFFN`, `AetherBlock`, `DropPath`, and `Upsample`) is guaranteed to be used across all training frameworks and inference pipelines. This eliminates discrepancies that can arise from duplicating model code.
3.  **Enhanced Maintainability:** Any architectural change, bug fix, or new feature development related to the core AetherNet model only needs to be implemented and updated in `aether_core.py`. This central modification automatically propagates to all integrated codebases, drastically reducing maintenance overhead and the potential for errors.
4.  **Clear Separation of Concerns:** This structure clearly distinguishes the model's fundamental architectural definition from framework-specific training loops, data loading mechanisms, or inference handling logic. This modularity improves code readability, testability, and overall project organization.

## 💡 Key Implementations within `aether_core.py`

* **`aether` Class:** The main `torch.nn.Module` defining the AetherNet. It orchestrates the flow of data through shallow feature extraction, a deep body of `AetherBlock`s, and final upsampling and reconstruction layers. It also includes methods for pixel normalization/denormalization.
* **`ReparamLargeKernelConv`:** Implements structural re-parameterization, allowing a large kernel convolution and a parallel small kernel to be trained together, then fused into a single, efficient convolution for inference. This is key to achieving high speed without sacrificing large receptive fields.
* **`GatedFFN`:** A novel feed-forward network with a gating mechanism, enhancing non-linear feature transformations and feature mixing within the network blocks.
* **`AetherBlock`:** The core repeatable building block, combining `ReparamLargeKernelConv`, `GatedFFN`, `LayerNorm`, and `DropPath` for robust feature learning.
* **`DropPath`:** A utility for Stochastic Depth regularization, improving model robustness and preventing overfitting.
* **`Upsample`:** Efficient upsampling via PixelShuffle, ensuring high-quality image reconstruction.
* **Quantization-Aware Training (QAT) Methods (`prepare_qat`, `convert_to_quantized`):** Integrated directly into the `aether` class, these methods facilitate training the model to be robust to quantization. They insert `FakeQuantize` modules during QAT and convert them to true quantized operations for deployment, enabling high-performance INT8 inference with minimal quality loss.

## Usage by Wrappers

Other directories in this repository (e.g., `neosr/`, `spandrel/`, `traiNNer-redux/`) will contain framework-specific "wrapper" files. These wrappers will directly import the `aether` class from `aether_core.py` and adapt it to their respective ecosystems.

For example, a `neosr` wrapper might look something like:

```python
# neosr/archs/aether_arch.py
from common.aether_core import aether as AetherNetCore # assuming 'common' is added to sys.path

# Then register AetherNetCore with neosr's ARCH_REGISTRY
# and adapt its parameters to neosr's config system.
@ARCH_REGISTRY.register()
class aether(AetherNetCore):
    # ... neosr specific initialization and overrides ...
````

By adhering to this pattern, we ensure that the AetherNet model remains consistent, robust, and easy to manage throughout its lifecycle.

## License

AetherNet's core implementation (`aether_core.py`) is released under the MIT License. See the `LICENSE` file in the project root for more details.

-----

**Created by Philip Hofmann with the help of AI.**
