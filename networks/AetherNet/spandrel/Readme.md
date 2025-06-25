# AetherNet Spandrel Integration

This directory contains the necessary files to seamlessly integrate AetherNet models (both unfused and fused) with the `spandrel` model introspection and loading library.

## What is Spandrel?

[Spandrel](https://github.com/sczhou/spandrel) is a powerful Python library designed for loading, inspecting, and using various deep learning models, particularly those related to image processing tasks like Super-Resolution. It aims to provide a unified interface for handling different model architectures and their checkpoints, automatically deducting model parameters and metadata.

## AetherNet and Spandrel: Effortless Model Loading

The goal of this integration is to allow `spandrel` to automatically detect, characterize, and load your AetherNet models (whether they are in their original unfused state or have been optimized via the fusion script) without requiring explicit configuration or metadata files.

### Key Features of this Integration:

1. **Automatic Fused/Unfused Detection:**

   * The provided `__init__.py` intelligently inspects the keys within an AetherNet model's state dictionary.

   * It distinguishes between models that are in their original **unfused** state (containing `lk_conv` and `sk_conv` weights) and models that have been **fused** (containing `fused_conv` weights).

   * This allows `spandrel` to load the correct `aether` model class variant with the appropriate `fused_init` parameter.

2. **Automatic Parameter Deduction:**

   * A crucial function, `_deduce_aether_params_from_state_dict` (located in `aether_core.py`), automatically infers AetherNet's architectural parameters.

   * This includes `embed_dim`, `depths` (number of blocks per layer group), `upscale` factor, input/output `in_chans`/`out_chans`, `mlp_ratio`, `lk_kernel`, `sk_kernel`, and `img_range`.

   * This means you **do not need to embed custom metadata or configuration files** within your AetherNet checkpoints for Spandrel to understand them.

3. **Simplified Loading:**

   * Once integrated, loading an AetherNet model becomes as simple as:

     ```
     import spandrel
     
     # Assuming your AetherNet model checkpoint is at this path
     model_path = "path/to/your_aethernet_model.pth" 
     
     # Spandrel will automatically detect the architecture and its parameters
     model_descriptor = spandrel.ModelLoader().load_from_file(model_path)
     
     # Access the PyTorch model
     model = model_descriptor.model
     print(f"Loaded AetherNet model: {model_descriptor.name}")
     print(f"Upscale factor: {model_descriptor.scale}")
     print(f"Tags: {model_descriptor.tags}")
     
     ```

## Installation / Setup

To enable `spandrel` to detect and load AetherNet models, you need to place the provided files in the correct directory within your `spandrel` installation.

1. **Ensure Spandrel is Installed:**
   If you haven't already, install Spandrel:

````

pip install spandrel

```

2. **Locate Spandrel's `architectures` Directory:**
This is typically found within your Python environment's `site-packages` directory where `spandrel` is installed. For example:
`path/to/your/python/env/lib/pythonX.Y/site-packages/spandrel/architectures/`

3. **Create the AetherNet Structure:**
Inside `spandrel/architectures/`, create a new folder named `AetherNet`, and then within `AetherNet`, create a sub-folder named `__arch`.

Your final directory structure should look like this:

```

path/to/your/python/env/lib/pythonX.Y/site-packages/spandrel/architectures/
└── AetherNet/
├── **init**.py               \<-- Place the AetherNet Spandrel **init**.py here
└── \_\_arch/
└── aether\_core.py        \<-- Place the AetherNet aether\_core.py here

```

4. **Copy the Files:**

* Copy the content of `spandrel/architectures/AetherNet/__init__.py` from this repository into the newly created `__init__.py` file.

* Copy the content of `spandrel/architectures/AetherNet/__arch/aether_core.py` from this repository into the newly created `aether_core.py` file.

Once these files are in place, `spandrel` will automatically discover and integrate AetherNet's architecture definitions the next time you use it.

## Files Provided in this Directory

* `architectures/AetherNet/__init__.py`:

* Defines the `AetherNetFusedArch` and `AetherNetUnfusedArch` classes.

* Registers these classes with `spandrel`.

* Contains the `detect` methods that identify whether a given state dictionary corresponds to a fused or unfused AetherNet model.

* Orchestrates the model loading process, leveraging the parameter deduction from `aether_core.py`.

* `architectures/AetherNet/__arch/aether_core.py`:

* Contains the pure PyTorch definition of the `aether` network and its sub-modules (`ReparamLargeKernelConv`, `AetherBlock`, `GatedFFN`, `Upsample`, etc.). This is the core architectural code, independent of any specific framework.

* Includes the `_deduce_aether_params_from_state_dict` function, essential for Spandrel's ability to automatically infer model configuration from a checkpoint.

This comprehensive integration ensures that AetherNet models are easily discoverable and usable within the `spandrel` ecosystem, simplifying model management and deployment workflows.