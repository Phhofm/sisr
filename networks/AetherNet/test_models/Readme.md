# AetherNet Test Models

This folder contains a set of very simple test models for AetherNet. These models are **NOT intended for high-quality super-resolution inference** but are provided purely for **code testing, experimentation, and playing around** with the AetherNet architecture and its associated scripts (fusion, ONNX conversion, Spandrel integration, etc.).

## Characteristics of These Models:

* **Model Variant:** All models provided here are `aether_medium` variants.

* **Training:** They were trained using **L1 loss only** for a very short duration of **20,000 iterations**.

* **Patch Size:** Training was performed exclusively on **32x32 pixel patches**.

* **Status:** Due to the limited training, these models are **undertrained** and will produce sub-optimal super-resolution results.

## Files Included:

* `fused_net_g_20000.onnx`: An ONNX representation of the fused model.

* `fused_net_g_20000.pth`: The PyTorch checkpoint of the fused model.

* `net_g_20000.pth`: The PyTorch checkpoint of the unfused model (the raw output from training).

## Future Plans:

More robust and higher-quality AetherNet models, trained for longer durations and with more advanced configurations, will be released in the future. These current models serve solely as a starting point for exploring the AetherNet codebase.