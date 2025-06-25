# sisr

This repository serves as a collection of my self-created resources, scripts, and tools related to Single Image Super-Resolution (SISR). It's designed to house various components of my SISR work, *excluding* the trained models themselves (which reside in a separate repository).

## Table of Contents

* [Scripts](#scripts)

* [Networks](#networks)

* [Datasets](#datasets)

* [Workflows](#workflows)

* [Guides](#guides)

* [Related Projects](#related-projects)

## Scripts

This folder contains a variety of Python scripts and utility tools for SISR-related tasks, including:

* **Downscaling:** Scripts for generating lower-resolution versions of images (e.g., using DPID for bicubic downscaling).

* **Dataset Filtering:** Tools for cleaning, preparing, and filtering datasets.

* **Complexity Calculation:** Scripts for analyzing image and model complexity.

* **Metrics Calculation:** Implementations for calculating common SISR evaluation metrics (e.g., PSNR, SSIM, LPIPS).

* **Multiscaling:** Utilities for handling multi-scale image processing.

* And more!

## Networks

This directory is dedicated to self-developed and experimental Single Image Super-Resolution network architectures.

* **AetherNet:** An example of a custom network architecture developed for SISR.

## Datasets

This section will contain information about and links to various datasets used for SISR training and validation. It may include:

* Links to public SISR datasets.

* Descriptions and structures of custom training or validation datasets.

## Workflows

Here, you'll find custom workflows for various image processing tools that support SISR, such as:

* [Chainner](https://github.com/joeyballentine/Chainner) workflows.

* [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows.

## Guides

This section will host various guides, tutorials, and notes related to SISR, covering topics from dataset preparation to model evaluation and deployment.

## Related Projects

While this repository focuses on the tools and resources for SISR, my self-trained SISR models are hosted separately:

* [**models**](https://github.com/Phhofm/models): My primary repository for self-trained SISR models, organized by GitHub releases.

Additionally, for historical context, I previously maintained an upscaling website:

* **Upscale Site**: An older, unmaintained website (last updated May 2023) that served as an early exploration into upscaling. While outdated, it might still offer some useful insights for certain users.
