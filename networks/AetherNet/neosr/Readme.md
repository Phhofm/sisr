```markdown
# `neosr/`: AetherNet Integration for Training and Validation

This `neosr/` directory contains the necessary files to integrate the AetherNet super-resolution network into the `neosr` training framework. `neosr` is a powerful and flexible framework for image restoration tasks, providing robust tools for data handling, model training, logging, and evaluation.

## 📂 Directory Structure within `neosr/`

* `archs/aether_arch.py`: This file acts as the `neosr`-specific wrapper for AetherNet. It imports the core AetherNet definition from `../common/aether_core.py` and registers it with `neosr`'s architecture registry, making it discoverable for training configurations. It also includes the `prepare_qat` and `convert_to_quantized` methods specific to PyTorch's quantization utilities.
* `common/`: This is intended to be a directory containing `aether_core.py`.
* `train.py`: The modified `neosr` training script. It includes adjustments to correctly add the `common/` directory to `sys.path` to enable direct imports of `aether_core.py`, and integrates the Quantization-Aware Training (QAT) preparation step into the training loop if `enable_qat` is set in the configuration.
* `options/`: Contains example configuration files (`.toml`) for training and testing AetherNet models within `neosr`.
    * `2xBHI_aether_small_l1pretrain.toml`
    * `2xBHI_aether_medium_l1pretrain.toml`
    * `2xBHI_aether_large_l1pretrain.toml`
    * `test_aether.toml`

## ⚙️ AetherNet-neosr Integration Details

To use AetherNet within your `neosr` project (e.g., a fork like `neosr-fork`), follow these steps:

### Step 1: Place `aether_core.py` in `common/`

The AetherNet design uses `aether_core.py` as a single source of truth. To integrate it into your `neosr` project, create a `common/` directory at the **root level of your `neosr` project** (i.e., at the same level as `neosr/` and `prepare_release/`) and place the `aether_core.py` file inside it.

Your project structure should look similar to this:

```

your-neosr-project-root/
├── common/
│   └── aether\_core.py
├── neosr/
│   ├── archs/
│   │   └── aether\_arch.py
│   ├── options/
│   │   ├── 2xBHI\_aether\_small\_l1pretrain.toml
│   │   ├── 2xBHI\_aether\_medium\_l1pretrain.toml
│   │   ├── 2xBHI\_aether\_large\_l1pretrain.toml
│   │   └── test\_aether.toml
│   └── train.py  (modified)
├── prepare\_release/
│   └── convert\_aethernet.py
└── README.md (root)

````

### Step 2: Add `aether_arch.py`

Place the provided `aether_arch.py` file into your `neosr/archs/` directory. This file acts as the wrapper that connects AetherNet to `neosr`'s internal systems.

### Step 3: Modify `neosr/train.py`

The `neosr/train.py` script needs a small modification to ensure it can find and import `aether_core.py` from the `common/` directory. Locate the `if __name__ == "__main__":` block at the end of your `neosr/train.py` and ensure it includes the following lines to add the `common` directory to Python's system path:

```python
# Excerpt from neosr/train.py (ensure these lines are present)

if __name__ == "__main__":
    # Determine the absolute path of the current file (train.py)
    current_file_abs_path = Path(__file__).resolve()
    
    # The project root (e.g., 'neosr-fork/') is the parent directory of train.py
    project_root_directory = current_file_abs_path.parent
    
    # The 'common' directory is expected to be a direct subfolder of the project root
    common_dir_path = project_root_directory / "common"

    # Add the 'common' directory to Python's system path if it's not already there.
    # This is crucial for importing modules from the 'common' directory.
    if str(common_dir_path) not in sys.path:
        sys.sys.path.insert(0, str(common_dir_path)) # Corrected line: sys.path.insert
        print(f"Added '{common_dir_path}' to sys.path for common modules (explicitly added from train.py).")

    # Start the main training pipeline, passing the absolute project root path.
    train_pipeline(str(project_root_directory))
````

This change is crucial for `aether_arch.py` to successfully import `aether` from `common.aether_core`.

### Step 4: Use Provided Configuration Files

Copy the provided TOML configuration files (e.g., `2xBHI_aether_medium_l1pretrain.toml`, `test_aether.toml`) into your `neosr/options/` directory. These files are pre-configured to use the AetherNet architecture and to enable Quantization-Aware Training (QAT) where appropriate.

#### To start training with QAT:

Navigate to the root of your `neosr` project (where `train.py` is located) and run:

```bash
python neosr/train.py -opt neosr/options/2xBHI_aether_medium_l1pretrain.toml
```

Replace `2xBHI_aether_medium_l1pretrain.toml` with the specific AetherNet variant you wish to train (e.g., `aether_small` or `aether_large`). Ensure that `enable_qat = true` is set in your chosen training configuration file to leverage QAT.

#### To run testing/inference:

Similarly, from the root of your `neosr` project, run:

```bash
python neosr/test.py -opt neosr/options/test_aether.toml
```

Remember to update the `pretrain_network_g` path in `test_aether.toml` to point to your actual fused PyTorch model checkpoint. Setting `fused_init = true` in `test_aether.toml` will initialize the model directly in its inference-optimized state.

```
```
