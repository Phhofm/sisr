# aethernet_to_onnx.py
# This script converts a fused AetherNet PyTorch model (.pth) into an ONNX format.
# It includes argument parsing for model configuration and an optional verification
# step to ensure the ONNX model produces similar outputs to the PyTorch model.

import torch
import torch.nn as nn
import math
import argparse
import os
from typing import Dict, Any, Tuple

try:
    import onnxruntime as ort
    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    _ONNXRUNTIME_AVAILABLE = False
    print("Warning: onnxruntime not found. ONNX verification will be skipped.")

# ==============================================================================
# AetherNet Architecture Definition (Self-Contained for this script)
# This section is a direct copy of the core AetherNet PyTorch modules,
# ensuring the script can instantiate the model without external framework dependencies.
# ==============================================================================

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Applies Stochastic Depth (DropPath) to the input tensor."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """A PyTorch module wrapper for the drop_path function."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ReparamLargeKernelConv(nn.Module):
    """
    Structural re-parameterization block. Fuses large and small kernels for inference.
    Designed to be initialized in a fused state for this script's purpose.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        
        self.fused = fused_init

        if self.fused:
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=True
            )
        else:
            # Unfused components are not expected to be used directly in this script
            # as it targets already fused models. Included for completeness of the class.
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups, bias=False
            )
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)
        
        # This branch should ideally not be taken if fused_init=True
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return lk_out + self.lk_bias.view(1, -1, 1, 1) + sk_out + self.sk_bias.view(1, -1, 1, 1)

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method to compute the fused kernel and bias."""
        if self.fused:
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")
        
        # This part should ideally not be reached if fusing an already fused model.
        # But kept for completeness of ReparamLargeKernelConv class for unfused models too.
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuses the ReparamLargeKernelConv layers in-place."""
        if self.fused:
            return # Already fused
            
        # This part should ideally not be reached if fusing an already fused model.
        fused_kernel, fused_bias = self._fuse_kernel()
        
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True
        )
        
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        
        self.__delattr__('lk_conv')
        self.__delattr__('sk_conv')
        self.__delattr__('lk_bias')
        self.__delattr__('sk_bias')
        
        self.fused = True


class GatedFFN(nn.Module):
    """Gated Feed-Forward Network with GELU activation."""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_gate = nn.Linear(in_features, hidden_features)
        self.fc1_main = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc1_gate(x)
        main = self.fc1_main(x)
        x = self.act(gate) * main
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AetherBlock(nn.Module):
    """The core building block of AetherNet."""
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0., 
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3, fused_init: bool = False):
        super().__init__()
        
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel, 
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GatedFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        x = self.ffn(x)
        
        x = x.permute(0, 3, 1, 2)
        
        x = shortcut + self.drop_path(x)
        
        return x


class Upsample(nn.Sequential):
    """Upsample module using PixelShuffle."""
    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Upscale factor {scale} is not supported.")
        super(Upsample, self).__init__(*m)


class aether(nn.Module):
    r"""
    AetherNet: A high-performance Single Image Super-Resolution (SISR) network.
    """
    def _init_weights(self, m: nn.Module):
        """Initializes weights for various module types. (Not directly used if fused_init=True)."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        depths: Tuple[int, ...] = (6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        upscale: int = 4, 
        img_range: float = 1.0, 
        fused_init: bool = False, # Key: True when loading a fused model directly
        **kwargs,
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = upscale
        self.fused_init = fused_init
        
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)
            
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur_drop_path_idx = 0
        for i_layer in range(self.num_layers):
            layer_blocks = []
            for i in range(depths[i_layer]):
                layer_blocks.append(AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur_drop_path_idx + i],
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=self.fused_init # Pass fused_init to ReparamLargeKernelConv
                ))
            self.layers.append(nn.Sequential(*layer_blocks))
            cur_drop_path_idx += depths[i_layer]
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        num_feat_upsample = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(self.upscale, num_feat_upsample)
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        if not self.fused_init:
            self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        
        x_first = self.conv_first(x)
        
        res = x_first
        for layer in self.layers:
            res = layer(res)
        
        res = res.permute(0, 2, 3, 1)
        res = self.norm(res)
        res = res.permute(0, 3, 1, 2)

        res = self.conv_after_body(res)
        res += x_first

        x = self.conv_before_upsample(res)
        x = self.conv_last(self.upsample(x))

        return x / self.img_range + self.mean
        
    def fuse_model(self):
        """
        Fuses ReparamLargeKernelConv modules. For this script, the model
        should already be fused, so this method mostly serves as a placeholder
        or for verification that the model *is* fused.
        """
        if self.fused_init:
            print("Model already initialized in a fused state. Skipping fuse_model().")
            return
            
        print("Performing in-place fusion of ReparamLargeKernelConv modules...")
        for module in self.modules():
            if isinstance(module, ReparamLargeKernelConv):
                if not module.fused:
                    module.fuse()
        self.fused_init = True
        print("Fusion complete (if performed).")


# --- Helper functions for model instantiation ---

def get_aether_model(model_type_str: str, upscale_factor: int, fused_init: bool = False, **kwargs) -> aether:
    """
    Returns an AetherNet model instance based on string identifier.
    Args:
        model_type_str (str): Type of AetherNet model ('aether_small', 'aether_medium', 'aether_large').
        upscale_factor (int): Upscale factor of the model.
        fused_init (bool): If True, initializes the model with fused convolutions.
        **kwargs: Additional arguments to pass to the aether constructor.
    Returns:
        aether: An instantiated AetherNet model.
    """
    if model_type_str == "aether_small":
        return aether(
            embed_dim=96,
            depths=(4, 4, 4, 4),
            mlp_ratio=2.0,
            upscale=upscale_factor,
            fused_init=fused_init,
            **kwargs
        )
    elif model_type_str == "aether_medium":
        return aether(
            embed_dim=128,
            depths=(6, 6, 6, 6, 6, 6),
            mlp_ratio=2.0,
            upscale=upscale_factor,
            fused_init=fused_init,
            **kwargs
        )
    elif model_type_str == "aether_large":
        return aether(
            embed_dim=180,
            depths=(8, 8, 8, 8, 8, 8, 8, 8),
            mlp_ratio=2.5,
            lk_kernel=13, # AetherNet-Large has a different default lk_kernel
            upscale=upscale_factor,
            fused_init=fused_init,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown AetherNet model type: {model_type_str}. "
                         "Supported types are 'aether_small', 'aether_medium', 'aether_large'.")

# ==============================================================================
# Main ONNX Export and Verification Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a fused AetherNet PyTorch model to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument("--input_model_path", type=str, required=True,
                        help="Path to the fused PyTorch model checkpoint (.pth).")
    parser.add_argument("--output_onnx_path", type=str, required=True,
                        help="Path to save the exported ONNX model (.onnx).")
    
    # Model Configuration Arguments
    parser.add_argument("--model_type", type=str, default="aether_medium",
                        choices=["aether_small", "aether_medium", "aether_large"],
                        help="Type of AetherNet model (e.g., 'aether_medium'). "
                             "Must match the variant used to create the fused model.")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Upscale factor of the model (e.g., 2 for 2x SR).")
    parser.add_argument("--input_channels", type=int, default=3,
                        help="Number of input image channels (e.g., 3 for RGB).")
    
    # Dynamic Shapes Arguments
    parser.add_argument("--dynamic_shapes", action="store_true",
                        help="Export ONNX model with dynamic batch size, height, and width.")
    parser.add_argument("--opt_batch_size", type=int, default=1,
                        help="Optimal batch size for ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Minimum batch size for dynamic ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--max_batch_size", type=int, default=4,
                        help="Maximum batch size for dynamic ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--opt_height", type=int, default=64,
                        help="Optimal height of the dummy low-resolution input image for ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--min_height", type=int, default=32,
                        help="Minimum height for dynamic ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--max_height", type=int, default=256,
                        help="Maximum height for dynamic ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--opt_width", type=int, default=64,
                        help="Optimal width of the dummy low-resolution input image for ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--min_width", type=int, default=32,
                        help="Minimum width for dynamic ONNX export (used with --dynamic_shapes).")
    parser.add_argument("--max_width", type=int, default=256,
                        help="Maximum width for dynamic ONNX export (used with --dynamic_shapes).")
                        
    # ONNX Export Specific Arguments
    parser.add_argument("--opset_version", type=int, default=17,
                        help="The ONNX opset version to use for export. Common values: 11, 13, 14, 17.")
    parser.add_argument("--fp_mode", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="Floating point mode for the ONNX model (FP32 or FP16). "
                             "Note: FP16 conversion may require TensorRT for optimal results.")
    
    # Verification Arguments
    parser.add_argument("--verify_onnx", action="store_true",
                        help="Perform verification by comparing PyTorch and ONNX outputs.")
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance for ONNX output verification (e.g., 1e-3 for FP16).")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for ONNX output verification (e.g., 1e-2 for FP16).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to load the PyTorch model on (e.g., 'cuda', 'cpu').")
    
    args = parser.parse_args()

    # --- Initial Checks ---
    if not os.path.exists(args.input_model_path):
        print(f"Error: Input model path '{args.input_model_path}' does not exist.")
        exit(1)

    output_dir = os.path.dirname(args.output_onnx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"--- AetherNet Fused Model to ONNX Converter ---")
    print(f"Input Model: {args.input_model_path}")
    print(f"Output ONNX: {args.output_onnx_path}")
    print(f"Model Type: {args.model_type}, Scale: {args.scale}x")
    
    if args.dynamic_shapes:
        print(f"Dynamic Input Dimensions:")
        print(f"  Batch: min={args.min_batch_size}, opt={args.opt_batch_size}, max={args.max_batch_size}")
        print(f"  Height: min={args.min_height}, opt={args.opt_height}, max={args.max_height}")
        print(f"  Width: min={args.min_width}, opt={args.opt_width}, max={args.max_width}")
    else:
        print(f"Static Input Dimensions: {args.opt_batch_size}x{args.input_channels}x{args.opt_height}x{args.opt_width}")

    print(f"FP Mode: {args.fp_mode}, Opset Version: {args.opset_version}, Device: {args.device}")

    try:
        # 1. Load the Fused PyTorch Model
        model = get_aether_model(
            model_type_str=args.model_type,
            upscale_factor=args.scale,
            in_chans=args.input_channels,
            fused_init=True # Important: Initialize in fused mode
        ).to(args.device)
        model.eval() # Set to evaluation mode

        state_dict = torch.load(args.input_model_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        print("Fused PyTorch model loaded successfully.")

        # Apply FP16 conversion if requested
        if args.fp_mode == "fp16":
            model = model.half()
            print("Model converted to FP16 precision for export.")
            dummy_input_dtype = torch.float16
            if args.atol == 1e-3: args.atol = 1e-2 # Loosen default for FP16
            if args.rtol == 1e-5: args.rtol = 1e-2 # Loosen default for FP16
            print(f"Adjusted verification tolerances for FP16: atol={args.atol}, rtol={args.rtol}")
        else:
            dummy_input_dtype = torch.float32

        # 2. Create Dummy Input (using optimal dimensions for tracing)
        dummy_input = torch.randn(
            args.opt_batch_size,
            args.input_channels,
            args.opt_height,
            args.opt_width,
            dtype=dummy_input_dtype
        ).to(args.device)

        # Define dynamic axes
        dynamic_axes = {}
        if args.dynamic_shapes:
            dynamic_axes['input'] = {0: 'batch_size', 2: 'height', 3: 'width'}
            dynamic_axes['output'] = {0: 'batch_size', 2: 'output_height', 3: 'output_width'} # Output height/width also dynamic
        
        # Flag to control saving
        should_save_onnx = True

        # 3. Verification (if enabled)
        if args.verify_onnx:
            if not _ONNXRUNTIME_AVAILABLE:
                print("ONNX verification skipped: onnxruntime not installed.")
            else:
                print("\n--- Verifying ONNX model output ---")
                
                # Run inference with PyTorch model
                torch_output = model(dummy_input).detach().cpu().numpy()
                print("PyTorch model inference complete.")

                # Temporarily export to a temp ONNX file for verification
                temp_onnx_path = args.output_onnx_path + ".tmp_verify.onnx"
                print(f"Temporarily exporting to {temp_onnx_path} for verification...")
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        dummy_input,
                        temp_onnx_path,
                        export_params=True,
                        opset_version=args.opset_version,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes=dynamic_axes,
                        verbose=False
                    )
                print("Temporary ONNX export complete for verification.")

                # Load and run inference with ONNX model
                ort_session = ort.InferenceSession(temp_onnx_path)
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
                ort_output = ort_session.run(None, ort_inputs)[0]
                print("ONNX model inference complete.")

                # Compare outputs
                if torch.allclose(torch.from_numpy(torch_output), torch.from_numpy(ort_output), 
                                  atol=args.atol, rtol=args.rtol):
                    print(f"Verification successful: PyTorch and ONNX outputs match "
                          f"(atol={args.atol}, rtol={args.rtol}).")
                else:
                    print(f"Verification FAILED: PyTorch and ONNX outputs do NOT match "
                          f"(atol={args.atol}, rtol={args.rtol}).")
                    max_diff = torch.max(torch.abs(torch.from_numpy(torch_output) - torch.from_numpy(ort_output)))
                    print(f"Max absolute difference: {max_diff.item()}")
                    should_save_onnx = False # Do NOT save if verification fails
                
                # Clean up temporary ONNX file
                if os.path.exists(temp_onnx_path):
                    os.remove(temp_onnx_path)
                    print(f"Removed temporary ONNX file: {temp_onnx_path}")
        
        # 4. Export to ONNX (final save) - only if verification passes or not requested
        if should_save_onnx:
            print(f"\nExporting PyTorch model to ONNX (opset_version={args.opset_version})...")
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    args.output_onnx_path,
                    export_params=True,
                    opset_version=args.opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            print(f"ONNX model saved to: {args.output_onnx_path}")
        else:
            print("\nONNX model NOT saved due to verification failure.")
            exit(1) # Exit with an error code if not saved

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()
