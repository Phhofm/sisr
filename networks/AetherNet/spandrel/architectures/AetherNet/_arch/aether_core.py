# aether_core.py
# This file contains the pure PyTorch AetherNet module definition
# and its sub-modules, designed to be framework-agnostic.

import math
import torch
from torch import nn
from torch.nn.init import trunc_normal_


# --- Parameter Deduction Helper (for Spandrel) ---
# This helper is part of the core model definition for Spandrel's auto-detection.
def _deduce_aether_params_from_state_dict(state_dict: dict):
    """
    Deduces AetherNet model parameters (embed_dim, depths, upscale, in/out_chans,
    mlp_ratio, lk_kernel, sk_kernel, img_range) from a given state_dict.
    This function is used by Spandrel for automatic model reconstruction.
    """
    # Deduce embed_dim from conv_first layer's output channels
    # Assumes 'conv_first.weight' is always present and the first convolution.
    embed_dim = state_dict['conv_first.weight'].shape[0]

    # Deduce input and output channels from the first and last convolutions
    in_chans = state_dict['conv_first.weight'].shape[1]
    out_chans = state_dict['conv_last.weight'].shape[0]

    # Deduce depths by counting blocks in the 'layers' ModuleList.
    # It assumes the 'layers' structure is consistent across the model,
    # and ReparamLargeKernelConv (or its fused version) are inside.
    depths = []
    # Check for both fused and unfused keys
    max_layer_group_idx = -1
    for key in state_dict.keys():
        if key.startswith('layers.'):
            try:
                parts = key.split('.')
                layer_group_idx = int(parts[1])
                max_layer_group_idx = max(max_layer_group_idx, layer_group_idx)
            except ValueError:
                continue

    if max_layer_group_idx >= 0:
        for i in range(max_layer_group_idx + 1):
            block_count_in_group = 0
            for key in state_dict.keys():
                if key.startswith(f'layers.{i}.') and ('.conv.lk_conv.weight' in key or '.conv.fused_conv.weight' in key):
                    try:
                        block_idx = int(key.split('.')[2])
                        block_count_in_group = max(block_count_in_group, block_idx + 1)
                    except ValueError:
                        continue
            if block_count_in_group > 0:
                depths.append(block_count_in_group)
    depths = tuple(depths)

    # Fallback for depths if deduction from state_dict keys is difficult or initial
    # This assumes standard AetherNet variants based on embed_dim
    if not depths and embed_dim > 0:
        if embed_dim == 96:
            depths = (4, 4, 4, 4)
        elif embed_dim == 128:
            depths = (6, 6, 6, 6, 6, 6)
        elif embed_dim == 180:
            depths = (8, 8, 8, 8, 8, 8, 8, 8)
        else:
            # Default to a safe tuple or raise error if specific depth is crucial.
            # This means model might not load correctly if arbitrary depths are used.
            print(f"Warning: Could not deduce 'depths' for embed_dim={embed_dim}. Using default (4,4,4,4).")
            depths = (4, 4, 4, 4)


    # Deduce upscale factor from the Upsample layer's first conv output
    # `upsample.0.weight` is the weight for the first conv in Upsample module
    # Its output channels are num_feat * scale^2 (for PixelShuffle)
    # The `conv_before_upsample.0.weight` output channels provide `num_feat_upsample`
    num_feat_upsample = state_dict['conv_before_upsample.0.weight'].shape[0]
    upsample_first_conv_out_channels = state_dict['upsample.0.weight'].shape[0]

    upscale = 4 # Default to 4x, common for SR models
    if num_feat_upsample > 0 and upsample_first_conv_out_channels > 0:
        ratio = upsample_first_conv_out_channels / num_feat_upsample
        if abs(ratio - 4.0) < 1e-6: # Check for 2x (4 = 2^2)
            upscale = 2
        elif abs(ratio - 9.0) < 1e-6: # Check for 3x (9 = 3^2)
            upscale = 3
        # For 4x, AetherNet uses two 2x PixelShuffles, so `upsample` is `nn.Sequential(Conv2d, PixelShuffle, Conv2d, PixelShuffle)`
        # This means `upsample.2.weight` would exist.
        elif 'upsample.2.weight' in state_dict: # Checks for the second PixelShuffle's preceding conv
             upscale = 4
        else:
            print(f"Warning: Could not precisely deduce upscale from upsample layer. Defaulting to 4. Ratio: {ratio}")
    else:
        print("Warning: Could not deduce upscale due to missing feature counts. Defaulting to 4.")


    # Deduce mlp_ratio from GatedFFN's first linear layer (fc1_gate)
    mlp_ratio = 2.0 # Default
    # Check if a GatedFFN layer exists and deduce from its dimensions
    # Assuming at least one block in the first layer group has an FFN
    if depths and 'layers.0.0.ffn.fc1_gate.weight' in state_dict:
        # fc1_gate.weight.shape[0] is hidden_features, embed_dim is input_features
        hidden_features = state_dict['layers.0.0.ffn.fc1_gate.weight'].shape[0]
        if embed_dim > 0:
            mlp_ratio = round(hidden_features / embed_dim, 1) # Round for typical ratios like 2.0, 2.5

    # lk_kernel and sk_kernel are usually fixed per variant.
    # We can infer them if aether_small/medium/large follow a consistent pattern.
    lk_kernel = 11 # Common default
    sk_kernel = 3 # Common default
    if embed_dim == 180 and depths == (8, 8, 8, 8, 8, 8, 8, 8): # Aether Large
        lk_kernel = 13 # Aether large uses a larger kernel

    # img_range is often a fixed hyperparameter, not always directly deducible from weights.
    # Assuming common value 1.0 for [0,1] or 255.0 for [0,255]
    # If the model explicitly stores `img_range` in its state_dict (e.g., as `model.img_range`),
    # you could retrieve it, but it's not standard for generic checkpoints.
    img_range = 1.0 # Default value, typical for normalized inputs

    return {
        'embed_dim': embed_dim,
        'depths': depths,
        'mlp_ratio': mlp_ratio,
        'drop_rate': 0.0,        # Usually fixed, not in state_dict
        'drop_path_rate': 0.1,   # Usually fixed, not in state_dict
        'lk_kernel': lk_kernel,
        'sk_kernel': sk_kernel,
        'upscale': upscale,
        'in_chans': in_chans,
        'out_chans': out_chans,
        'img_range': img_range,
    }

# --- Core AetherNet Modules (continued) ---
# These are identical to the versions used in neosr/traiNNer-redux aether_arch.py
# and are kept here to make this file self-contained for Spandrel integration.

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ReparamLargeKernelConv(nn.Module):
    """
    A structural re-parameterization block with a large kernel and a parallel small kernel.
    The two branches are fused into a single convolution during inference for high speed.
    `fused_init` allows direct initialization in a fused state.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, fused_init=False):
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
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups, bias=False
            )
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))
        

    def forward(self, x):
        if self.fused:
            return self.fused_conv(x)
        
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return lk_out + self.lk_bias.view(1, -1, 1, 1) + sk_out + self.sk_bias.view(1, -1, 1, 1)

    def _fuse_kernel(self):
        """Internal method to fuse the kernels and biases of the two branches."""
        if self.fused:
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")

        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        
        return fused_kernel, fused_bias

    def fuse(self):
        """
        Switch the module to inference mode by fusing the branches.
        This operation is done in-place and modifies the module's structure.
        """
        if self.fused:
            return
            
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
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_gate = nn.Linear(in_features, hidden_features)
        self.fc1_main = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        gate = self.fc1_gate(x)
        main = self.fc1_main(x)
        x = self.act(gate) * main
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AetherBlock(nn.Module):
    """
    The core building block of AetherNet.
    It uses a re-parameterized large kernel convolution for global context
    and a Gated FFN for feature mixing.
    `fused_init` allows direct initialization in a fused state.
    """
    def __init__(self, dim, mlp_ratio=2.0, drop=0., drop_path=0., lk_kernel=11, sk_kernel=3, fused_init=False):
        super().__init__()
        
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel, 
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GatedFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x)
        
        x = self.ffn(x)
        
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        
        x = shortcut + self.drop_path(x)
        
        return x


class Upsample(nn.Sequential):
    """Upsample module using PixelShuffle."""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n (e.g., 2, 4, 8)
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
    r"""AetherNet: A high-performance Single Image Super-Resolution (SISR) network.
    
    This architecture utilizes structural re-parameterization (ReparamLargeKernelConv)
    and a Gated Feed-Forward Network (GatedFFN) for efficient feature extraction.
    
    Args:
        in_chans (int): Number of input image channels (e.g., 3 for RGB).
        embed_dim (int): Feature dimension of the network.
        depths (tuple[int]): Number of AetherBlocks in each layer group.
        mlp_ratio (float): Ratio of MLP hidden features to embedding dimension.
        drop_rate (float): Dropout rate for FFN.
        drop_path_rate (float): Stochastic depth rate.
        lk_kernel (int): Large kernel size for ReparamLargeKernelConv.
        sk_kernel (int): Small kernel size for ReparamLargeKernelConv.
        upscale (int): Upscale factor (e.g., 2, 3, 4).
        img_range (float): Pixel value range (e.g., 1.0 for [0,1] input).
        fused_init (bool): If True, initializes the network with fused convolutions directly,
                           suitable for loading pre-fused models for inference.
    """
    def _init_weights(self, m):
        """Initializes weights for various module types."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def __init__(
        self,
        in_chans=3,
        embed_dim=128,
        depths=(6, 6, 6, 6),
        mlp_ratio=2.0,
        drop_rate=0.,
        drop_path_rate=0.1,
        lk_kernel=11,
        sk_kernel=3,
        upscale=4, 
        img_range=1.0, 
        fused_init=False, 
        **kwargs,
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = upscale
        self.fused_init = fused_init
        
        if in_chans == 3:
            self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
            
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
                    fused_init=self.fused_init
                ))
            self.layers.append(nn.Sequential(*layer_blocks))
            cur_drop_path_idx += depths[i_layer]
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        num_feat_upsample = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, num_feat_upsample)
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        if not self.fused_init:
            self.apply(self._init_weights)

    def forward(self, x):
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
        Call this method after loading weights to fuse the re-parameterization blocks
        for fast inference.
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
        print("Fusion complete.")

