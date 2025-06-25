# __init__.py for spandrel/architectures/AetherNet/
# This file defines the spandrel.Architecture classes for AetherNet
# and registers them with spandrel, enabling automatic detection and loading.

from typing import Literal, Type, Any

# Import the core AetherNet module (relative import from __arch sub-directory)
from .__arch.aether_core import aether, _deduce_aether_params_from_state_dict 

# Import necessary spandrel components
from spandrel import Architecture, ImageModelDescriptor, register_arch, StateDict


# --- Helper function to determine fused state from state_dict ---

def _has_fused_keys(state_dict: StateDict) -> bool:
    """
    Checks if the state_dict contains keys typical of a fused AetherNet model.
    A fused model has 'fused_conv.weight' and 'fused_conv.bias' keys within its
    ReparamLargeKernelConv layers, but *not* the 'lk_conv', 'sk_conv', 'lk_bias', 'sk_bias' keys.
    """
    has_fused_candidate = False
    has_unfused_candidate = False
    
    for key in state_dict.keys():
        if '.conv.fused_conv.weight' in key:
            has_fused_candidate = True
        if '.conv.lk_conv.weight' in key or '.conv.sk_conv.weight' in key:
            has_unfused_candidate = True
        
        # If both fused and unfused keys exist, it's a problematic state_dict,
        # likely not a fully fused or unfused model.
        if has_fused_candidate and has_unfused_candidate:
            return False 

    # A model is considered fully fused if it has fused keys and no unfused keys.
    return has_fused_candidate and not has_unfused_candidate


# --- Common Base Architecture for AetherNet in Spandrel ---

class AetherNetCommonArch(Architecture[aether]):
    """
    Common base class for AetherNet Spandrel architectures.
    Handles shared deduction logic and model creation for both fused and unfused variants.
    """
    tags = ("AetherNet", "Super-Resolution") # Base tags for all AetherNet models
    
    @staticmethod
    def _create_model_and_load_state_dict(state_dict: StateDict, fused_init: bool) -> aether:
        """
        Creates an AetherNet model instance with deduced parameters and loads the state_dict.
        """
        # Deduce parameters using the helper function from aether_core.py
        parameters = _deduce_aether_params_from_state_dict(state_dict)
        parameters['fused_init'] = fused_init # Set the fused_init flag
        
        # Instantiate the aether model with deduced and explicit parameters
        model = aether(
            in_chans=parameters['in_chans'],
            embed_dim=parameters['embed_dim'],
            depths=parameters['depths'],
            mlp_ratio=parameters['mlp_ratio'],
            drop_rate=parameters['drop_rate'],
            drop_path_rate=parameters['drop_path_rate'],
            lk_kernel=parameters['lk_kernel'],
            sk_kernel=parameters['sk_kernel'],
            upscale=parameters['upscale'],
            img_range=parameters['img_range'],
            fused_init=parameters['fused_init']
        )
        
        model.load_state_dict(state_dict, strict=True)
        return model, parameters


# --- Fused AetherNet Architecture for Spandrel ---

@register_arch
class AetherNetFusedArch(AetherNetCommonArch):
    """
    Spandrel Architecture definition for a FUSED AetherNet model.
    This class is responsible for detecting and loading fused AetherNet models.
    """
    id = "AetherNet-Fused"
    name = "AetherNet (Fused)"
    
    @staticmethod
    def detect(state_dict: StateDict) -> bool:
        """
        Detects if the given state_dict belongs to a fused AetherNet model.
        It checks for the presence of fused convolution keys and absence of unfused keys.
        """
        return _has_fused_keys(state_dict)

    @staticmethod
    def load(state_dict: StateDict) -> ImageModelDescriptor[aether]:
        """
        Loads a fused AetherNet model from the state_dict.
        Initializes the model directly in its fused state.
        """
        model, parameters = AetherNetCommonArch._create_model_and_load_state_dict(state_dict, fused_init=True)
        model.eval() # Ensure model is in evaluation mode
        
        # Determine model variant for tags (small, medium, large)
        variant_tag = ""
        if parameters['embed_dim'] == 96 and parameters['depths'] == (4, 4, 4, 4):
            variant_tag = "small"
        elif parameters['embed_dim'] == 128 and parameters['depths'] == (6, 6, 6, 6, 6, 6):
            variant_tag = "medium"
        elif parameters['embed_dim'] == 180 and parameters['depths'] == (8, 8, 8, 8, 8, 8, 8, 8):
            variant_tag = "large"
        
        # Create ImageModelDescriptor with relevant metadata
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=AetherNetFusedArch, # Reference to this architecture class
            purpose="Super-Resolution",
            tags=AetherNetFusedArch.tags + (
                f"x{parameters['upscale']}",
                variant_tag, # Add the variant tag
                "fused"
            ) if variant_tag else AetherNetFusedArch.tags + (f"x{parameters['upscale']}", "fused"),
            scale=parameters['upscale'],
            input_channels=parameters['in_chans'],
            output_channels=parameters['out_chans'],
            supports_half=True,       # Assuming AetherNet supports half precision
            supports_bfloat16=True,   # Assuming AetherNet supports bfloat16 precision
        )


# --- Unfused AetherNet Architecture for Spandrel ---

@register_arch
class AetherNetUnfusedArch(AetherNetCommonArch):
    """
    Spandrel Architecture definition for an UN-FUSED AetherNet model.
    This class is responsible for detecting and loading unfused AetherNet models.
    """
    id = "AetherNet-Unfused"
    name = "AetherNet (Unfused)"

    @staticmethod
    def detect(state_dict: StateDict) -> bool:
        """
        Detects if the given state_dict belongs to an unfused AetherNet model.
        It checks for the presence of unfused convolution keys and absence of fused keys,
        along with a characteristic key like 'conv_first.weight' to confirm it's AetherNet.
        """
        has_lk_conv = False
        has_sk_conv = False
        has_fused_conv = False

        for key in state_dict.keys():
            if '.conv.lk_conv.weight' in key:
                has_lk_conv = True
            if '.conv.sk_conv.weight' in key:
                has_sk_conv = True
            if '.conv.fused_conv.weight' in key:
                has_fused_conv = True
            
            if (has_lk_conv or has_sk_conv) and has_fused_conv:
                return False # Conflict

        # Check for core unfused structure and a distinctive AetherNet key
        return (has_lk_conv or has_sk_conv) and 'conv_first.weight' in state_dict and not has_fused_conv

    @staticmethod
    def load(state_dict: StateDict) -> ImageModelDescriptor[aether]:
        """
        Loads an unfused AetherNet model from the state_dict.
        Initializes the model in its unfused state.
        """
        model, parameters = AetherNetCommonArch._create_model_and_load_state_dict(state_dict, fused_init=False)
        model.eval() # Ensure model is in evaluation mode

        # Determine model variant for tags
        variant_tag = ""
        if parameters['embed_dim'] == 96 and parameters['depths'] == (4, 4, 4, 4):
            variant_tag = "small"
        elif parameters['embed_dim'] == 128 and parameters['depths'] == (6, 6, 6, 6, 6, 6):
            variant_tag = "medium"
        elif parameters['embed_dim'] == 180 and parameters['depths'] == (8, 8, 8, 8, 8, 8, 8, 8):
            variant_tag = "large"

        # Create ImageModelDescriptor with relevant metadata
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=AetherNetUnfusedArch,
            purpose="Super-Resolution",
            tags=AetherNetUnfusedArch.tags + (
                f"x{parameters['upscale']}",
                variant_tag, # Add the variant tag
                "unfused"
            ) if variant_tag else AetherNetUnfusedArch.tags + (f"x{parameters['upscale']}", "unfused"),
            scale=parameters['upscale'],
            input_channels=parameters['in_chans'],
            output_channels=parameters['in_chans'], # Output channels are usually same as input for SR
            supports_half=True,       # Assuming AetherNet supports half precision
            supports_bfloat16=True,   # Assuming AetherNet supports bfloat16 precision
        )

