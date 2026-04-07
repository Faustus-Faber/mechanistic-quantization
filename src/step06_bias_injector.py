import os
import torch
import torch.nn as nn

class BiasInjectorWrapper(nn.Module):
    """
    A computationally weight-folded wrapper for Linear layers.
    Because bitsandbytes Linear4bit layers are heavily optimized CUDA kernels that 
    often do not instantiate a native bias parameter for Gemma, attempting to inject 
    data into model.mlp.down_proj.bias directly can cause kernel segmentation faults.
    
    Instead, this wrapper structurally encompasses the 4-bit module and calculates an
    identically performant addition in the physical fp16 space, representing a static 
    Weight-Fold.
    """
    def __init__(self, original_module: nn.Module, patch_tensor: torch.Tensor):
        super().__init__()
        self.original_module = original_module
        # Register as a non-trainable state buffer ensuring it moves to the correct device
        self.register_buffer("patch_bias", patch_tensor.detach().clone())
        
        # Mirror attributes for transparency
        if hasattr(original_module, 'weight'):
            self.weight = original_module.weight
        if hasattr(original_module, 'bias'):
            self.bias = original_module.bias
            
    def forward(self, *args, **kwargs):
        # 1. Defer to the heavily optimized 4-bit CUDA backend
        base_out = self.original_module(*args, **kwargs)
        # 2. Add the calibrated physical vector extracted from the monosemantic geometry
        return base_out + self.patch_bias

def apply_sae_patches(model, layers_to_patch=(29, 30, 31)):
    """
    Applies the extracted corrective SAE patches directly into the Transformer graph.
    
    Args:
        model: The instantiated transformers HuggingFace model containing the architecture.
        layers_to_patch: Tuple of layer indices representing where patches are applied.
    Returns:
        The structurally modified model.
    """
    print(f"\n[Patch Compiler] Applying Causal SAE patches to layers {list(layers_to_patch)}...")
    patches_dir = os.path.join(os.path.dirname(__file__), "..", "data", "patches")
    
    patch_count = 0
    for layer_idx in layers_to_patch:
        patch_file = os.path.join(patches_dir, f"sae_patch_layer_{layer_idx}.pt")
        
        if not os.path.exists(patch_file):
            print(f"[Patch Compiler] WARNING: Patch for layer {layer_idx} not found at {patch_file}. Skipping.")
            continue
            
        # Load the carefully calibrated patch vector
        patch_tensor = torch.load(patch_file, map_location=model.device, weights_only=True)
        # Ensure precision matches the compute space
        patch_tensor = patch_tensor.to(model.dtype)
        
        # Locate the specific terminal connection of the MLP where the vector needs injecting.
        # In Gemma, the residual output from the MLP block is governed by `down_proj`.
        target_submodule = model.model.language_model.layers[layer_idx].mlp.down_proj
        
        # Inject our custom wrapper
        folded_module = BiasInjectorWrapper(target_submodule, patch_tensor)
        model.model.language_model.layers[layer_idx].mlp.down_proj = folded_module
        
        print(f"[Patch Compiler] Successfully folded 4.7KB bias tensor into Layer {layer_idx} MLP projection.")
        patch_count += 1
        
    print(f"[Patch Compiler] Intervention complete. Applied {patch_count} static patches.")
    return model
