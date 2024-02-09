import torch
import numpy as np

def model_parameters_memory_size(model, in_bytes=True):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())
    total_size = total_params + total_buffers
    # Assuming 32-bit (4 bytes) float for each parameter
    total_memory_bytes = total_size * 4
    return total_memory_bytes if in_bytes else total_memory_bytes / (1024)  # Convert to KB
