import os
import torch
import gc

# Configure PyTorch CUDA memory management
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:512,"  # Maximum size of a single memory block
        "expandable_segments:True"  # Allow memory segments to expand
    )

# Constants for memory management
RESERVED_VRAM_GB = 6.0  # Reserve 6GB VRAM as recommended by FramePack-Batch
HIGH_VRAM_THRESHOLD = 24.0  # More reasonable threshold for high VRAM mode

def get_cuda_free_memory_gb():
    """Get free CUDA memory in GB"""
    if torch.cuda.is_available():
        memory_stats = torch.cuda.memory_stats()
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info()
        bytes_inactive_reserved = bytes_reserved - bytes_active
        bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
        return bytes_total_available / (1024**3)
    return 0

def get_available_vram():
    """Get available VRAM after reservation"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        free_memory = get_cuda_free_memory_gb()
        # Ensure we maintain the reserved memory
        available_memory = max(0, free_memory - RESERVED_VRAM_GB)
        print(f"Total VRAM: {total_memory:.2f}GB, Free: {free_memory:.2f}GB, Available: {available_memory:.2f}GB (Reserved: {RESERVED_VRAM_GB}GB)")
        return available_memory
    return 0

def clear_vram(*models_or_pipelines):
    """Clear VRAM by moving models to CPU and running garbage collection"""
    # First check available memory
    if torch.cuda.is_available():
        before_mem = get_available_vram()
    
    for item in models_or_pipelines:
        if item is None:
            continue
        
        if hasattr(item, 'cpu') and callable(getattr(item, 'cpu')):
            item.cpu()  # If it's a pipeline/model with a .cpu() method
        elif hasattr(item, 'model') and hasattr(item.model, 'cpu') and callable(getattr(item.model, 'cpu')):
            item.model.cpu()
    
    # Delete references to allow garbage collection
    del models_or_pipelines
    
    # Run garbage collection and clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check memory after cleanup
    if torch.cuda.is_available():
        after_mem = get_available_vram()
        print(f"VRAM cleared. Available memory before: {before_mem:.2f}GB, after: {after_mem:.2f}GB")
    else:
        print("VRAM cleared and memory collected.")

def reserve_vram():
    """Reserve VRAM by allocating a tensor"""
    if torch.cuda.is_available():
        # Reserve memory by allocating a tensor
        reserve_tensor = torch.zeros((int(RESERVED_VRAM_GB * 1024**3 / 4),), device='cuda', dtype=torch.float32)
        print(f"Reserved {RESERVED_VRAM_GB}GB of VRAM")
        return reserve_tensor
    return None

def is_high_vram_mode():
    """Check if we should operate in high VRAM mode"""
    available_mem = get_available_vram()
    high_vram = available_mem > HIGH_VRAM_THRESHOLD
    print(f"Operating in {'high' if high_vram else 'low'} VRAM mode (Available: {available_mem:.2f}GB)")
    return high_vram 