import os

# Set PyTorch memory management environment variables
def configure_memory_management():
    """Configure PyTorch CUDA memory management settings"""
    
    # Expandable segments for better memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:512,"  # Maximum size of a single memory block
        "expandable_segments:True,"  # Allow memory segments to expand
        "garbage_collection_threshold:0.8,"  # Trigger GC when 80% of memory is used
        "roundup_power2:True"  # Round up allocations to power of 2 for better efficiency
    )
    
    # Optional: Set CUDA device memory management
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA execution for better error tracking
    
    # Optional: Set NCCL environment variables for multi-GPU setups
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P memory access which can cause issues
    
    print("PyTorch CUDA memory management configured for optimal performance") 