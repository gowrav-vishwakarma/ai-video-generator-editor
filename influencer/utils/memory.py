import torch
import gc

def clear_vram(*models_or_pipelines):
    """Clear VRAM by moving models to CPU and running garbage collection"""
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
    print("VRAM cleared and memory collected.") 