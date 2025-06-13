# In system.py
import json
import os
from pydantic import BaseModel, Field
from typing import Optional, Tuple

# --- START OF MODIFICATION ---
# Import necessary libraries for detection
try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None
# --- END OF MODIFICATION ---


SYSTEM_CONFIG_FILE = "system.json"

class SystemConfig(BaseModel):
    """Stores the user's available system resources."""
    vram_gb: float = Field(description="Available GPU VRAM in GB.")
    ram_gb: float = Field(description="Available system RAM in GB.")

def save_system_config(vram_gb: float, ram_gb: float) -> None:
    """Saves the system resource configuration to system.json."""
    config = SystemConfig(vram_gb=vram_gb, ram_gb=ram_gb)
    with open(SYSTEM_CONFIG_FILE, 'w') as f:
        f.write(config.model_dump_json(indent=4))
    print(f"System configuration saved to {SYSTEM_CONFIG_FILE}")

def load_system_config() -> Optional[SystemConfig]:
    """Loads the system resource configuration from system.json if it exists."""
    if not os.path.exists(SYSTEM_CONFIG_FILE):
        return None
    try:
        with open(SYSTEM_CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return SystemConfig(**data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error loading or parsing {SYSTEM_CONFIG_FILE}: {e}. Please re-enter details.")
        return None

# --- START OF MODIFICATION ---
def detect_system_specs() -> Tuple[float, float]:
    """
    Attempts to detect available system RAM and GPU VRAM.
    Returns (vram_in_gb, ram_in_gb).
    Defaults to 8.0 for VRAM and 16.0 for RAM if detection fails.
    """
    # Default values
    detected_ram_gb = 16.0
    detected_vram_gb = 8.0

    # 1. Detect System RAM
    if psutil:
        try:
            ram_bytes = psutil.virtual_memory().total
            # Round to the nearest whole number for a cleaner UI
            detected_ram_gb = round(ram_bytes / (1024**3))
            print(f"Detected System RAM: {detected_ram_gb} GB")
        except Exception as e:
            print(f"Could not detect system RAM using psutil: {e}. Falling back to default.")
    else:
        print("psutil not installed. Cannot detect RAM. Falling back to default.")

    # 2. Detect GPU VRAM
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Use the VRAM of the first detected GPU
                gpu = gpus[0]
                # VRAM is in MB, convert to GB and round to one decimal place
                detected_vram_gb = round(gpu.memoryTotal / 1024, 1)
                print(f"Detected GPU: {gpu.name} with {detected_vram_gb} GB VRAM")
            else:
                print("GPUtil found no GPUs. Falling back to default VRAM.")
        except Exception as e:
            print(f"Could not detect GPU VRAM using GPUtil: {e}. Falling back to default.")
    else:
        print("GPUtil not installed. Cannot detect VRAM. Falling back to default.")
        
    return detected_vram_gb, detected_ram_gb
# --- END OF MODIFICATION ---