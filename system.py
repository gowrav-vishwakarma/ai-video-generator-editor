# In system.py
import streamlit as st
import psutil
import GPUtil
from module_discovery import discover_modules
from pydantic import BaseModel

class SystemConfig(BaseModel):
    vram_gb: float
    ram_gb: float

def go_to_step(step_name: str):
    """Utility function to navigate between app steps."""
    st.session_state.current_step = step_name
    st.rerun()

def select_item(item_type: str, item_uuid):
    """Sets the currently selected item and forces a UI refresh."""
    st.session_state.selected_item_type = item_type
    st.session_state.selected_item_uuid = item_uuid
    st.rerun()

@st.cache_resource
def get_discovered_modules():
    """Caches the discovered modules for the session."""
    print("--- Discovering all available modules... ---")
    return discover_modules()

@st.cache_resource
def get_system_config() -> SystemConfig:
    """Detects and caches system specs."""
    print("--- Detecting system specifications... ---")
    try:
        gpus = GPUtil.getGPUs()
        vram = gpus[0].memoryTotal / 1024 if gpus else 0.0
    except Exception:
        vram = 0.0
    ram = psutil.virtual_memory().total / (1024**3)
    return SystemConfig(vram_gb=round(vram, 1), ram_gb=round(ram, 1))

def initialize_system():
    """Ensure all system-level resources are loaded."""
    if 'system_config' not in st.session_state:
        st.session_state.system_config = get_system_config()
    if 'discovered_modules' not in st.session_state:
        st.session_state.discovered_modules = get_discovered_modules()