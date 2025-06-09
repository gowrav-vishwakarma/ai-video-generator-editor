# In module_discovery.py

import os
import importlib
import inspect
from typing import Dict, List, Any, Type
# Correctly import from base_modules
from base_modules import BaseLLM, BaseTTS, BaseT2I, BaseI2V, BaseT2V, ModuleCapabilities

MODULE_TYPES = {
    "llm": {"base_class": BaseLLM, "path": "llm_modules"},
    "tts": {"base_class": BaseTTS, "path": "tts_modules"},
    "t2i": {"base_class": BaseT2I, "path": "t2i_modules"},
    "i2v": {"base_class": BaseI2V, "path": "i2v_modules"},
    "t2v": {"base_class": BaseT2V, "path": "t2v_modules"},
}

def discover_modules() -> Dict[str, List[Dict[str, Any]]]:
    """
    Scans module directories, imports classes, and gets their capabilities.
    """
    discovered_modules = {key: [] for key in MODULE_TYPES}
    
    for module_type, info in MODULE_TYPES.items():
        module_path = info["path"]
        base_class = info["base_class"]
        
        if not os.path.exists(module_path):
            continue
            
        for filename in os.listdir(module_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"{module_path}.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)
                        if inspect.isclass(attribute) and issubclass(attribute, base_class) and attribute is not base_class:
                            caps = attribute.get_capabilities()
                            discovered_modules[module_type].append({
                                "name": attribute.__name__,
                                "path": f"{module_name}.{attribute.__name__}",
                                "caps": caps,
                                "class": attribute
                            })
                except Exception as e:
                    print(f"Warning: Could not load module {module_name}. Error: {e}")
                    
    return discovered_modules