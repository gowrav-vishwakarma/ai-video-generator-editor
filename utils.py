# In utils.py
import datetime
import json
import os
from PIL import Image, ImageOps
import streamlit as st

def load_and_correct_image_orientation(image_source):
    """Loads an image and corrects its orientation based on EXIF data."""
    try:
        image = Image.open(image_source)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        st.error(f"Could not load or correct image: {e}")
        return None

# --- REWRITTEN list_projects function ---
def list_projects():
    """
    Lists all projects from the output directory based on the NEW project structure.
    It looks for 'project_state.json'.
    """
    projects = []
    base_dir = "modular_reels_output"
    if not os.path.exists(base_dir):
        return []

    for project_dir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project_dir)
        if os.path.isdir(project_path):
            # Look for the new project state file
            project_file = os.path.join(project_path, "project_state.json")
            if os.path.exists(project_file):
                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract info directly from the new ProjectState model structure
                    projects.append({
                        'name': project_dir,  # The directory name is the unique ID
                        'title': data.get('title', 'Untitled Project'),
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse project_state.json in {project_dir}")
                except Exception as e:
                    print(f"Error loading project {project_dir}: {e}")
                    
    # Sort by directory name (which includes timestamp) for chronological order
    return sorted(projects, key=lambda p: p['name'], reverse=True)