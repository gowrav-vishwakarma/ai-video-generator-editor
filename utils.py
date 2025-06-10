# In utils.py
import datetime
import json
import os
from PIL import Image, ImageOps
import streamlit as st # Keep st for st.error

def load_and_correct_image_orientation(image_source):
    """
    Loads an image from a source (file path or uploaded file object)
    and corrects its orientation based on EXIF data.
    """
    try:
        image = Image.open(image_source)
        # The magic is in exif_transpose
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        # Using st.error here is okay for a simple app, but for true separation,
        # you might log the error and return None, letting the caller handle the UI.
        # For this project, this is fine.
        st.error(f"Could not load or correct image: {e}")
        return None

def list_projects():
    """Lists all projects from the output directory."""
    projects = []
    base_dir = "modular_reels_output"
    if not os.path.exists(base_dir): return []
    for project_dir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project_dir)
        if os.path.isdir(project_path):
            project_file = os.path.join(project_path, "project.json")
            if os.path.exists(project_file):
                try:
                    with open(project_file, 'r') as f: data = json.load(f)
                    projects.append({
                        'name': project_dir, 
                        'topic': data['project_info']['topic'], 
                        'created_at': datetime.fromtimestamp(data['project_info']['created_at']), 
                        'status': data['project_info']['status']
                    })
                except Exception as e:
                    print(f"Error loading project {project_dir}: {e}") # Use print for NiceGUI server logs
    return sorted(projects, key=lambda p: p['created_at'], reverse=True)