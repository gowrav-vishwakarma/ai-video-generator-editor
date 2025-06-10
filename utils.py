# In utils.py
import datetime
import json
import os
from PIL import Image, ImageOps
from moviepy import VideoFileClip
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
    """Lists all projects from the output directory with extended details including modules."""
    projects = []
    base_dir = "modular_reels_output"
    if not os.path.exists(base_dir): return []
    for project_dir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project_dir)
        if os.path.isdir(project_path):
            project_file = os.path.join(project_path, "project.json")
            if os.path.exists(project_file):
                try:
                    with open(project_file, 'r') as f: 
                        data = json.load(f)
                    
                    config = data.get('project_info', {}).get('config', {})
                    final_video_info = data.get('final_video', {})
                    status = data.get('project_info', {}).get('status', 'unknown')

                    flow = "Image-to-Video" if config.get('use_svd_flow', True) else "Text-to-Video"
                    
                    final_video_path = None
                    duration = 0.0
                    if status == 'completed':
                        stored_path = final_video_info.get('path')
                        if stored_path and os.path.exists(stored_path):
                            final_video_path = stored_path
                            try:
                                with VideoFileClip(final_video_path) as clip:
                                    duration = clip.duration
                            except Exception as e:
                                print(f"Could not read video duration for {final_video_path}: {e}")
                                duration = 0.0
                    
                    # --- NEW: Extract module selections ---
                    modules = config.get('module_selections', {})
                    # --- END OF NEW ---

                    projects.append({
                        'name': project_dir, 
                        'topic': data['project_info']['topic'], 
                        'created_at': datetime.datetime.fromtimestamp(data['project_info']['created_at']), 
                        'status': status,
                        'flow': flow,
                        'final_video_path': final_video_path,
                        'duration': duration,
                        'modules': modules, # Add modules to the returned dictionary
                    })
                except Exception as e:
                    print(f"Error loading project {project_dir}: {e}") 
    return sorted(projects, key=lambda p: p['created_at'], reverse=True)