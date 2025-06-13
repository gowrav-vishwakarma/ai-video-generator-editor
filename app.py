# In app.py
import streamlit as st
import os
import time
import torch

# Local imports
from project_manager import ProjectManager
from ui_components import render_asset_panel, render_inspector_panel, render_timeline, render_viewer, render_project_selection_page
from system import initialize_system, go_to_step, select_item # Import select_item
from ui_task_executor import UITaskExecutor

# --- Page Config & Initialization ---
st.set_page_config(page_title="Modular AI Video Studio", page_icon="🎬", layout="wide")

# Fix for Streamlit/Torch conflict
torch.classes.__path__ = []

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'current_project_manager': None,
        'current_step': 'project_selection',
        'selected_item_uuid': None,
        'selected_item_type': None,
        'ui_executor': None,
        'is_processing': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    initialize_system()

init_session_state()


def load_project(project_name: str):
    """Loads a project into the session state and navigates to the editor."""
    project_path = os.path.join("modular_reels_output", project_name)
    pm = ProjectManager(project_path)
    if pm.load_project():
        st.session_state.current_project_manager = pm
        st.session_state.ui_executor = UITaskExecutor(pm)
        st.session_state.is_processing = False
        # --- THIS IS THE FIX ---
        # Set the project itself as the initially selected item.
        select_item('project', pm.state.uuid)
        # --- END OF FIX ---
        go_to_step('editor')
    else:
        st.error("Failed to load project.")


def create_new_project(title: str, video_format: str):
    """Creates a new project and navigates to the editor."""
    name = "".join(c for c in title.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
    output_dir = f"modular_reels_output/{name}_{int(time.time())}"
    
    pm = ProjectManager(output_dir)
    pm.initialize_project(title, video_format)
    
    st.session_state.current_project_manager = pm
    st.session_state.ui_executor = UITaskExecutor(pm)
    st.session_state.is_processing = False
    # --- THIS IS THE FIX ---
    # Set the project itself as the initially selected item.
    select_item('project', pm.state.uuid)
    # --- END OF FIX ---
    go_to_step('editor')


def render_editor():
    """Renders the main video editor interface."""
    pm = st.session_state.current_project_manager
    if not pm or not pm.state:
        st.warning("No project loaded. Returning to project selection.")
        go_to_step('project_selection')
        return

    # Allow the project title to be selected, showing the "Add Scene" button
    if st.button(f"🎬 Studio: {pm.state.title}", use_container_width=True):
        select_item('project', pm.state.uuid)
    
    # --- Main Toolbar ---
    toolbar_cols = st.columns([2, 2, 5])
    with toolbar_cols[0]:
        if st.button("⬅️ Back to Projects"):
            st.session_state.current_project_manager = None
            st.session_state.selected_item_uuid = None
            st.session_state.selected_item_type = None
            go_to_step('project_selection')
    
    with toolbar_cols[1]:
        if st.button("🏆 Assemble Final Video", type="primary", use_container_width=True):
            st.session_state.ui_executor.assemble_final_video()


    st.divider()

    # --- Main Editor Layout ---
    left_col, center_col, right_col = st.columns([1, 2, 1.5])

    with left_col:
        render_asset_panel()

    with center_col:
        render_viewer()
        st.divider()
        render_timeline()

    with right_col:
        st.subheader("⚙️ Inspector")
        with st.container(border=True, height=800):
            render_inspector_panel()

# --- Main Application Router ---
if st.session_state.current_step == 'project_selection':
    render_project_selection_page(create_new_project, load_project)
elif st.session_state.current_step == 'editor':
    render_editor()
else:
    go_to_step('project_selection')