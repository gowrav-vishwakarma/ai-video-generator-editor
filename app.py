# In app.py
import streamlit as st
import os
import time
import torch

from project_manager import ProjectManager
from ui_components import render_asset_panel, render_inspector_panel, render_timeline, render_viewer, render_project_selection_page
from system import initialize_system, go_to_step, select_item
from ui_task_executor import UITaskExecutor

st.set_page_config(page_title="Modular AI Video Studio", page_icon="🎬", layout="wide")
torch.classes.__path__ = []

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'current_project_path': None,
        'current_step': 'project_selection',
        'selected_item_uuid': None,
        'selected_item_type': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    initialize_system()

init_session_state()

def load_project(project_name: str):
    """Sets the current project path and navigates to the editor."""
    st.session_state.current_project_path = os.path.join("modular_reels_output", project_name)
    st.session_state.current_step = 'editor'
    st.session_state.selected_item_uuid = None # Reset selection on load
    st.rerun()

def create_new_project(title: str, video_format: str):
    """Creates a new project, sets its path, and navigates to the editor."""
    name = "".join(c for c in title.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
    output_dir = f"modular_reels_output/{name}_{int(time.time())}"
    pm = ProjectManager(output_dir)
    pm.initialize_project(title, video_format)
    st.session_state.current_project_path = output_dir
    st.session_state.current_step = 'editor'
    st.session_state.selected_item_uuid = None # Reset selection on create
    st.rerun()

def render_editor():
    """Renders the main video editor interface."""
    if not st.session_state.current_project_path:
        st.warning("No project path set. Returning to project selection.")
        go_to_step('project_selection')
        return

    pm = ProjectManager(st.session_state.current_project_path)
    if not pm.load_project():
        st.error(f"Failed to load project from: {st.session_state.current_project_path}")
        go_to_step('project_selection')
        return
        
    ui_executor = UITaskExecutor(pm)
    
    # --- START: DEFINITIVE FIX ---
    # If nothing is selected, default to selecting the project itself.
    # We no longer pass the 'rerun' argument.
    if not st.session_state.get('selected_item_uuid'):
        select_item('project', pm.state.uuid)
    # --- END: DEFINITIVE FIX ---

    if st.button(f"🎬 Studio: {pm.state.title}", use_container_width=True):
        select_item('project', pm.state.uuid)
        st.rerun() # Explicitly rerun on this click
    
    toolbar_cols = st.columns([2, 2, 5])
    with toolbar_cols[0]:
        if st.button("⬅️ Back to Projects"):
            st.session_state.current_project_path = None
            st.session_state.selected_item_uuid = None
            st.session_state.selected_item_type = None
            go_to_step('project_selection')
    with toolbar_cols[1]:
        if st.button("🏆 Assemble Final Video", type="primary", use_container_width=True):
            ui_executor.assemble_final_video()

    st.divider()
    left_col, center_col, right_col = st.columns([1, 2, 1.5])
    with left_col:
        render_asset_panel(pm, ui_executor)
    with center_col:
        render_viewer(pm)
        st.divider()
        render_timeline(pm, ui_executor)
    with right_col:
        st.subheader("⚙️ Inspector")
        with st.container(border=True, height=800):
            render_inspector_panel(pm, ui_executor)

if st.session_state.current_step == 'project_selection':
    render_project_selection_page(create_new_project, load_project)
elif st.session_state.current_step == 'editor':
    render_editor()
else:
    go_to_step('project_selection')