import streamlit as st
import os
import json
from datetime import datetime
import torch
import time

# Fix for torch.classes issue if needed, though often it's better to manage environments.
torch.classes.__path__ = []

from project_manager import ProjectManager
from config_manager import ContentConfig
from ui_task_executor import UITaskExecutor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Video Generation Pipeline",
    page_icon="🎥",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    defaults = {
        'current_project': None,
        'current_step': 'project_selection',
        'auto_mode': True,
        'ui_executor': None,
        'speaker_audio': None,
        'is_processing': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- HELPER FUNCTIONS ---

def list_projects():
    """List all existing projects."""
    projects = []
    base_dir = "modular_reels_output"
    if not os.path.exists(base_dir):
        return []
    for project_dir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project_dir)
        if os.path.isdir(project_path):
            project_file = os.path.join(project_path, "project.json")
            if os.path.exists(project_file):
                try:
                    with open(project_file, 'r') as f:
                        data = json.load(f)
                    projects.append({
                        'name': project_dir,
                        'topic': data['project_info']['topic'],
                        'created_at': datetime.fromtimestamp(data['project_info']['created_at']),
                        'status': data['project_info']['status']
                    })
                except Exception as e:
                    st.error(f"Error loading project {project_dir}: {e}")
    return sorted(projects, key=lambda p: p['created_at'], reverse=True)

def go_to_step(step_name):
    """Navigate to a specific step in the UI."""
    st.session_state.current_step = step_name
    st.rerun()

def load_project(project_name):
    """Load an existing project and navigate to the dashboard."""
    project_manager = ProjectManager(f"modular_reels_output/{project_name}")
    if project_manager.load_project():
        st.session_state.current_project = project_manager
        st.session_state.ui_executor = UITaskExecutor(project_manager)
        st.session_state.auto_mode = False  # Default to manual mode when loading a project
        st.session_state.is_processing = False
        go_to_step('processing_dashboard')
    else:
        st.error("Failed to load project.")

def create_new_project(topic, auto_mode, uploaded_audio):
    """Create a new project and generate the initial script."""
    project_name = "".join(c for c in topic.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
    output_dir = f"modular_reels_output/{project_name}_{int(time.time())}"

    content_cfg = ContentConfig(output_dir=output_dir)
    project_manager = ProjectManager(output_dir)
    project_manager.initialize_project(topic, content_cfg)
    
    st.session_state.current_project = project_manager
    st.session_state.ui_executor = UITaskExecutor(project_manager)
    st.session_state.auto_mode = auto_mode

    if uploaded_audio:
        speaker_audio_path = os.path.join(output_dir, "speaker_audio.wav")
        with open(speaker_audio_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        st.session_state.speaker_audio = speaker_audio_path
    
    with st.spinner("Generating initial script..."):
        success = st.session_state.ui_executor.task_executor.execute_task(
            "generate_script", {"topic": topic}
        )
    
    if success:
        st.success("Script generated successfully!")
        go_to_step('processing_dashboard')
    else:
        st.error("Failed to generate script. Please try again.")
        st.session_state.current_project = None # Clear failed project

# --- UI RENDERING FUNCTIONS ---

def render_project_selection():
    """Render the main screen for selecting or creating a project."""
    st.title("🎥 AI Video Generation Pipeline")
    st.write("Create a new project or load an existing one to continue.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Existing Projects")
        projects = list_projects()
        if projects:
            for p in projects:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 2, 1])
                    c1.write(f"**{p['topic']}**")
                    c2.write(f"_{p['created_at'].strftime('%Y-%m-%d %H:%M')}_")
                    c3.button("Load", key=f"load_{p['name']}", on_click=load_project, args=(p['name'],), use_container_width=True)
        else:
            st.info("No projects found. Create one to get started!")

    with col2:
        with st.form("new_project_form"):
            st.subheader("Create New Project")
            topic = st.text_input("Video Topic", placeholder="e.g., The History of Ancient Rome")
            auto_mode = st.checkbox("Run in Automatic Mode", value=True, help="If checked, the entire video will be generated automatically. Uncheck for manual control over each step.")
            uploaded_audio = st.file_uploader("Reference Speaker Audio (Optional, .wav)", type=['wav'])
            
            submitted = st.form_submit_button("Create & Start", type="primary")
            if submitted:
                if not topic:
                    st.error("A video topic is required.")
                else:
                    create_new_project(topic, auto_mode, uploaded_audio)

def render_processing_dashboard():
    """The main workspace for building the video, supporting both modes."""
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor

    st.title(f"🎬 Project: {project.state.project_info['topic']}")

    # --- HEADER CONTROLS ---
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        if st.button("⬅️ Back to Projects"):
            go_to_step('project_selection')
    with c3:
        st.session_state.auto_mode = st.toggle("Automatic Mode", value=st.session_state.auto_mode, disabled=st.session_state.is_processing)

    st.divider()

    # --- AUTOMATIC PROCESSING HANDLER ---
    if st.session_state.auto_mode and st.session_state.is_processing:
        with st.container(border=True):
            pm = st.session_state.current_project
            
            total_tasks, completed_tasks = 0, 0
            if pm.state:
                audio_tasks = len(pm.state.script['narration_parts'])
                total_tasks += audio_tasks * 2 # audio + scene creation
                completed_tasks += sum(1 for part in pm.state.script['narration_parts'] if part.get("status") == "generated")
                completed_tasks += len(pm.state.scenes)
                chunk_tasks = sum(len(scene.get('chunks', [])) for scene in pm.state.scenes)
                total_tasks += chunk_tasks
                completed_tasks += sum(1 for scene in pm.state.scenes for chunk in scene.get('chunks', []) if chunk.get('status') == 'video_generated')
                scene_assembly_tasks = len(pm.state.scenes)
                total_tasks += scene_assembly_tasks
                completed_tasks += sum(1 for scene in pm.state.scenes if scene.get('status') == 'completed')
                total_tasks += 1 # final assembly
                if pm.state.final_video.get('status') == 'generated':
                    completed_tasks += 1

            progress_value = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            st.write("🤖 **Automatic processing is active...**")
            st.progress(progress_value)
            status_placeholder = st.empty()
            if st.button("Stop Automatic Processing"):
                st.session_state.is_processing = False
                st.rerun()

        task_name, task_data = pm.get_next_pending_task()
        if task_name is None:
            st.success("✅ All tasks completed!")
            st.session_state.is_processing = False
            go_to_step('video_assembly')
            return
        
        status_placeholder.info(f"🔄 **Executing:** `{task_name}` for `{task_data.get('topic') or f'Part {task_data.get("scene_idx", -1) + 1}'}`")
        if task_name == 'generate_audio':
            task_data['speaker_wav'] = st.session_state.speaker_audio
        
        success = ui_executor.task_executor.execute_task(task_name, task_data)
        if success:
            st.rerun()
        else:
            st.error(f"❌ **Failed on task:** `{task_name}`. Stopping automatic processing.")
            st.session_state.is_processing = False
            st.rerun()

    elif st.session_state.auto_mode and not st.session_state.is_processing:
        if st.button("🚀 Start Automatic Processing", type="primary"):
            st.session_state.is_processing = True
            st.rerun()

    # --- MAIN CONTENT DASHBOARD ---
    st.subheader("Content Generation Dashboard")
    
    with st.expander("Reference Speaker Audio"):
        uploaded_file = st.file_uploader("Upload New Speaker Audio (.wav)", type=['wav'], key="speaker_upload", disabled=st.session_state.is_processing)
        if uploaded_file:
            speaker_audio_path = os.path.join(project.output_dir, "speaker_audio.wav")
            with open(speaker_audio_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.session_state.speaker_audio = speaker_audio_path
            st.success("Speaker audio updated!"); st.rerun()

        if st.session_state.speaker_audio and os.path.exists(st.session_state.speaker_audio):
            st.write("Current reference audio:"); st.audio(st.session_state.speaker_audio)
        else:
            st.info("No reference audio provided. A default TTS voice will be used.")
    
    # #############################################################################
    # # --- CHANGE START: Logic for showing the final assembly button
    # #############################################################################
    
    # Check the next task
    next_task_name, _ = project.get_next_pending_task()
    
    # Show the final assembly button if the next task is final assembly OR if there are no tasks left
    show_final_assembly_button = (next_task_name == "assemble_final") or (next_task_name is None)

    # Put button in a column to control its position
    _, button_col = st.columns([3, 1])
    with button_col:
        if show_final_assembly_button and project.state.scenes:
            if st.button("Assemble / View Final Video ➡️", type="primary", use_container_width=True):
                go_to_step('video_assembly')

    st.write("---")
    # #############################################################################
    # # --- CHANGE END
    # #############################################################################


    for i, part in enumerate(project.state.script["narration_parts"]):
        with st.container(border=True):
            st.header(f"Scene {i+1}")
            
            st.subheader("Narration")
            new_text = st.text_area(f"Script Text", value=part["text"], key=f"text_{i}", height=100, label_visibility="collapsed", disabled=st.session_state.is_processing)
            if new_text != part["text"]:
                ui_executor.update_narration_text(i, new_text)
                st.rerun()

            if part.get("audio_path") and os.path.exists(part["audio_path"]):
                st.audio(part["audio_path"])
                if st.button(f"Regenerate Audio", key=f"regen_audio_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("Regenerating audio..."):
                        ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio)
                    st.rerun()
            else:
                if st.button(f"Generate Audio", key=f"gen_audio_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("Generating audio..."):
                        ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio)
                    st.rerun()
            
            st.divider()

            st.subheader("Visual Chunks")
            scene = project.get_scene_info(i)
            if scene:
                for chunk in scene["chunks"]:
                    chunk_idx = chunk['chunk_idx']
                    with st.container(border=True):
                        prompt_col, image_col, video_col = st.columns([2, 1, 1])
                        
                        with prompt_col:
                            st.write(f"**Chunk {chunk_idx + 1} Prompts**")
                            new_visual = st.text_area("Visual Prompt", value=chunk['visual_prompt'], key=f"v_prompt_{i}_{chunk_idx}", height=125, disabled=st.session_state.is_processing)
                            if new_visual != chunk['visual_prompt']:
                                ui_executor.update_chunk_prompts(i, chunk_idx, visual_prompt=new_visual)
                                st.rerun()
                            
                            new_motion = st.text_area("Motion Prompt", value=chunk.get('motion_prompt', ''), key=f"m_prompt_{i}_{chunk_idx}", height=75, disabled=st.session_state.is_processing)
                            if new_motion != chunk.get('motion_prompt', ''):
                                ui_executor.update_chunk_prompts(i, chunk_idx, motion_prompt=new_motion)
                                st.rerun()
                        
                        with image_col:
                            st.write("**Keyframe Image**")
                            if chunk.get("keyframe_image_path") and os.path.exists(chunk["keyframe_image_path"]):
                                st.image(chunk["keyframe_image_path"])
                            else:
                                st.info("Image pending...")
                        
                        with video_col:
                            st.write("**Video Chunk**")
                            if chunk.get("video_path") and os.path.exists(chunk["video_path"]):
                                st.video(chunk["video_path"])
                                if st.button("Regenerate Video", key=f"regen_chunk_{i}_{chunk_idx}", disabled=st.session_state.is_processing):
                                    with st.spinner("Regenerating video chunk..."):
                                        ui_executor.regenerate_chunk(i, chunk_idx)
                                    st.rerun()
                            else:
                                st.info("Video pending...")
                                if st.button("Generate Video", key=f"gen_chunk_{i}_{chunk_idx}", disabled=st.session_state.is_processing):
                                    with st.spinner("Generating video chunk..."):
                                        ui_executor.regenerate_chunk(i, chunk_idx)
                                    st.rerun()
                    st.divider()

            elif part.get("status") == "generated":
                 if st.button("Create Scene & Chunks", key=f"create_scene_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("Generating scene and chunk prompts..."):
                        ui_executor.create_scene(i)
                    st.rerun()
            else:
                st.info("Generate audio before scene creation is possible.")

def render_video_assembly():
    """Render the final step for assembling and viewing the video."""
    st.title("Final Video Assembly")
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor

    if st.button("⬅️ Back to Dashboard"):
        go_to_step('processing_dashboard')
    
    st.divider()

    final_video_path = project.state.final_video.get("path")
    if final_video_path and os.path.exists(final_video_path):
        st.subheader("Your Final Video is Ready!")
        st.video(final_video_path)
        
        with st.expander("Video Details"):
            st.write("**Full Narration:**")
            st.write(project.state.final_video.get("full_narration_text"))
            st.write("**Suggested Hashtags:**")
            st.info(", ".join(project.state.final_video.get("hashtags", [])))
    
    if st.button("Re-Assemble Final Video", type="primary"):
        with st.spinner("Assembling all scenes and final video..."):
            all_scenes_completed = all(s.get('status') == 'completed' for s in project.state.scenes)
            
            # First assemble any individual scenes that are not completed
            if not all_scenes_completed:
                for scene in project.state.scenes:
                    if scene['status'] != 'completed':
                        st.write(f"Assembling scene {scene['scene_idx']+1}...")
                        ui_executor.task_executor.execute_task("assemble_scene", {"scene_idx": scene["scene_idx"]})
            
            st.write("Assembling final video...")
            success = ui_executor.assemble_final_video()
        if success:
            st.success("Final video assembled successfully!")
            st.rerun()
        else:
            st.error("Failed to assemble final video.")


# --- MAIN APPLICATION ROUTER ---
if st.session_state.current_step == 'project_selection':
    render_project_selection()
elif st.session_state.current_project:
    if st.session_state.current_step == 'processing_dashboard':
        render_processing_dashboard()
    elif st.session_state.current_step == 'video_assembly':
        render_video_assembly()
else:
    go_to_step('project_selection')