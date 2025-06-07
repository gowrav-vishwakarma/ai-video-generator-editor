import streamlit as st
import os
import json
from datetime import datetime
import torch

# Fix for torch.classes issue
torch.classes.__path__ = []

from project_manager import ProjectManager
from config_manager import ContentConfig, ModuleSelectorConfig
from task_executor import TaskExecutor
from ui_task_executor import UITaskExecutor

# Configure page
st.set_page_config(
    page_title="Video Generation Pipeline",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'project_selection'
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = True
if 'ui_executor' not in st.session_state:
    st.session_state.ui_executor = None
if 'speaker_audio' not in st.session_state:
    st.session_state.speaker_audio = None

def list_projects():
    """List all existing projects in the output directory"""
    projects = []
    base_dir = "modular_reels_output"
    if os.path.exists(base_dir):
        for project_dir in os.listdir(base_dir):
            project_path = os.path.join(base_dir, project_dir)
            if os.path.isdir(project_path):
                project_file = os.path.join(project_path, "project.json")
                if os.path.exists(project_file):
                    try:
                        with open(project_file, 'r') as f:
                            project_data = json.load(f)
                            projects.append({
                                'name': project_dir,
                                'topic': project_data['project_info']['topic'],
                                'created_at': datetime.fromtimestamp(project_data['project_info']['created_at']),
                                'status': project_data['project_info']['status']
                            })
                    except Exception as e:
                        st.error(f"Error loading project {project_dir}: {e}")
    return projects

def create_new_project():
    """Create a new project"""
    st.session_state.current_step = 'topic_input'
    st.session_state.current_project = None
    st.session_state.ui_executor = None

def load_project(project_name):
    """Load an existing project"""
    project_manager = ProjectManager(f"modular_reels_output/{project_name}")
    if project_manager.load_project():
        st.session_state.current_project = project_manager
        st.session_state.ui_executor = UITaskExecutor(project_manager)
        st.session_state.current_step = 'script_generation'
    else:
        st.error("Failed to load project")

def render_project_selection():
    """Render the project selection screen"""
    st.title("Video Generation Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Existing Projects")
        projects = list_projects()
        if projects:
            for project in projects:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    col1.write(f"**{project['topic']}**")
                    col2.write(f"Created: {project['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    col3.write(f"Status: {project['status']}")
                    if st.button("Load Project", key=f"load_{project['name']}"):
                        load_project(project['name'])
                    st.divider()
        else:
            st.info("No existing projects found")
    
    with col2:
        st.subheader("Create New Project")
        if st.button("Create New Project", type="primary"):
            create_new_project()

def render_topic_input():
    """Render the topic input screen"""
    st.title("New Project")
    
    topic = st.text_input("Enter the topic for your video")
    st.session_state.auto_mode = st.checkbox("Run in automatic mode", value=True)
    
    if st.button("Generate Script", type="primary", disabled=not topic):
        # Create project name from topic
        project_name = topic.lower().replace(" ", "_")[:30]  # Limit length and replace spaces
        output_dir = f"modular_reels_output/{project_name}"
        
        # Initialize project
        content_cfg = ContentConfig(
            target_video_length_hint=15,
            model_max_video_chunk_duration=2.5,
            max_scene_narration_duration_hint=7.0,
            min_scenes=2,
            max_scenes=4,
            use_svd_flow=True,
            fps=10,
            generation_resolution=(576, 1024),
            final_output_resolution=(1080, 1920),
            output_dir=output_dir,
            font_for_subtitles="Arial"
        )
        
        project_manager = ProjectManager(content_cfg.output_dir)
        project_manager.initialize_project(topic, content_cfg)
        st.session_state.current_project = project_manager
        st.session_state.ui_executor = UITaskExecutor(project_manager)
        
        # Generate initial script
        with st.spinner("Generating script..."):
            success = st.session_state.ui_executor.task_executor.execute_task(
                "generate_script",
                {"topic": topic}
            )
            if success:
                st.success("Script generated successfully!")
                # Automatically move to script generation screen
                st.session_state.current_step = 'script_generation'
                st.rerun()  # Force a rerun to update the UI
            else:
                st.error("Failed to generate script. Please try again.")

def render_script_generation(project: ProjectManager):
    """Render the script generation screen"""
    st.subheader("Script Generation")
    
    # Speaker Audio Section
    st.subheader("Speaker Audio")
    
    # File uploader for new audio
    uploaded_file = st.file_uploader("Upload Speaker Audio (WAV)", type=['wav'])
    if uploaded_file:
        # Save the uploaded file
        speaker_audio_path = os.path.join(project.state.project_info["config"]["output_dir"], "speaker_audio.wav")
        with open(speaker_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.speaker_audio = speaker_audio_path
        st.success("Speaker audio uploaded successfully!")
    
    # Dropdown for existing audio files
    project_dir = project.state.project_info["config"]["output_dir"]
    existing_files = [f for f in os.listdir(project_dir) if f.endswith('.wav') and f != 'speaker_audio.wav']
    
    if existing_files:
        selected_file = st.selectbox("Or select existing audio file:", existing_files)
        if selected_file:
            st.session_state.speaker_audio = os.path.join(project_dir, selected_file)
    
    # Display current speaker audio
    if st.session_state.speaker_audio and os.path.exists(st.session_state.speaker_audio):
        st.audio(st.session_state.speaker_audio)
        st.write(f"Current speaker audio: {st.session_state.speaker_audio}")
    else:
        st.warning("No speaker audio selected. Audio will be generated with default voice.")
    
    st.divider()
    
    # Script Parts Section
    st.subheader("Script Parts")
    
    # Display each script part with its audio and regeneration options
    for i, part in enumerate(project.state.script["narration_parts"]):
        with st.expander(f"Part {i+1}: {part['text'][:50]}..."):
            st.write(part["text"])
            
            # Audio section
            if part.get("audio_path") and os.path.exists(part["audio_path"]):
                st.audio(part["audio_path"])
                if st.button(f"Regenerate Audio for Part {i+1}", key=f"regenerate_audio_{i}"):
                    with st.spinner("Regenerating audio..."):
                        success = st.session_state.ui_executor.regenerate_audio(
                            i, part["text"], st.session_state.speaker_audio
                        )
                        if success:
                            st.rerun()
            else:
                if st.button(f"Generate Audio for Part {i+1}", key=f"generate_audio_{i}"):
                    with st.spinner("Generating audio..."):
                        success = st.session_state.ui_executor.regenerate_audio(
                            i, part["text"], st.session_state.speaker_audio
                        )
                        if success:
                            st.rerun()
            
            # Scene section
            scene = next((s for s in project.state.scenes if s["scene_idx"] == i), None)
            if scene:
                st.subheader(f"Scene {i+1}")
                for chunk in scene["chunks"]:
                    st.write(f"Chunk {chunk['chunk_idx']+1}")
                    st.write(f"Visual Prompt: {chunk['visual_prompt']}")
                    if chunk.get("motion_prompt"):
                        st.write(f"Motion Prompt: {chunk['motion_prompt']}")
                    
                    if chunk.get("video_path") and os.path.exists(chunk["video_path"]):
                        # Display video in a smaller size
                        st.video(chunk["video_path"], format="video/mp4", start_time=0)
                        if st.button(f"Regenerate Chunk {chunk['chunk_idx']+1}", key=f"regenerate_chunk_{i}_{chunk['chunk_idx']}"):
                            with st.spinner("Regenerating chunk..."):
                                success = st.session_state.ui_executor.generate_chunk(
                                    i, chunk["chunk_idx"], chunk["visual_prompt"], chunk.get("motion_prompt")
                                )
                                if success:
                                    st.rerun()
                    else:
                        if st.button(f"Generate Chunk {chunk['chunk_idx']+1}", key=f"generate_chunk_{i}_{chunk['chunk_idx']}"):
                            with st.spinner("Generating chunk..."):
                                success = st.session_state.ui_executor.generate_chunk(
                                    i, chunk["chunk_idx"], chunk["visual_prompt"], chunk.get("motion_prompt")
                                )
                                if success:
                                    st.rerun()
                    st.divider()
            else:
                if part.get("audio_path") and os.path.exists(part["audio_path"]):
                    if st.button(f"Create Scene for Part {i+1}", key=f"create_scene_{i}"):
                        with st.spinner("Creating scene..."):
                            success = st.session_state.ui_executor.create_scene(i)
                            if success:
                                st.rerun()
    
    # Add new script part button
    if st.button("Add New Script Part"):
        with st.spinner("Adding new script part..."):
            success = st.session_state.ui_executor.add_new_script_part(project)
            if success:
                st.rerun()
    
    # Check if all chunks are generated
    all_chunks_generated = True
    for scene in project.state.scenes:
        for chunk in scene["chunks"]:
            if not chunk.get("video_path") or not os.path.exists(chunk["video_path"]):
                all_chunks_generated = False
                break
        if not all_chunks_generated:
            break
    
    # Show Next button if all chunks are generated
    if all_chunks_generated and project.state.scenes:
        st.divider()
        if st.button("Next: Video Assembly", type="primary"):
            st.session_state.current_step = 'video_assembly'
            st.rerun()

def render_scene_generation():
    """Render the scene generation screen"""
    st.title("Scene Generation")
    
    if not st.session_state.current_project:
        st.error("No project loaded")
        return
    
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor
    
    # Display scenes and their chunks
    for scene_idx, scene in enumerate(project.state.scenes):
        st.subheader(f"Scene {scene_idx + 1}")
        
        for chunk_idx, chunk in enumerate(scene["chunks"]):
            with st.expander(f"Chunk {chunk_idx + 1}", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("Visual Prompt:")
                    new_visual = st.text_area("", value=chunk["visual_prompt"], key=f"visual_{scene_idx}_{chunk_idx}")
                    if new_visual != chunk["visual_prompt"]:
                        project.update_chunk_content(scene_idx, chunk_idx, visual_prompt=new_visual)
                    
                    st.write("Motion Prompt:")
                    new_motion = st.text_area("", value=chunk.get("motion_prompt", ""), key=f"motion_{scene_idx}_{chunk_idx}")
                    if new_motion != chunk.get("motion_prompt"):
                        project.update_chunk_content(scene_idx, chunk_idx, motion_prompt=new_motion)
                
                with col2:
                    if chunk.get("keyframe_image_path"):
                        st.image(chunk["keyframe_image_path"])
                    if chunk.get("video_path"):
                        # Display video in a smaller size
                        st.video(chunk["video_path"], format="video/mp4", start_time=0)
                    
                    if st.button("Regenerate", key=f"regen_{scene_idx}_{chunk_idx}"):
                        ui_executor.regenerate_chunk(scene_idx, chunk_idx)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Script"):
            st.session_state.current_step = 'script_generation'
    with col2:
        if st.button("Next: Video Assembly", type="primary"):
            st.session_state.current_step = 'video_assembly'

def render_video_assembly():
    """Render the video assembly screen"""
    st.title("Video Assembly")
    
    if not st.session_state.current_project:
        st.error("No project loaded")
        return
    
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor
    
    # First, check if all scenes are assembled
    scenes_need_assembly = False
    for scene in project.state.scenes:
        if scene["status"] != "completed":
            scenes_need_assembly = True
            break
    
    if scenes_need_assembly:
        st.info("Assembling individual scenes first...")
        for scene in project.state.scenes:
            if scene["status"] != "completed":
                with st.spinner(f"Assembling scene {scene['scene_idx'] + 1}..."):
                    success = ui_executor.task_executor.execute_task("assemble_scene", {"scene_idx": scene["scene_idx"]})
                    if not success:
                        st.error(f"Failed to assemble scene {scene['scene_idx'] + 1}")
                        return
        st.success("All scenes assembled successfully!")
        st.rerun()
    
    # Display final video if available
    if project.state.final_video.get("path"):
        st.video(project.state.final_video["path"])
        st.write("Full Narration Text:")
        st.write(project.state.final_video["full_narration_text"])
        st.write("Hashtags:", ", ".join(project.state.final_video["hashtags"]))
    else:
        st.info("Final video not yet generated")
    
    # Assembly controls
    if st.button("Assemble Final Video", type="primary"):
        with st.spinner("Assembling final video..."):
            success = ui_executor.assemble_final_video()
            if success:
                st.success("Final video assembled successfully!")
                st.rerun()
            else:
                st.error("Failed to assemble final video")
    
    # Navigation buttons
    if st.button("Back to Scenes"):
        st.session_state.current_step = 'scene_generation'

# Main app navigation
if st.session_state.current_step == 'project_selection':
    render_project_selection()
elif st.session_state.current_step == 'topic_input':
    render_topic_input()
elif st.session_state.current_step == 'script_generation':
    render_script_generation(st.session_state.current_project)
elif st.session_state.current_step == 'scene_generation':
    render_scene_generation()
elif st.session_state.current_step == 'video_assembly':
    render_video_assembly() 