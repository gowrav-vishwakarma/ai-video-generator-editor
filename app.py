# In app.py

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

# Page Config
st.set_page_config(page_title="AI Video Generation Pipeline", page_icon="🎥", layout="wide")

# Session State
def init_session_state():
    defaults = {'current_project': None, 'current_step': 'project_selection', 'auto_mode': True, 'ui_executor': None, 'speaker_audio': None, 'is_processing': False}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state()

# Helper Functions
def list_projects():
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
                    projects.append({'name': project_dir, 'topic': data['project_info']['topic'], 'created_at': datetime.fromtimestamp(data['project_info']['created_at']), 'status': data['project_info']['status']})
                except Exception as e:
                    st.error(f"Error loading project {project_dir}: {e}")
    return sorted(projects, key=lambda p: p['created_at'], reverse=True)

def go_to_step(step_name):
    st.session_state.current_step = step_name; st.rerun()

def load_project(project_name):
    project_manager = ProjectManager(f"modular_reels_output/{project_name}")
    if project_manager.load_project():
        st.session_state.current_project = project_manager
        st.session_state.ui_executor = UITaskExecutor(project_manager)
        st.session_state.auto_mode = False; st.session_state.is_processing = False
        go_to_step('processing_dashboard')
    else:
        st.error("Failed to load project.")

def create_new_project(topic, auto, audio, video_format, length, min_s, max_s, use_svd):
    name = "".join(c for c in topic.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
    output_dir = f"modular_reels_output/{name}_{int(time.time())}"
    
    cfg = ContentConfig(
        output_dir=output_dir, 
        aspect_ratio_format=video_format,
        target_video_length_hint=length, 
        min_scenes=min_s, 
        max_scenes=max_s, 
        use_svd_flow=use_svd
    )
    pm = ProjectManager(output_dir)
    pm.initialize_project(topic, cfg)
    st.session_state.current_project = pm
    st.session_state.ui_executor = UITaskExecutor(pm)
    st.session_state.auto_mode = auto
    if audio:
        speaker_path = os.path.join(output_dir, "speaker_audio.wav")
        with open(speaker_path, "wb") as f: f.write(audio.getbuffer())
        st.session_state.speaker_audio = speaker_path
        
    with st.spinner("Generating script..."):
        # The first task is now part of the UI executor's responsibility
        success = st.session_state.ui_executor.task_executor.execute_task("generate_script", {"topic": topic})

    if success:
        st.success("Script generated!")
        # Reload project to get the script data
        st.session_state.current_project.load_project()
        go_to_step('processing_dashboard')
    else:
        st.error("Failed to generate script.")
        st.session_state.current_project = None

# UI Rendering
def render_project_selection():
    st.title("🎥 AI Video Generation Pipeline")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Existing Projects")
        for p in list_projects():
            with st.container(border=True):
                pc1, pc2, pc3 = st.columns([4, 2, 1])
                pc1.write(f"**{p['topic']}**"); pc2.write(f"_{p['created_at'].strftime('%Y-%m-%d %H:%M')}_")
                pc3.button("Load", key=f"load_{p['name']}", on_click=load_project, args=(p['name'],), use_container_width=True)
    with c2:
        with st.form("new_project_form"):
            st.subheader("Create New Project")
            topic = st.text_area("Video Topic")
            flow = st.radio("Flow", ("Image to Video (High Quality)", "Text to Video (Fast)"), horizontal=True)
            use_svd = "Image to Video" in flow
            col1, col2 = st.columns(2)
            fmt = col1.selectbox("Format", ("Portrait", "Landscape"), index=0)
            length = col2.number_input("Length (s)", min_value=5, value=20, step=5)
            st.write("Scene Count:"); c1, c2 = st.columns(2)
            min_s = c1.number_input("Min", 1, 10, 2, 1)
            max_s = c2.number_input("Max", min_s, 10, 5, 1)
            auto = st.checkbox("Automatic Mode", value=True)
            audio = st.file_uploader("Reference Speaker Audio (Optional, .wav)", type=['wav'])
            if st.form_submit_button("Create & Start", type="primary"):
                if not topic: st.error("Topic required.")
                else: create_new_project(topic, auto, audio, fmt, length, min_s, max_s, use_svd)

def render_processing_dashboard():
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor
    use_svd_flow = project.state.project_info.config.get("use_svd_flow", True)

    st.title(f"🎬 Project: {project.state.project_info.topic}")
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        if st.button("⬅️ Back to Projects"): go_to_step('project_selection')
    with c2:
        if st.session_state.auto_mode:
            btn_text = "⏹️ Stop" if st.session_state.is_processing else "🚀 Start"
            if st.button(f"{btn_text} Automatic Processing", use_container_width=True, type="primary" if not st.session_state.is_processing else "secondary"):
                st.session_state.is_processing = not st.session_state.is_processing; st.rerun()
    with c3:
        st.session_state.auto_mode = st.toggle("Automatic Mode", value=st.session_state.auto_mode, disabled=st.session_state.is_processing)
    st.divider()

    st.subheader("Content Generation Dashboard")
    with st.expander("Reference Speaker Audio"):
        uploaded_file = st.file_uploader("Upload New Speaker Audio (.wav)", key="speaker_upload", disabled=st.session_state.is_processing)
        if uploaded_file:
            speaker_path = os.path.join(project.output_dir, "speaker_audio.wav")
            with open(speaker_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.session_state.speaker_audio = speaker_path; st.success("Speaker audio updated!"); st.rerun()
        if st.session_state.speaker_audio and os.path.exists(st.session_state.speaker_audio):
            st.write("Current audio:"); st.audio(st.session_state.speaker_audio)
        else:
            st.info("No reference audio provided.")
    
    next_task_name, next_task_data = project.get_next_pending_task()
    if (next_task_name == "assemble_final") or (next_task_name is None):
        if st.button("Assemble / View Final Video ➡️", type="primary"): go_to_step('video_assembly')
    st.write("---")

    for i, part in enumerate(project.state.script.narration_parts):
        with st.container(border=True):
            st.header(f"Scene {i+1}")
            st.subheader("Narration")
            new_text = st.text_area("Script", part.text, key=f"text_{i}", height=100, label_visibility="collapsed", disabled=st.session_state.is_processing)
            if new_text != part.text: ui_executor.update_narration_text(i, new_text); st.rerun()

            if part.audio_path and os.path.exists(part.audio_path):
                st.audio(part.audio_path)
                if st.button("Regen Audio", key=f"regen_audio_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("..."): ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio); st.rerun()
            else:
                if st.button("Gen Audio", key=f"gen_audio_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("..."): ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio); st.rerun()
            
            st.divider(); st.subheader("Visual Chunks")
            scene = project.get_scene_info(i)
            if scene:
                for chunk in scene.chunks:
                    chunk_idx = chunk.chunk_idx
                    with st.container(border=True):
                        if use_svd_flow:
                            p_col, i_col, v_col = st.columns([2, 1, 1])
                            with p_col:
                                st.write(f"**Chunk {chunk_idx + 1}**")
                                vis = st.text_area("Visual", chunk.visual_prompt, key=f"v_prompt_{i}_{chunk_idx}", height=125, disabled=st.session_state.is_processing)
                                if vis != chunk.visual_prompt: ui_executor.update_chunk_prompts(i, chunk_idx, visual_prompt=vis); st.rerun()
                                mot = st.text_area("Motion", chunk.motion_prompt, key=f"m_prompt_{i}_{chunk_idx}", height=75, disabled=st.session_state.is_processing)
                                if mot != chunk.motion_prompt: ui_executor.update_chunk_prompts(i, chunk_idx, motion_prompt=mot); st.rerun()
                            with i_col:
                                st.write("**Image**"); has_image = chunk.keyframe_image_path and os.path.exists(chunk.keyframe_image_path)
                                if has_image: st.image(chunk.keyframe_image_path)
                                else: st.info("Image pending...")
                                btn_txt = "Regen Image" if has_image else "Gen Image"
                                if st.button(btn_txt, key=f"gen_img_{i}_{chunk_idx}", disabled=st.session_state.is_processing, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_chunk_image(i, chunk_idx); st.rerun()
                            with v_col:
                                st.write("**Video**"); has_video = chunk.video_path and os.path.exists(chunk.video_path)
                                if has_video: st.video(chunk.video_path)
                                else: st.info("Video pending...")
                                btn_txt = "Regen Video" if has_video else "Gen Video"
                                if st.button(btn_txt, key=f"gen_vid_{i}_{chunk_idx}", disabled=st.session_state.is_processing or not has_image, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_chunk_video(i, chunk_idx); st.rerun()
                        else: # T2V Flow
                            p_col, v_col = st.columns([2, 1])
                            with p_col:
                                st.write(f"**Chunk {chunk_idx + 1} Prompt**")
                                vis = st.text_area("Prompt", chunk.visual_prompt, key=f"v_prompt_{i}_{chunk_idx}", height=125, disabled=st.session_state.is_processing)
                                if vis != chunk.visual_prompt: ui_executor.update_chunk_prompts(i, chunk_idx, visual_prompt=vis); st.rerun()
                            with v_col:
                                st.write("**Video**"); has_video = chunk.video_path and os.path.exists(chunk.video_path)
                                if has_video: st.video(chunk.video_path)
                                else: st.info("Video pending...")
                                btn_txt = "Regen Video" if has_video else "Gen Video"
                                if st.button(btn_txt, key=f"gen_t2v_{i}_{chunk_idx}", disabled=st.session_state.is_processing, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_chunk_t2v(i, chunk_idx); st.rerun()
            elif part.status == "generated":
                 if st.button("Create Scene", key=f"create_scene_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("..."): ui_executor.create_scene(i); st.rerun()
            else: st.info("Generate audio before scene creation.")

    if st.session_state.auto_mode and st.session_state.is_processing:
        if next_task_name is None:
            st.session_state.is_processing = False; st.toast("✅ All tasks done!"); go_to_step('video_assembly')
        else:
            msg = f"Executing: {next_task_name.replace('_', ' ')} for Scene {next_task_data.get('scene_idx', 0) + 1}..."
            if "chunk" in next_task_name: msg = f"Executing: {next_task_name.replace('_', ' ')} for Scene {next_task_data.get('scene_idx', 0) + 1} / Chunk {next_task_data.get('chunk_idx', 0) + 1}..."
            with st.spinner(msg):
                if next_task_name == 'generate_audio': next_task_data['speaker_wav'] = st.session_state.speaker_audio
                success = st.session_state.ui_executor.task_executor.execute_task(next_task_name, next_task_data)
            if success:
                fresh_pm = ProjectManager(st.session_state.current_project.output_dir); fresh_pm.load_project()
                st.session_state.current_project = fresh_pm
                st.session_state.ui_executor = UITaskExecutor(fresh_pm) # Re-init with fresh state
                st.rerun()
            else:
                st.error(f"❌ Failed on: {next_task_name}. Stopping."); st.session_state.is_processing = False; st.rerun()

def render_video_assembly():
    st.title("Final Video Assembly")
    project = st.session_state.current_project
    if st.button("⬅️ Back to Dashboard"): go_to_step('processing_dashboard')
    st.divider()
    final_path = project.state.final_video.path
    if final_path and os.path.exists(final_path):
        st.subheader("Final Video"); st.video(final_path)
        with st.expander("Details"):
            st.write("**Narration:**", project.state.final_video.full_narration_text)
            st.write("**Hashtags:**", ", ".join(project.state.final_video.hashtags))
            
    if st.button("Re-Assemble Final Video", type="primary"):
        with st.spinner("..."):
            # Ensure all scenes are assembled first
            if not all(s.status == 'completed' for s in project.state.scenes):
                for scene in project.state.scenes:
                    if scene.status != 'completed':
                        st.write(f"Assembling scene {scene.scene_idx+1}...")
                        st.session_state.ui_executor.task_executor.execute_task("assemble_scene", {"scene_idx": scene.scene_idx})
            
            # Now assemble final video
            st.write("Assembling final video...")
            success = st.session_state.ui_executor.assemble_final_video()
            
        if success: st.success("Assembled!"); st.rerun()
        else: st.error("Failed.")

# Main App Router
if st.session_state.current_step == 'project_selection': render_project_selection()
elif st.session_state.current_project:
    if st.session_state.current_step == 'processing_dashboard': render_processing_dashboard()
    elif st.session_state.current_step == 'video_assembly': render_video_assembly()
else: go_to_step('project_selection')