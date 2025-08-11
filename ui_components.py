# In ui_components.py
import streamlit as st
from system import get_discovered_modules, select_item
from utils import list_projects
import os

def render_project_selection_page(create_callback, load_callback):
    st.title("🎥 Modular AI Video Studio")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("🚀 Create New Project")
        with st.form("new_project_form"):
            title = st.text_input("Project Title", "My Awesome Video")
            video_format = st.selectbox("Format", ("Landscape", "Portrait"), index=0)
            if st.form_submit_button("Create Project", type="primary"):
                if title: create_callback(title, video_format)
                
    with c2:
        st.subheader("📂 Load Existing Project")
        projects = list_projects()
        if not projects:
            st.info("No projects found.")
        else:
            for p in projects:
                with st.container(border=True):
                    st.markdown(f"**{p['title']}**")
                    if st.button("Load", key=f"load_{p['name']}", use_container_width=True):
                        load_callback(p['name'])

def render_asset_panel(pm, ui_executor):
    st.subheader("🎭 Casting")
    with st.expander("Characters", expanded=True):
        if not pm.state.characters: st.caption("No characters created yet.")
        for char in pm.state.characters:
            cols = st.columns([1, 3, 1])
            if char.versions and char.versions[0].reference_image_path and os.path.exists(char.versions[0].reference_image_path):
                cols[0].image(char.versions[0].reference_image_path, width=50)
            cols[1].write(char.name)
            if cols[2].button("👁️", key=f"select_char_{char.uuid}", help="Inspect Character"): 
                select_item('character', char.uuid)
                st.rerun()
        with st.popover("➕ Add New Character", use_container_width=True):
            with st.form("add_char_form", clear_on_submit=True):
                name = st.text_input("Character Name"); image = st.file_uploader("Upload Base Image", type=['png', 'jpg', 'jpeg'])
                if st.form_submit_button("Create Character", type="primary"):
                    if name and image: ui_executor.create_character(name, image)

    with st.expander("Voices", expanded=True):
        if not pm.state.voices: st.caption("No voices created yet.")
        for voice in pm.state.voices:
            cols = st.columns([4, 1]); cols[0].write(voice.name)
            if cols[1].button("👁️", key=f"select_voice_{voice.uuid}", help="Inspect Voice"):
                select_item('voice', voice.uuid)
                st.rerun()
        with st.popover("➕ Add New Voice", use_container_width=True):
            with st.form("add_voice_form", clear_on_submit=True):
                name = st.text_input("Voice Name"); tts_modules = get_discovered_modules().get('tts', []); tts_options = {mod['caps'].title: mod['path'] for mod in tts_modules}
                selected_title = st.selectbox("TTS Model", options=list(tts_options.keys())); speaker_wav = st.file_uploader("Reference Speaker Audio (.wav)", type=['wav'])
                if st.form_submit_button("Create Voice", type="primary"):
                    if name and selected_title and speaker_wav: ui_executor.create_voice(name, tts_options[selected_title], speaker_wav)

def render_viewer(pm):
    st.subheader("📺 Viewer")
    selected_type = st.session_state.get('selected_item_type'); selected_uuid = st.session_state.get('selected_item_uuid')
    with st.container(border=True, height=450):
        item_to_show, video_path, image_path = None, None, None
        if selected_type == 'shot': item_to_show = pm.get_shot(*selected_uuid)
        elif selected_type == 'scene': item_to_show = pm.get_scene(selected_uuid)
        if item_to_show:
            video_path = getattr(item_to_show, 'video_path', None) or getattr(item_to_show, 'assembled_video_path', None)
            image_path = getattr(item_to_show, 'keyframe_image_path', None) or getattr(item_to_show, 'uploaded_image_path', None)
        if video_path and os.path.exists(video_path): st.video(video_path)
        elif image_path and os.path.exists(image_path): st.image(image_path, use_container_width=True)
        elif pm.state.final_video_path and os.path.exists(pm.state.final_video_path): st.video(pm.state.final_video_path)
        else: st.info("The final video or selected asset preview will appear here.")

def render_timeline(pm, ui_executor):
    st.subheader("🗓️ Timeline")
    st.markdown("""
        <style>
            .stButton>button { padding: 0.2rem 0.4rem; font-size: 0.8rem; margin-top: 2px; margin-bottom: 2px; }
            div[data-testid="stToolbar"] { padding-left: 0.5rem; }
            div.st-emotion-cache-16txtl3 { padding: 0.5rem 1rem; }
        </style>
    """, unsafe_allow_html=True)

    with st.container(border=True, height=350):
        if not pm.state.scenes:
            st.info("No scenes yet. Use the Inspector to add a new scene.")
        
        for i, scene in enumerate(pm.state.scenes):
            with st.container(border=True):
                is_selected_scene = st.session_state.get('selected_item_type') == 'scene' and st.session_state.get('selected_item_uuid') == scene.uuid
                
                header_cols = st.columns([5, 2, 1, 1, 1, 1])
                header_cols[0].markdown(f"**{i+1}. {scene.title}**")
                if header_cols[1].button("Inspect Scene", key=f"inspect_scene_{scene.uuid}", use_container_width=True, type="primary" if is_selected_scene else "secondary"):
                    select_item('scene', scene.uuid); st.rerun()
                if header_cols[2].button("⬆️", key=f"up_scene_{scene.uuid}", disabled=(i==0), use_container_width=True, help="Move scene up"):
                    ui_executor.reorder_scene(scene.uuid, 'up')
                if header_cols[3].button("⬇️", key=f"down_scene_{scene.uuid}", disabled=(i==len(pm.state.scenes)-1), use_container_width=True, help="Move scene down"):
                    ui_executor.reorder_scene(scene.uuid, 'down')
                if header_cols[4].button("➕", key=f"add_scene_after_{scene.uuid}", use_container_width=True, help="Insert scene below"):
                    ui_executor.add_scene(scene.uuid)
                if header_cols[5].button("🗑️", key=f"del_scene_{scene.uuid}", use_container_width=True, help="Delete scene"):
                    ui_executor.delete_scene(scene.uuid)

                st.divider()

                timeline_items = ["add_at_start"] + [item for shot in scene.shots for item in [shot, f"add_after_{shot.uuid}"]]
                shot_cols = st.columns(len(timeline_items))

                for j, item_container in enumerate(shot_cols):
                    shot_item = timeline_items[j]
                    with item_container:
                        if isinstance(shot_item, str) and shot_item.startswith("add"):
                            after_shot_uuid = shot_item.split("_")[-1] if "after" in shot_item else None
                            if st.button("➕", key=f"add_shot_{scene.uuid}_{shot_item}", help="Add new shot here", use_container_width=True):
                                ui_executor.add_shot_to_scene(scene.uuid, after_shot_uuid)
                        else:
                            shot = shot_item
                            shot_idx = scene.shots.index(shot)
                            is_selected = st.session_state.get('selected_item_type') == 'shot' and st.session_state.get('selected_item_uuid', (None, None))[1] == shot.uuid
                            status_icon = "🎬" if shot.status == 'video_generated' else "🖼️" if shot.status in ['image_generated', 'upload_complete'] else "⏳"
                            
                            with st.container(border=True):
                                if st.button(f"{status_icon} Shot {shot_idx + 1}", key=f"select_shot_{shot.uuid}", use_container_width=True, type="primary" if is_selected else "secondary"):
                                    select_item('shot', (scene.uuid, shot.uuid)); st.rerun()
                                if st.button("⬅️", key=f"left_shot_{shot.uuid}", disabled=(shot_idx==0), use_container_width=True):
                                    ui_executor.reorder_shot(scene.uuid, shot.uuid, 'left')
                                if st.button("➡️", key=f"right_shot_{shot.uuid}", disabled=(shot_idx==len(scene.shots)-1), use_container_width=True):
                                    ui_executor.reorder_shot(scene.uuid, shot.uuid, 'right')


def render_inspector_panel(pm, ui_executor):
    selected_type = st.session_state.get('selected_item_type'); selected_uuid = st.session_state.get('selected_item_uuid')
    if not selected_uuid: st.info("Select an item from the timeline to inspect its properties."); return
    
    if selected_type == 'project':
        st.markdown(f"**Project: {pm.state.title}**"); st.write(f"Format: {pm.state.video_format}")
        st.checkbox("Add Narration Text to Final Video", value=pm.state.add_narration_text_to_video, key="project_add_text", on_change=lambda: setattr(pm.state, 'add_narration_text_to_video', st.session_state.project_add_text) or pm._save_state())
        st.divider(); st.subheader("Timeline Management"); 
        if st.button("➕ Add New Scene to End", use_container_width=True, on_click=ui_executor.add_scene, args=(None,)): pass

    elif selected_type == 'scene':
        scene = pm.get_scene(selected_uuid); st.markdown(f"**🎬 Scene Inspector**")
        st.text_input("Scene Title", value=scene.title, key=f"title_{scene.uuid}", on_change=ui_executor.update_scene_title, args=(scene.uuid,))
        st.subheader("Narration"); st.text_area("Script", value=scene.narration.text, key=f"narration_{scene.uuid}", height=120, on_change=ui_executor.update_scene_narration, args=(scene.uuid,))
        voice_options = {v.name: v.uuid for v in pm.state.voices}
        if not voice_options: st.warning("No voices created!")
        else:
            voice_names = list(voice_options.keys()); current_voice = pm.get_voice(scene.narration.voice_uuid);
            idx = voice_names.index(current_voice.name) if current_voice and current_voice.name in voice_names else 0
            st.selectbox("Voice", options=voice_names, index=idx, key=f"voice_select_{scene.uuid}", on_change=ui_executor.update_scene_voice, args=(scene.uuid,))
        can_gen_audio = bool(voice_options and st.session_state.get(f"narration_{scene.uuid}", scene.narration.text) and scene.narration.voice_uuid)
        if scene.narration.audio_path and os.path.exists(scene.narration.audio_path):
            st.audio(scene.narration.audio_path)
            if st.button("Regen Audio", use_container_width=True, disabled=not can_gen_audio, on_click=ui_executor.generate_scene_audio, args=(scene.uuid,)): pass
        else:
            if st.button("Gen Audio", use_container_width=True, type="primary", disabled=not can_gen_audio, on_click=ui_executor.generate_scene_audio, args=(scene.uuid,)): pass

    elif selected_type == 'shot':
        scene_uuid, shot_uuid = selected_uuid; scene = pm.get_scene(scene_uuid); shot = pm.get_shot(scene_uuid, shot_uuid); shot_idx = scene.shots.index(shot)
        st.markdown(f"**🎞️ Shot {shot_idx + 1} Inspector**")
        
        # This radio button is NOT in the form. It re-runs the app on change.
        # Its value is read from session_state to determine which UI to show below.
        flow_options_map = {"T2I ➡️ I2V": "T2I_I2V", "T2V": "T2V", "Upload ➡️ I2V": "Upload_I2V"}
        flow_captions = list(flow_options_map.keys())
        current_flow_caption = next((k for k, v in flow_options_map.items() if v == shot.generation_flow), "T2I ➡️ I2V")
        
        st.radio(
            "Generation Flow", 
            options=flow_captions, 
            index=flow_captions.index(current_flow_caption), 
            horizontal=True,
            key=f"flow_radio_{shot.uuid}" # No on_change needed!
        )
        st.markdown("---")

        # Get the currently displayed flow from the radio button's state
        ui_flow_caption = st.session_state[f"flow_radio_{shot.uuid}"]
        ui_flow_value = flow_options_map[ui_flow_caption]

        with st.form(key=f"shot_form_{shot.uuid}"):
            core_tab, tech_tab, duration_tab = st.tabs(["Core", "Technical", "Duration"])

            with core_tab:
                visual = st.text_area("Visual Prompt", value=shot.visual_prompt, height=100)
                motion = st.text_area("Motion Prompt", value=shot.motion_prompt, height=68, help="For I2V models, describe desired motion.")
                char_options = {c.name: c.uuid for c in pm.state.characters}
                default_chars = [pm.get_character(uid).name for uid in shot.character_uuids if pm.get_character(uid)]
                assigned_chars = st.multiselect("Characters", options=list(char_options.keys()), default=default_chars)

            with tech_tab:
                modules = get_discovered_modules()
                t2i_options = {m['caps'].title: m['path'] for m in modules.get('t2i', [])}
                i2v_options = {m['caps'].title: m['path'] for m in modules.get('i2v', [])}
                t2v_options = {m['caps'].title: m['path'] for m in modules.get('t2v', [])}
                new_t2i_title, new_i2v_title, new_t2v_title, uploaded_file = None, None, None, None
                
                # The UI shown depends on the radio button's current value, not the saved state
                if ui_flow_value == "T2I_I2V":
                    if t2i_options: new_t2i_title = st.selectbox("Image Model", options=t2i_options.keys(), index=list(t2i_options.keys()).index(next((k for k,v in t2i_options.items() if v == shot.module_selections.get('t2i')), list(t2i_options.keys())[0])))
                    if i2v_options: new_i2v_title = st.selectbox("Video Model", options=i2v_options.keys(), index=list(i2v_options.keys()).index(next((k for k,v in i2v_options.items() if v == shot.module_selections.get('i2v')), list(i2v_options.keys())[0])))
                elif ui_flow_value == "T2V":
                    if t2v_options: new_t2v_title = st.selectbox("T2V Model", options=t2v_options.keys(), index=list(t2v_options.keys()).index(next((k for k,v in t2v_options.items() if v == shot.module_selections.get('t2v')), list(t2v_options.keys())[0])))
                elif ui_flow_value == "Upload_I2V":
                    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
                    if i2v_options: new_i2v_title = st.selectbox("Video Model", options=i2v_options.keys(), index=list(i2v_options.keys()).index(next((k for k,v in i2v_options.items() if v == shot.module_selections.get('i2v')), list(i2v_options.keys())[0])))

            with duration_tab:
                is_auto = shot.user_defined_duration is None
                auto_toggle = st.checkbox("Auto-set Duration", value=is_auto)
                user_duration = shot.user_defined_duration
                if not auto_toggle:
                    capacity = ui_executor.task_executor.get_module_max_duration(shot)
                    user_duration = st.slider("Manual Duration (s)", 0.1, capacity or 10.0, user_duration or min(4.0, capacity or 4.0), 0.1)
            
            st.divider()
            
            action_cols = st.columns(4)
            update_btn = action_cols[0].form_submit_button("💾 Update", use_container_width=True)
            gen_img_btn = action_cols[1].form_submit_button("🖼️ Gen Image", use_container_width=True, disabled=(ui_flow_value != 'T2I_I2V'))
            gen_vid_btn = action_cols[2].form_submit_button("📹 Gen Video", use_container_width=True)
            delete_btn = action_cols[3].form_submit_button("🗑️ Delete Shot", use_container_width=True)
            
            if update_btn or gen_img_btn or gen_vid_btn or delete_btn:
                if delete_btn:
                    ui_executor.delete_shot(scene_uuid, shot.uuid)
                    return

                if uploaded_file: ui_executor.handle_shot_image_upload(scene_uuid, shot_uuid, uploaded_file)
                
                final_selections = shot.module_selections.copy()
                if ui_flow_value == "T2I_I2V":
                    if t2i_options and new_t2i_title: final_selections['t2i'] = t2i_options[new_t2i_title]
                    if i2v_options and new_i2v_title: final_selections['i2v'] = i2v_options[new_i2v_title]
                elif ui_flow_value == "T2V":
                    if t2v_options and new_t2v_title: final_selections['t2v'] = t2v_options[new_t2v_title]
                elif ui_flow_value == "Upload_I2V":
                    if i2v_options and new_i2v_title: final_selections['i2v'] = i2v_options[new_i2v_title]
                
                update_data = {
                    "visual_prompt": visual, "motion_prompt": motion, "generation_flow": ui_flow_value,
                    "character_uuids": [char_options[name] for name in assigned_chars],
                    "module_selections": final_selections,
                    "user_defined_duration": None if auto_toggle else user_duration
                }
                ui_executor.pm.update_shot(scene_uuid, shot_uuid, update_data)
                
                if gen_img_btn:
                    st.toast("Updating and generating image...")
                    ui_executor.generate_shot_image(scene_uuid, shot_uuid)
                elif gen_vid_btn:
                    st.toast("Updating and generating video...")
                    if ui_flow_value == "T2V":
                        ui_executor.generate_shot_t2v(scene_uuid, shot_uuid)
                    else:
                        ui_executor.generate_shot_video(scene_uuid, shot_uuid)
                else: 
                    st.toast("Shot details updated!")
                    st.rerun()

    elif selected_type in ['character', 'voice']:
        asset = pm.get_character(selected_uuid) if selected_type == 'character' else pm.get_voice(selected_uuid)
        st.markdown(f"**{selected_type.title()} Inspector: {asset.name}**")
        if selected_type == 'voice' and asset.reference_wav_path and os.path.exists(asset.reference_wav_path):
            st.write(f"**TTS Model:** `{asset.tts_module_path.split('.')[-1]}`");
            st.audio(asset.reference_wav_path)
        if st.button(f"🗑️ Delete {selected_type.title()}", type="secondary", use_container_width=True):
            if selected_type == 'character': ui_executor.delete_character(asset.uuid)
            else: ui_executor.delete_voice(asset.uuid)
    
    with st.expander("🐞 Debug: Show Full Project State"):
        st.json(pm.state.model_dump_json(indent=2))