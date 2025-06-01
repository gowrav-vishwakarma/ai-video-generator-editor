from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.audio.AudioClip import concatenate_audioclips, AudioClip
import os
import numpy as np
from typing import List, Dict, Any, Optional
from influencer.config import ContentConfig

def assemble_final_video(
    video_clip_paths: List[str],
    audio_clip_paths: List[str],
    narration_parts: List[Dict],
    config: ContentConfig,
    output_filename: str = "final_reel.mp4",
    font_path: Optional[str] = None
) -> Optional[str]:
    """Assemble final video from individual scenes with audio and text"""
    print(f"Assembling final video from {len(video_clip_paths)} video clips, {len(audio_clip_paths)} audio clips, and {len(narration_parts)} narration parts...")
    final_clips = []
    source_clips_to_close = []

    # Use system font or specified font
    if font_path and os.path.exists(font_path):
        font_path_for_textclip = font_path
    else:
        # Use a specific, existing font path
        font_path_for_textclip = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if not os.path.exists(font_path_for_textclip):
            print(f"Warning: Font file not found at: {font_path_for_textclip}")
            print("Using system default font...")
            font_path_for_textclip = None

    # Group video clips by scene (they might be segmented)
    video_segments_by_scene = {}
    for path in video_clip_paths:
        # Extract scene number from filename (scene_0_segment_0_svd.mp4 or scene_0_svd.mp4)
        filename = os.path.basename(path)
        if "segment" in filename:
            # Format: scene_0_segment_0_svd.mp4
            scene_id = int(filename.split("_")[1])
        else:
            # Format: scene_0_svd.mp4
            scene_id = int(filename.split("_")[1])
            
        if scene_id not in video_segments_by_scene:
            video_segments_by_scene[scene_id] = []
        video_segments_by_scene[scene_id].append(path)
    
    # Sort segments within each scene
    for scene_id in video_segments_by_scene:
        if "segment" in os.path.basename(video_segments_by_scene[scene_id][0]):
            # Sort by segment number
            video_segments_by_scene[scene_id].sort(key=lambda x: int(os.path.basename(x).split("_")[3]))
    
    # Process each scene
    for i, (scene_id, scene_video_paths) in enumerate(sorted(video_segments_by_scene.items())):
        # Get audio path for this scene
        if i < len(audio_clip_paths):
            audio_path = audio_clip_paths[i]
        else:
            print(f"Warning: No audio for scene {i}")
            continue
            
        # Get narration for this scene
        if i < len(narration_parts):
            narration = narration_parts[i]
        else:
            print(f"Warning: No narration for scene {i}")
            continue
            
        # Load audio clip
        audio_clip_temp = AudioFileClip(audio_path)
        source_clips_to_close.append(audio_clip_temp)
        audio_duration = audio_clip_temp.duration
        
        # Set target duration based on narration or audio duration
        target_duration = max(narration["duration"], audio_duration)
        
        # Process all video segments for this scene
        scene_video_clips = []
        for video_path in scene_video_paths:
            video_clip_temp = VideoFileClip(video_path)
            source_clips_to_close.append(video_clip_temp)
            
            # Resize video to target resolution - MoviePy uses resized method
            video_clip_resized = video_clip_temp.resized(height=config.target_resolution[1])
            if video_clip_resized.w > config.target_resolution[0]:
                # MoviePy uses cropped method
                video_clip_final_shape = video_clip_resized.cropped(x_center=video_clip_resized.w/2, width=config.target_resolution[0])
            else:
                video_clip_final_shape = video_clip_resized
                
            # MoviePy uses with_position method
            video_clip_positioned = video_clip_final_shape.with_position('center')
            scene_video_clips.append(video_clip_positioned)
            
        # Concatenate all video segments for this scene
        if len(scene_video_clips) > 1:
            print(f"Concatenating {len(scene_video_clips)} video segments for scene {i}")
            combined_video = concatenate_videoclips(scene_video_clips)
        else:
            combined_video = scene_video_clips[0]
            
        # Get the duration of the combined video
        video_duration = combined_video.duration
        
        # Handle duration mismatches
        if video_duration > target_duration:
            # If video is longer, trim it using subclipped
            video_clip_timed = combined_video.subclipped(0, target_duration)
        else:
            # If video is shorter, loop it
            n_loops = int(np.ceil(target_duration / video_duration))
            video_clip_timed = concatenate_videoclips([combined_video] * n_loops)
            video_clip_timed = video_clip_timed.subclipped(0, target_duration)

        # Handle audio duration
        if audio_duration > target_duration:
            # If audio is longer, trim it using subclipped
            audio_clip_timed = audio_clip_temp.subclipped(0, target_duration)
        else:
            # If audio is shorter, pad with silence
            silence_duration = target_duration - audio_duration
            # MoviePy audio clip creation
            silence = AudioClip(lambda t: 0, duration=silence_duration)
            audio_clip_timed = concatenate_audioclips([audio_clip_temp, silence])

        # MoviePy uses with_audio method
        video_clip_with_audio = video_clip_timed.with_audio(audio_clip_timed)

        # Add text caption
        try:
            txt_clip_temp = TextClip(
                font_path_for_textclip,
                text=narration["text"],
                font_size=60,
                color='white',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(int(config.target_resolution[0]*0.8), None)
            )
            source_clips_to_close.append(txt_clip_temp)

            # MoviePy uses with_position and with_duration methods
            txt_clip_final = txt_clip_temp.with_position(('center', 0.8), relative=True).with_duration(target_duration)

            # Combine video and text
            scene_composite = CompositeVideoClip([video_clip_with_audio, txt_clip_final], size=config.target_resolution)
        except Exception as e:
            print(f"Error creating text caption: {e}. Using video without text.")
            scene_composite = video_clip_with_audio.resized(config.target_resolution)
            
        final_clips.append(scene_composite)

    if not final_clips:
        print("No clips to assemble!")
        for clip_to_close in source_clips_to_close:
            if hasattr(clip_to_close, 'close') and callable(getattr(clip_to_close, 'close')):
                clip_to_close.close()
        return None

    # Concatenate all scenes
    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video_path = os.path.join(config.output_dir, output_filename)

    try:
        final_video.write_videofile(
            final_video_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(config.output_dir, "temp-audio.m4a"),
            remove_temp=True,
            fps=config.fps
        )
        print(f"Final video saved to {final_video_path}")
    except Exception as e:
        print(f"Error writing video file: {e}")
        final_video_path = None
    finally:
        # Close all clips to free memory
        for clip_to_close in source_clips_to_close:
            if hasattr(clip_to_close, 'close') and callable(getattr(clip_to_close, 'close')):
                clip_to_close.close()
        if hasattr(final_video, 'close') and callable(getattr(final_video, 'close')):
            final_video.close()
    
    return final_video_path 