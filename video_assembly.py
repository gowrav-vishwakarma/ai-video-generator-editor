import math # For math.ceil
import os

from typing import List, Optional, Tuple, Dict, Any
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.audio.AudioClip import concatenate_audioclips, AudioClip
from moviepy.video.VideoClip import ColorClip

from config_manager import ContentConfig 


# --- 5. VIDEO ASSEMBLY ---

def assemble_scene_video_from_sub_clips(
    sub_clip_paths: List[str], 
    target_total_duration: float, 
    config: ContentConfig, 
    scene_idx: int
) -> str:
    """Assembles multiple video sub-clips into a single scene video with precise duration control.
    
    This function takes multiple video sub-clips and combines them into a single scene video
    that matches the target duration. If the combined duration is shorter than the target,
    the video will be looped. If longer, it will be trimmed.
    
    Args:
        sub_clip_paths (List[str]): List of paths to video sub-clips to be combined
        target_total_duration (float): Desired duration for the final scene video in seconds
        config (ContentConfig): Configuration object containing video settings
        scene_idx (int): Index of the scene being assembled
        
    Returns:
        str: Path to the assembled scene video file. Returns empty string if assembly fails.
        
    Note:
        - Handles resource cleanup properly
        - Supports video concatenation and duration adjustment
        - Creates output in the directory specified by config.output_dir
    """
    if not sub_clip_paths:
        print(f"Warning: No sub-clips provided for scene {scene_idx}. Cannot assemble scene video.")
        # Create a short black placeholder?
        placeholder_path = os.path.join(config.output_dir, f"scene_{scene_idx}_placeholder.mp4")
        # Simple way to make a black clip with moviepy if needed, but for now, just return empty string or raise error.
        # For now, let's assume this case is handled upstream or we expect valid paths.
        return "" 

    print(f"Assembling video for scene {scene_idx} from {len(sub_clip_paths)} sub-clips to match duration {target_total_duration:.2f}s.")
    
    clips_to_close = []
    video_sub_clips_mvp = []
    for path in sub_clip_paths:
        clip = VideoFileClip(path)
        video_sub_clips_mvp.append(clip)
        clips_to_close.append(clip)

    # Concatenate raw sub-clips first
    concatenated_raw_video = concatenate_videoclips(video_sub_clips_mvp, method="compose")
    clips_to_close.append(concatenated_raw_video)
    
    # Adjust final concatenated clip to precisely match target_total_duration
    current_duration = concatenated_raw_video.duration
    if abs(current_duration - target_total_duration) < 0.05 : # If very close, accept it
         final_scene_video_timed = concatenated_raw_video 
    elif current_duration > target_total_duration:
        final_scene_video_timed = concatenated_raw_video.subclipped(0, target_total_duration)
    else: # current_duration < target_total_duration - loop the whole concatenated clip
        num_loops = math.ceil(target_total_duration / current_duration)
        looped_clips = [concatenated_raw_video] * num_loops
        temp_looped_video = concatenate_videoclips(looped_clips, method="compose")
        clips_to_close.append(temp_looped_video) # Add to close list
        final_scene_video_timed = temp_looped_video.subclipped(0, target_total_duration)

    # Add the final timed clip to close list if it's a new object (subclip creates new)
    if final_scene_video_timed is not concatenated_raw_video and final_scene_video_timed not in clips_to_close:
        clips_to_close.append(final_scene_video_timed)

    final_scene_video_path = os.path.join(config.output_dir, f"scene_{scene_idx}_assembled_video.mp4")
    try:
        final_scene_video_timed.write_videofile(
            final_scene_video_path, 
            fps=config.fps, 
            codec="libx264", 
            audio=False, # Audio will be added in the final assembly step
            threads=4, preset="medium", logger=None # Quieter logs for sub-assemblies
        )
    except Exception as e:
        print(f"Error writing assembled scene video for scene {scene_idx}: {e}")
        # Fallback or error handling
        final_scene_video_path = "" # Indicate failure
    finally:
        for clip_obj in clips_to_close:
            if hasattr(clip_obj, 'close') and callable(getattr(clip_obj, 'close')):
                clip_obj.close()
    
    print(f"Assembled video for scene {scene_idx} saved to {final_scene_video_path} with duration {final_scene_video_timed.duration:.2f}s.")
    return final_scene_video_path


# In video_assembly.py

def assemble_final_reel(
    processed_scene_assets: List[Tuple[str, str, Dict[str, Any]]],
    config: ContentConfig,
    output_filename: str = "final_reel.mp4"
) -> Optional[str]:
    """Creates the final video reel by combining multiple scene videos with audio and text overlays.
    
    This function takes processed scene assets (video, audio, and narration info) and combines them
    into a final video reel. It handles video resizing, cropping, audio synchronization, and text
    overlay placement. The function ensures proper resource management and cleanup.
    
    Args:
        processed_scene_assets (List[Tuple[str, str, Dict[str, Any]]]): List of tuples containing:
            - scene_video_path: Path to the scene video file
            - scene_audio_path: Path to the scene audio file
            - narration_info: Dictionary containing narration text and duration
        config (ContentConfig): Configuration object containing video settings
        output_filename (str, optional): Name for the final output file. Defaults to "final_reel.mp4"
        
    Returns:
        Optional[str]: Path to the final assembled video file. Returns None if assembly fails.
        
    Features:
        - Combines video, audio, and text captions for each scene
        - Handles video resizing and cropping to target resolution
        - Manages audio synchronization
        - Adds text overlays with proper positioning
        - Implements comprehensive resource cleanup
        
    Note:
        - Requires proper font file for text overlays
        - Handles memory efficiently through proper resource cleanup
        - Provides error handling and fallback mechanisms
    """
    print("Assembling final reel...")
    if not processed_scene_assets:
        print("No processed scene assets to assemble. Final video cannot be created.")
        return None

    print(f"config.add_narration_text_to_video: {config.add_narration_text_to_video}")
    
    final_scene_video_clips = [] # Renamed from final_scene_clips_for_reel for clarity
    
    # This list will store all clips that are loaded or created
    # and should be closed in the finally block.
    all_clips_to_close = []

    font_path_for_textclip = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path_for_textclip):
        print(f"Warning: Font file not found at {font_path_for_textclip}. TextClip will use a default font.")
        font_path_for_textclip = "Liberation-Sans-Bold" # Or "Arial" or None

    for i, (scene_video_path, scene_audio_path, narration_info) in enumerate(processed_scene_assets):
        video_clip_for_scene = None
        audio_clip_for_scene = None
        text_clip_for_scene = None
        background_clip_for_scene = None
        
        try:
            narration_text = narration_info["text"]
            actual_audio_duration = narration_info["duration"] # This is the target duration for this scene

            if not (scene_video_path and os.path.exists(scene_video_path) and \
                    scene_audio_path and os.path.exists(scene_audio_path)):
                print(f"Skipping scene {i} due to missing media files.")
                continue

            # Load video and audio clips
            video_clip_for_scene = VideoFileClip(scene_video_path)
            audio_clip_for_scene = AudioFileClip(scene_audio_path)
            all_clips_to_close.extend([video_clip_for_scene, audio_clip_for_scene])

            video_duration = video_clip_for_scene.duration
            
            # --- Video Duration Matching (using subclipped and concatenate_videoclips for loop) ---
            # First, resize and crop to final shape before timing adjustments IF POSSIBLE,
            # or do timing first. Let's stick to your old code's order:
            # Resize, Crop, Position, THEN Time, then Audio.

            # 1. Resize video to target height
            temp_video_clip = video_clip_for_scene.resized(height=config.final_output_resolution[1])

            # 2. Crop if wider than target width, or pad if narrower
            if temp_video_clip.w > config.final_output_resolution[0]:
                # Using .cropped() as per your working old code
                temp_video_clip = temp_video_clip.cropped(x_center=temp_video_clip.w / 2,
                                                          width=config.final_output_resolution[0])
            elif temp_video_clip.w < config.final_output_resolution[0]:
                # Pad with a background
                background_clip_for_scene = ColorClip(size=config.final_output_resolution,
                                           color=(0,0,0), # Black background
                                           duration=actual_audio_duration) # Duration for background
                all_clips_to_close.append(background_clip_for_scene)
                # Composite video onto background
                temp_video_clip = CompositeVideoClip([background_clip_for_scene, temp_video_clip.with_position('center')],
                                                     size=config.final_output_resolution)
            
            # 3. Position video in center (if not already handled by padding composite)
            # The .with_position('center') might have been applied already if padded.
            # If not padded, apply it now.
            if not (video_clip_for_scene.w < config.final_output_resolution[0] and temp_video_clip.w == config.final_output_resolution[0]):
                 temp_video_clip = temp_video_clip.with_position('center')

            # 4. Handle duration mismatches for the video
            if video_duration > actual_audio_duration: # If original video was longer
                video_clip_timed = temp_video_clip.subclipped(0, actual_audio_duration)
            elif video_duration < actual_audio_duration: # If original video was shorter, loop it
                # Note: we loop the `temp_video_clip` which is already resized/cropped/positioned
                num_loops = math.ceil(actual_audio_duration / video_duration) # Loop based on original duration
                if num_loops == 0 : num_loops = 1 # Ensure at least one instance
                # Create a list of the clip to be looped
                looped_video_parts = [temp_video_clip] * num_loops
                video_clip_concatenated_for_loop = concatenate_videoclips(looped_video_parts)
                all_clips_to_close.append(video_clip_concatenated_for_loop) # This new clip needs closing
                video_clip_timed = video_clip_concatenated_for_loop.subclipped(0, actual_audio_duration)
            else: # Durations match closely enough
                video_clip_timed = temp_video_clip # temp_video_clip is already at its full duration here

            final_audio_for_scene = audio_clip_for_scene # Start with the loaded audio
            if final_audio_for_scene.duration > actual_audio_duration:
                final_audio_for_scene = final_audio_for_scene.subclipped(0, actual_audio_duration)
            elif final_audio_for_scene.duration < actual_audio_duration:
                silence_needed = actual_audio_duration - final_audio_for_scene.duration
                if silence_needed > 0.01: # Only add if significant
                    silence_clip = AudioClip(frame_function=lambda t: 0, duration=silence_needed)
                    all_clips_to_close.append(silence_clip)
                    final_audio_for_scene = concatenate_audioclips([final_audio_for_scene, silence_clip])


            # 5. Combine video and audio
            video_clip_with_audio = video_clip_timed.with_audio(final_audio_for_scene)
            
            # This list will hold the video clip, and conditionally, the text clip.
            clips_for_composition = [video_clip_with_audio]

            # 6. Add text caption (if enabled in config)
            if config.add_narration_text_to_video:
                print(f"Adding narration text for scene {i}...")
                # Calculate font size based on video height (e.g., 5% of height)
                base_font_size = int(config.final_output_resolution[1] * 0.05)  # 5% of height
                font_size = max(40, min(base_font_size, 60))  # Between 40 and 60

                text_width = int(config.final_output_resolution[0] * 0.8)
                aspect_ratio = config.final_output_resolution[0] / config.final_output_resolution[1]
                vertical_position = 0.7 if aspect_ratio < 1 else 0.75

                # --- THIS IS THE FIX: Reverted to the original working syntax ---
                text_clip_for_scene = TextClip(
                    font_path_for_textclip,
                    text=narration_text,
                    font_size=font_size,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    method='caption',
                    size=(text_width, None)
                )
                all_clips_to_close.append(text_clip_for_scene)

                text_clip_final = text_clip_for_scene.with_position(('center', vertical_position), relative=True).with_duration(actual_audio_duration)
                
                clips_for_composition.append(text_clip_final)
            else:
                print(f"Skipping narration text for scene {i} as per config.")


            # 7. Combine video and (optional) text into final scene composite
            scene_composite = CompositeVideoClip(
                clips_for_composition,
                size=config.final_output_resolution # Ensure composite is target size
            )
            final_scene_video_clips.append(scene_composite)

        except Exception as e_scene:
            print(f"Error processing scene {i}: {e_scene}")
            import traceback
            traceback.print_exc()
            # Any clips opened in this iteration (video_clip_for_scene, etc.) are already in all_clips_to_close
            continue

    if not final_scene_video_clips:
        print("No scenes were successfully composed.")
        # Close any clips that might have been opened
        for clip_obj in all_clips_to_close:
            if hasattr(clip_obj, 'close') and callable(getattr(clip_obj, 'close')):
                try: clip_obj.close()
                except: pass # Ignore errors during cleanup after failure
        return None

    final_video_output_clip = None
    final_video_path = os.path.join(config.output_dir, output_filename)
    try:
        final_video_output_clip = concatenate_videoclips(final_scene_video_clips, method="compose")
        all_clips_to_close.append(final_video_output_clip) # Add final concatenated clip for closing

        final_video_output_clip.write_videofile(
            final_video_path,
            fps=config.fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium", # "ultrafast" for speed, "medium" for balance
            logger='bar'
        )
    except Exception as e_write:
        print(f"Error during final video writing: {e_write}")
        import traceback
        traceback.print_exc()
        final_video_path = None # Indicate failure
    finally:
        # Close all clips.
        # `final_scene_video_clips` contains CompositeVideoClips that are sources for `final_video_output_clip`.
        # Closing `final_video_output_clip` should ideally handle its sources if method='compose'.
        # `all_clips_to_close` contains initial VideoFileClips, AudioFileClips, created ColorClips, TextClips,
        # and potentially intermediate concatenated clips.
        
        # Make a set of unique clip objects to close to avoid issues with multiple references
        # to the same underlying resources.
        clips_to_actually_close = {id(c): c for c in all_clips_to_close if c}.values()
        
        for clip_obj in clips_to_actually_close:
            if hasattr(clip_obj, 'close') and callable(getattr(clip_obj, 'close')):
                try:
                    clip_obj.close()
                except Exception as e_close:
                    # print(f"Error closing a clip {type(clip_obj)}: {e_close}") # Can be noisy
                    pass
        
        # Also ensure the list of scene composites themselves are closed, as they are also clips
        for scene_comp in final_scene_video_clips:
            if hasattr(scene_comp, 'close') and callable(getattr(scene_comp, 'close')):
                try: scene_comp.close()
                except: pass


    if final_video_path:
        print(f"Final reel saved to {final_video_path}")
    return final_video_path