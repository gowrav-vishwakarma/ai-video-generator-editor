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
    print("Assembling final video...")
    final_clips = []
    source_clips_to_close = []

    # Use a specific, existing font path or find a system font
    font_path_for_textclip = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path_for_textclip):
        print(f"Warning: Font file not found at: {font_path_for_textclip}")
        print("Using system default font...")
        font_path_for_textclip = None

    # Process each scene
    for i, (video_path, audio_path, narration) in enumerate(zip(video_clip_paths, audio_clip_paths, narration_parts)):
        # Load video and audio clips
        video_clip_temp = VideoFileClip(video_path)
        audio_clip_temp = AudioFileClip(audio_path)
        source_clips_to_close.extend([video_clip_temp, audio_clip_temp])

        # Get durations
        video_duration = video_clip_temp.duration
        audio_duration = audio_clip_temp.duration
        target_duration = narration["duration"]

        # Resize video to target resolution - MoviePy 1.x uses resized instead of resize
        video_clip_resized = video_clip_temp.resized(height=config.target_resolution[1])
        if video_clip_resized.w > config.target_resolution[0]:
            # MoviePy 1.x uses cropped instead of crop
            video_clip_final_shape = video_clip_resized.cropped(x_center=video_clip_resized.w/2, width=config.target_resolution[0])
        else:
            video_clip_final_shape = video_clip_resized

        # MoviePy 1.x uses with_position instead of set_position
        video_clip_positioned = video_clip_final_shape.with_position('center')

        # Handle duration mismatches
        if video_duration > target_duration:
            # If video is longer, trim it using subclipped
            video_clip_timed = video_clip_positioned.subclipped(0, target_duration)
        else:
            # If video is shorter, loop it
            n_loops = int(np.ceil(target_duration / video_duration))
            video_clip_timed = concatenate_videoclips([video_clip_positioned] * n_loops)
            video_clip_timed = video_clip_timed.subclipped(0, target_duration)

        # Handle audio duration
        if audio_duration > target_duration:
            # If audio is longer, trim it using subclipped
            audio_clip_timed = audio_clip_temp.subclipped(0, target_duration)
        else:
            # If audio is shorter, pad with silence
            silence_duration = target_duration - audio_duration
            # MoviePy 1.x audio clip creation
            silence = AudioClip(frame_function=lambda t: 0, duration=silence_duration)
            audio_clip_timed = concatenate_audioclips([audio_clip_temp, silence])

        # MoviePy 1.x uses with_audio instead of set_audio
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

            # MoviePy 1.x uses with_position and with_duration instead of set_position and set_duration
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
            fps=config.fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",
            logger='bar'
        )
    except Exception as e:
        print(f"Error during video writing: {e}")
        print("Make sure ffmpeg is correctly installed and accessible by MoviePy.")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up all clips
        for clip_to_close in source_clips_to_close:
            if hasattr(clip_to_close, 'close') and callable(getattr(clip_to_close, 'close')):
                try:
                    clip_to_close.close()
                except Exception as e_close:
                    print(f"Error closing clip {type(clip_to_close)}: {e_close}")
        if hasattr(final_video, 'close') and callable(getattr(final_video, 'close')):
            try:
                final_video.close()
            except Exception as e_close:
                print(f"Error closing final_video: {e_close}")

    print(f"Final video saved to {final_video_path}")
    return final_video_path 