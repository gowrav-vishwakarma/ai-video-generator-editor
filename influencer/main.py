import os
# Set PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import importlib.util
import torch
from typing import Optional, Any, Callable, Dict, Tuple, List

# Import configuration
from influencer.config import ContentConfig

# Import utility functions
from influencer.utils.memory import clear_vram

# Import default model handlers
from influencer.models.text.llm import load_llm, generate_script_and_prompts
from influencer.models.audio.tts import load_tts, generate_scene_audio
from influencer.models.image.t2i import load_t2i_pipeline, generate_scene_images
from influencer.models.video.img2vid import load_i2v_pipeline, generate_scene_videos_from_images
from influencer.models.video.text2vid import load_t2v_pipeline, generate_scene_videos_from_text
from influencer.models.video.framepack import load_framepack_pipeline, generate_scene_videos_with_framepack

# Import default assembler
from influencer.generators.video_assembly import assemble_final_video


def import_module_from_file(file_path: str) -> Any:
    """Import a module from a file path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find module at {file_path}")
    
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def main_automation_flow(topic: str, config: Optional[ContentConfig] = None):
    """Main workflow to generate Instagram content from a topic"""
    if config is None:
        config = ContentConfig()

    # Initialize variables to None to ensure proper cleanup in finally block
    llm_model, llm_tokenizer = None, None
    tts_model = None
    t2i_pipe, refiner_pipe = None, None
    i2v_pipe = None
    t2v_pipe = None
    framepack_pipe = None
    final_video_path = None

    # Import custom implementations if specified
    custom_modules = {}
    
    if config.text_model_implementation:
        custom_modules['text'] = import_module_from_file(config.text_model_implementation)
        
    if config.tts_model_implementation:
        custom_modules['audio'] = import_module_from_file(config.tts_model_implementation)
        
    if config.image_model_implementation:
        custom_modules['image'] = import_module_from_file(config.image_model_implementation)
        
    if config.img2vid_model_implementation:
        custom_modules['img2vid'] = import_module_from_file(config.img2vid_model_implementation)
        
    if config.text2vid_model_implementation:
        custom_modules['text2vid'] = import_module_from_file(config.text2vid_model_implementation)
        
    if config.framepack_model_implementation:
        custom_modules['framepack'] = import_module_from_file(config.framepack_model_implementation)
        
    if config.video_assembly_implementation:
        custom_modules['assembly'] = import_module_from_file(config.video_assembly_implementation)

    try:
        # 1. LLM for Script and Prompts
        print("\n=== Generating Script and Visual Prompts ===")
        
        if 'text' in custom_modules and hasattr(custom_modules['text'], 'load_llm'):
            llm_model, llm_tokenizer = custom_modules['text'].load_llm(config.text_model, config.device)
            narration_scenes, visual_prompts_scenes, hashtags = custom_modules['text'].generate_script_and_prompts(
                topic, llm_model, llm_tokenizer, config
            )
        else:
            llm_model, llm_tokenizer = load_llm(config.text_model, config.device)
            narration_scenes, visual_prompts_scenes, hashtags = generate_script_and_prompts(
                topic, llm_model, llm_tokenizer, config
            )
            
        clear_vram(llm_model)

        # Set default speaker_wav if not provided and using XTTS (moved before TTS step)
        if (config.speaker_wav is None or not os.path.exists(config.speaker_wav)) and "xtts" in config.tts_model.lower():
            default_wav = "record_out.wav"
            print(f"DEBUG: Looking for default_wav at: {os.path.abspath(default_wav)}")
            print(f"DEBUG: Current working directory: {os.getcwd()}")
            if os.path.exists(default_wav):
                config.speaker_wav = default_wav
            else:
                print(f"WARNING: XTTSv2 speaker reference audio not found at {default_wav}. TTS might use a default voice or fail.")

        # 2. TTS for Narration
        print("\n=== Generating Audio Narration ===")
        
        if 'audio' in custom_modules and hasattr(custom_modules['audio'], 'load_tts'):
            tts_model = custom_modules['audio'].load_tts(config.tts_model, config.device)
            print(f"DEBUG: config.speaker_wav = {config.speaker_wav}, exists: {os.path.exists(config.speaker_wav) if config.speaker_wav else 'N/A'}")
            audio_paths = custom_modules['audio'].generate_scene_audio(narration_scenes, tts_model, config)
        else:
            tts_model = load_tts(config.tts_model, config.device)
            audio_paths = generate_scene_audio(narration_scenes, tts_model, config)
            
        clear_vram(tts_model)

        # 3. Visual Generation
        video_clip_paths = []
        
        if config.video_generation_mode in ["img2vid", "framepack"]:
            # First generate keyframe images for either img2vid or framepack
            print("\n=== Generating Keyframe Images ===")
            
            if 'image' in custom_modules and hasattr(custom_modules['image'], 'load_t2i_pipeline'):
                t2i_pipe, refiner_pipe = custom_modules['image'].load_t2i_pipeline(
                    config.image_model, 
                    config.image_refiner,
                    config.device
                )
                image_paths = custom_modules['image'].generate_scene_images(
                    visual_prompts_scenes, 
                    t2i_pipe, 
                    refiner_pipe, 
                    config
                )
            else:
                t2i_pipe, refiner_pipe = load_t2i_pipeline(
                    config.image_model, 
                    config.image_refiner,
                    config.device
                )
                image_paths = generate_scene_images(
                    visual_prompts_scenes, 
                    t2i_pipe, 
                    refiner_pipe, 
                    config
                )
                
            clear_vram(t2i_pipe, refiner_pipe)
            
            # Now generate videos based on the selected mode
            if config.video_generation_mode == "img2vid":
                # Stable Video Diffusion (SVD) flow
                print("\n=== Generating Videos from Images using SVD ===")
                
                if 'img2vid' in custom_modules and hasattr(custom_modules['img2vid'], 'load_i2v_pipeline'):
                    i2v_pipe = custom_modules['img2vid'].load_i2v_pipeline(config.img2vid_model, config.device)
                    video_clip_paths = custom_modules['img2vid'].generate_scene_videos_from_images(
                        image_paths,
                        i2v_pipe,
                        narration_scenes,
                        config
                    )
                else:
                    i2v_pipe = load_i2v_pipeline(config.img2vid_model, config.device)
                    video_clip_paths = generate_scene_videos_from_images(
                        image_paths,
                        i2v_pipe,
                        narration_scenes,
                        config
                    )
                    
                clear_vram(i2v_pipe)
                
            elif config.video_generation_mode == "framepack":
                # Hunyuan Video Framepack flow
                print("\n=== Generating Videos from Images using Framepack ===")
                
                if 'framepack' in custom_modules and hasattr(custom_modules['framepack'], 'load_framepack_pipeline'):
                    framepack_pipe = custom_modules['framepack'].load_framepack_pipeline(
                        transformer_model_id=config.framepack_transformer_model,
                        hunyuan_model_id=config.framepack_hunyuan_model,
                        image_encoder_id=config.framepack_image_encoder,
                        device=config.device
                    )
                    video_clip_paths = custom_modules['framepack'].generate_scene_videos_with_framepack(
                        visual_prompts_scenes,
                        image_paths,
                        framepack_pipe,
                        narration_scenes,
                        config
                    )
                else:
                    framepack_pipe = load_framepack_pipeline(
                        transformer_model_id=config.framepack_transformer_model,
                        hunyuan_model_id=config.framepack_hunyuan_model,
                        image_encoder_id=config.framepack_image_encoder,
                        device=config.device
                    )
                    video_clip_paths = generate_scene_videos_with_framepack(
                        visual_prompts_scenes,
                        image_paths,
                        framepack_pipe,
                        narration_scenes,
                        config
                    )
                    
                clear_vram(framepack_pipe)
                
        elif config.video_generation_mode == "text2vid":
            # Direct Text-to-Video (ModelScope) flow
            print("\n=== Generating Videos Directly from Text ===")
            
            if 'text2vid' in custom_modules and hasattr(custom_modules['text2vid'], 'load_t2v_pipeline'):
                t2v_pipe = custom_modules['text2vid'].load_t2v_pipeline(config.text2vid_model, config.device)
                video_clip_paths = custom_modules['text2vid'].generate_scene_videos_from_text(
                    visual_prompts_scenes,
                    t2v_pipe,
                    narration_scenes,
                    config
                )
            else:
                t2v_pipe = load_t2v_pipeline(config.text2vid_model, config.device)
                video_clip_paths = generate_scene_videos_from_text(
                    visual_prompts_scenes,
                    t2v_pipe,
                    narration_scenes,
                    config
                )
                
            clear_vram(t2v_pipe)
        
        else:
            raise ValueError(f"Unsupported video generation mode: {config.video_generation_mode}")

        # 4. Video Assembly
        if video_clip_paths and audio_paths:
            print("\n=== Assembling Final Video ===")
            
            if 'assembly' in custom_modules and hasattr(custom_modules['assembly'], 'assemble_final_video'):
                final_video_path = custom_modules['assembly'].assemble_final_video(
                    video_clip_paths, 
                    audio_paths, 
                    narration_scenes,
                    config
                )
            else:
                final_video_path = assemble_final_video(
                    video_clip_paths, 
                    audio_paths, 
                    narration_scenes,
                    config
                )
        else:
            print("Not enough assets generated to assemble video.")

        # 5. Output final info
        if final_video_path:
            print("\n=== AUTOMATION COMPLETE ===")
            print(f"Final Video: {final_video_path}")
            print(f"Suggested Instagram Caption Text:\n{' '.join([scene['text'] for scene in narration_scenes])}")
            print(f"Suggested Hashtags: {', '.join(hashtags)}")
        else:
            print("\n=== AUTOMATION FAILED ===")
            print("Check logs for errors.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all models are cleared from VRAM
        print("Cleaning up any remaining models from VRAM...")
        models_to_clear = [m for m in [
            llm_model, llm_tokenizer, tts_model, 
            t2i_pipe, refiner_pipe, i2v_pipe, t2v_pipe, framepack_pipe
        ] if m is not None]
        
        if models_to_clear:
            clear_vram(*models_to_clear)
        print("Cleanup finished.")


if __name__ == "__main__":
    # Example configuration
    config = ContentConfig(
        # Video settings
        target_video_length=30.0,
        max_scene_length=3.0,
        min_scenes=2,
        max_scenes=3,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Model selection
        text_model="HuggingFaceH4/zephyr-7b-beta",
        tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
        speaker_wav="record_out.wav",  # For XTTS voice cloning
        
        # Image settings
        image_model="stabilityai/stable-diffusion-xl-base-1.0",
        image_refiner="stabilityai/stable-diffusion-xl-refiner-1.0",
        
        # Video generation mode
        video_generation_mode="img2vid",  # or "text2vid" or "framepack"
        img2vid_model="stabilityai/stable-video-diffusion-img2vid-xt",
        text2vid_model="damo-vilab/text-to-video-ms-1.7b",
        
        # Output settings
        output_dir="instagram_content"
    )

    topic_for_reel = "benefits of meditation"
    main_automation_flow(topic_for_reel, config) 