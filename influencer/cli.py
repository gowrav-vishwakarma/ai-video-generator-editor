#!/usr/bin/env python
import argparse
import json
import os
import sys

import torch
from influencer.config import ContentConfig
from influencer.main import main_automation_flow

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Instagram content from a topic')
    
    # Basic arguments
    parser.add_argument('--topic', type=str, required=True, help='Topic for the Instagram reel')
    parser.add_argument('--output-dir', type=str, default='instagram_content', help='Output directory for generated files')
    
    # Video settings
    parser.add_argument('--video-length', type=float, default=30.0, help='Target video length in seconds')
    parser.add_argument('--scene-length', type=float, default=3.0, help='Maximum scene length in seconds')
    parser.add_argument('--min-scenes', type=int, default=2, help='Minimum number of scenes')
    parser.add_argument('--max-scenes', type=int, default=3, help='Maximum number of scenes')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    parser.add_argument('--resolution', type=str, default='1080x1920', help='Video resolution (width x height)')
    
    # Model selection
    parser.add_argument('--text-model', type=str, default='HuggingFaceH4/zephyr-7b-beta', help='LLM model for script generation')
    parser.add_argument('--text-model-impl', type=str, help='Custom implementation file for text generation')
    
    parser.add_argument('--tts-model', type=str, default='tts_models/multilingual/multi-dataset/xtts_v2', help='TTS model for audio generation')
    parser.add_argument('--tts-model-impl', type=str, help='Custom implementation file for TTS')
    parser.add_argument('--speaker-wav', type=str, help='Speaker reference audio file for voice cloning (required for XTTS)')
    
    parser.add_argument('--image-model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Image generation model')
    parser.add_argument('--image-model-impl', type=str, help='Custom implementation file for image generation')
    parser.add_argument('--image-refiner', type=str, default='stabilityai/stable-diffusion-xl-refiner-1.0', help='Image refiner model (optional)')
    
    parser.add_argument('--video-mode', type=str, choices=['img2vid', 'text2vid', 'framepack'], default='img2vid', 
                        help='Video generation mode: img2vid (Image->Video), text2vid (Text->Video), or framepack (Hunyuan Video Framepack)')
    
    # img2vid options
    parser.add_argument('--img2vid-model', type=str, default='stabilityai/stable-video-diffusion-img2vid-xt', help='Image to video model')
    parser.add_argument('--img2vid-model-impl', type=str, help='Custom implementation file for image-to-video')
    
    # text2vid options
    parser.add_argument('--text2vid-model', type=str, default='damo-vilab/text-to-video-ms-1.7b', help='Text to video model')
    parser.add_argument('--text2vid-model-impl', type=str, help='Custom implementation file for text-to-video')
    
    # framepack options
    parser.add_argument('--framepack-transformer', type=str, default='lllyasviel/FramePackI2V_HY', help='Framepack transformer model')
    parser.add_argument('--framepack-hunyuan', type=str, default='hunyuanvideo-community/HunyuanVideo', help='Hunyuan base model')
    parser.add_argument('--framepack-image-encoder', type=str, default='lllyasviel/flux_redux_bfl', help='Image encoder model')
    parser.add_argument('--framepack-model-impl', type=str, help='Custom implementation file for framepack')
    parser.add_argument('--framepack-sampling', type=str, default='inverted_anti_drifting', 
                        choices=['inverted_anti_drifting', 'vanilla'], help='Framepack sampling type')
    parser.add_argument('--enable-last-frame', action='store_true', help='Enable last frame control for Framepack')
    
    # Assembly
    parser.add_argument('--video-assembly-impl', type=str, help='Custom implementation file for video assembly')
    
    # Device settings
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for model inference')
    
    # Advanced configuration
    parser.add_argument('--config-file', type=str, help='JSON configuration file with all settings')
    parser.add_argument('--save-config', type=str, help='Save current configuration to JSON file')
    
    return parser.parse_args()

def build_config_from_args(args):
    """Build ContentConfig from command line arguments"""
    # Parse resolution
    if 'x' in args.resolution:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    else:
        resolution = (1080, 1920)  # Default resolution
    
    # Create framepack model params
    framepack_model_params = {
        "num_inference_steps": 30,
        "guidance_scale": 9.0,
        "sampling_type": args.framepack_sampling
    }
    
    # Create config
    config = ContentConfig(
        # Video settings
        target_video_length=args.video_length,
        max_scene_length=args.scene_length,
        min_scenes=args.min_scenes,
        max_scenes=args.max_scenes,
        fps=args.fps,
        target_resolution=resolution,
        
        # Device
        device=args.device,
        
        # Model selection
        text_model=args.text_model,
        text_model_implementation=args.text_model_impl,
        
        tts_model=args.tts_model,
        tts_model_implementation=args.tts_model_impl,
        speaker_wav=args.speaker_wav,
        
        # Image settings
        image_model=args.image_model,
        image_model_implementation=args.image_model_impl,
        image_refiner=args.image_refiner,
        
        # Video generation mode
        video_generation_mode=args.video_mode,
        
        # img2vid settings
        img2vid_model=args.img2vid_model,
        img2vid_model_implementation=args.img2vid_model_impl,
        
        # text2vid settings
        text2vid_model=args.text2vid_model,
        text2vid_model_implementation=args.text2vid_model_impl,
        
        # framepack settings
        framepack_transformer_model=args.framepack_transformer,
        framepack_hunyuan_model=args.framepack_hunyuan,
        framepack_image_encoder=args.framepack_image_encoder,
        framepack_model_implementation=args.framepack_model_impl,
        framepack_model_params=framepack_model_params,
        enable_framepack_last_frame=args.enable_last_frame,
        
        # Video assembly
        video_assembly_implementation=args.video_assembly_impl,
        
        # Output settings
        output_dir=args.output_dir
    )
    
    return config

def load_config_from_file(file_path):
    """Load configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Create a ContentConfig object and update its attributes
        config = ContentConfig()
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        return config
    except Exception as e:
        print(f"Error loading configuration from {file_path}: {e}")
        sys.exit(1)

def save_config_to_file(config, file_path):
    """Save configuration to JSON file"""
    # Convert config to dictionary, excluding methods and complex objects
    config_dict = {key: value for key, value in config.__dict__.items() 
                  if not key.startswith('_') and not callable(value)}
    
    # Convert tuple to list for JSON serialization
    if 'target_resolution' in config_dict:
        config_dict['target_resolution'] = list(config_dict['target_resolution'])
    
    try:
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {file_path}")
    except Exception as e:
        print(f"Error saving configuration to {file_path}: {e}")

def main():
    """Main CLI entry point"""
    args = parse_arguments()
    
    # Load config from file if specified
    if args.config_file and os.path.exists(args.config_file):
        config = load_config_from_file(args.config_file)
    else:
        config = build_config_from_args(args)
    
    # Save config if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)
    
    # Run the main automation workflow
    main_automation_flow(args.topic, config)

if __name__ == '__main__':
    main() 