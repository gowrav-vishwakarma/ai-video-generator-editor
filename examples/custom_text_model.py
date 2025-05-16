"""
Example custom text model implementation

This file demonstrates how to create a custom text generation implementation
that can be used with the Influencer framework.
"""

import torch
import json
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ContentConfig is imported for type hints, but we accept any config-like object
from influencer.config import ContentConfig

def load_llm(model_id: str, device: str = "cuda"):
    """
    Load a custom language model and tokenizer
    
    This function must follow the same signature as the default implementation.
    """
    print(f"Loading custom LLM implementation: {model_id}...")
    
    # Use different tokenizer settings or loading procedures
    # This example uses Mistral (different chat format than Zephyr)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Custom model loading
    # You can use quantization, LoRA adapters, or other customizations here
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        load_in_8bit=True,  # Example using 8-bit quantization
    )
    
    # Move to device if needed
    if device != "cuda" or not hasattr(model, "hf_device_map"):
        model = model.to(device)
        
    return model, tokenizer

def generate_script_and_prompts(
    topic: str, 
    model: Any,
    tokenizer: Any, 
    config: ContentConfig
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Custom script and visual prompts generator
    
    This function must follow the same signature as the default implementation
    and return the same data structure.
    """
    print(f"Generating script with custom implementation for topic: {topic}")

    # Create a custom message format
    # This example uses Mistral's chat format
    messages = [
        {"role": "system", "content": "You are an expert content creator for Instagram Reels. Your task is to create engaging scripts and visual descriptions for short videos."},
        {"role": "user", "content": f"""
        Please create a script for an Instagram Reel about "{topic}".
        
        The video should be {config.target_video_length} seconds long, with {config.min_scenes}-{config.max_scenes} scenes.
        Each scene should be approximately {config.max_scene_length} seconds.
        
        For each scene, provide:
        1. The narration text
        2. A detailed visual description
        3. The scene duration in seconds
        
        Also suggest 5-7 hashtags related to this topic.
        
        Format your response as valid JSON with this structure:
        {{
            "narration": [
                {{"scene": 1, "text": "narration text", "duration": seconds}},
                {{"scene": 2, "text": "narration text", "duration": seconds}}
            ],
            "visuals": [
                {{"scene": 1, "prompt": "detailed visual description"}},
                {{"scene": 2, "prompt": "detailed visual description"}}
            ],
            "hashtags": ["tag1", "tag2", "tag3"]
        }}
        """}
    ]
    
    # Convert messages to model input
    # This is where custom chat templates would be applied
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Custom generation parameters
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": 800,  # Longer output for better JSON completion
        "do_sample": True,
        "temperature": 0.6,  # Lower temperature for more reliable JSON
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Generate
    outputs = model.generate(**generation_kwargs)
    decoded_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract JSON from the output
    # Sometimes the model might add extra text before or after the JSON
    try:
        # First try to parse the entire output
        response_data = json.loads(decoded_output)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON using a simple heuristic
        try:
            json_start = decoded_output.find('{')
            json_end = decoded_output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = decoded_output[json_start:json_end]
                response_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in the output")
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            # Use fallback
            return fallback_content(topic, config)
    
    # Process the parsed JSON
    try:
        # Sort scenes by scene number to ensure correct order
        narration_data = sorted(response_data.get("narration", []), key=lambda x: x["scene"])
        visuals_data = sorted(response_data.get("visuals", []), key=lambda x: x["scene"])
        
        narration_scenes = []
        visual_prompts = []
        
        # Process narration
        for scene in narration_data:
            narration_scenes.append({
                "text": scene["text"],
                "duration": float(scene.get("duration", config.max_scene_length))
            })
        
        # Process visuals
        for scene in visuals_data:
            # Add quality boosting terms to each prompt
            prompt = scene["prompt"]
            enhanced_prompt = f"{prompt}, high quality, detailed, cinematic lighting, trending on artstation"
            visual_prompts.append(enhanced_prompt)
        
        # Process hashtags - ensure they have # prefix
        hashtags = response_data.get("hashtags", [])
        hashtags = ["#" + tag.lstrip('#') for tag in hashtags]
        
        # Validate scene counts
        if len(narration_scenes) != len(visual_prompts):
            raise ValueError("Mismatch between number of narration and visual scenes")
            
        # Ensure we have at least the minimum number of scenes
        if len(narration_scenes) < config.min_scenes:
            raise ValueError(f"Not enough scenes: {len(narration_scenes)} < {config.min_scenes}")
            
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return fallback_content(topic, config)
        
    return narration_scenes, visual_prompts, hashtags

def fallback_content(topic: str, config: ContentConfig):
    """Generate fallback content in case of failure"""
    print("Using fallback content...")
    
    # Create basic scenes based on config
    narration_scenes = []
    visual_prompts = []
    
    # Intro scene
    narration_scenes.append({
        "text": f"Let's explore {topic} together.",
        "duration": config.max_scene_length
    })
    visual_prompts.append(f"Professional cinematic shot of {topic}, high quality, 8K, trending on artstation")
    
    # Main scenes - generate based on min_scenes
    for i in range(1, max(2, config.min_scenes)):
        narration_scenes.append({
            "text": f"Here's an important aspect of {topic} to consider.",
            "duration": config.max_scene_length
        })
        visual_prompts.append(f"Detailed visualization of {topic}, aspect {i}, 8K quality, professional lighting")
    
    # Conclusion scene
    narration_scenes.append({
        "text": f"Thanks for exploring {topic} with me today!",
        "duration": config.max_scene_length
    })
    visual_prompts.append(f"Inspiring closing shot about {topic}, cinematic, high quality, trending")
    
    # Generate hashtags
    topic_tag = topic.lower().replace(" ", "")
    hashtags = [f"#{topic_tag}", "#InstaReels", "#DidYouKnow", "#Trending", "#FollowForMore"]
    
    return narration_scenes, visual_prompts, hashtags 