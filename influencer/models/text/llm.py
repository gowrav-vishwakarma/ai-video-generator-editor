import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Dict, List, Tuple, Any, Optional
from influencer.config import ContentConfig

def load_llm(model_id: str, device: str = "cuda"):
    """Load a language model and tokenizer"""
    print(f"Loading LLM: {model_id}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure pad_token is set if model doesn't have one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Move to device if not using device_map="auto"
    if device != "cuda" or not hasattr(model, "hf_device_map"):
        model = model.to(device)
        
    return model, tokenizer

def generate_script_and_prompts(
    topic: str,
    model: Any,
    tokenizer: Any,
    config: ContentConfig
) -> Tuple[List[Dict], List[str], List[str]]:
    """Generate script and prompts for Instagram content"""
    print(f"Generating script and prompts for topic: {topic}")
    
    # Define the messages for the chat template
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant creating content for an Instagram Reel. Your response must be in valid JSON format with the following structure: {\"narration\": [{\"scene\": 1, \"text\": \"text\", \"duration\": seconds}], \"visuals\": [{\"scene\": 1, \"prompt\": \"prompt\"}], \"hashtags\": [\"tag1\", \"tag2\"]}"
        },
        {
            "role": "user",
            "content": f"""
            Create content for an Instagram Reel about "{topic}".
            The Reel should be engaging and around {config.target_video_length} seconds long.
            Each scene MUST be exactly {config.max_scene_length} seconds.
            Generate between {config.min_scenes} and {config.max_scenes} scenes.
            
            IMPORTANT: Each scene's duration MUST be exactly {config.max_scene_length} seconds, no more and no less.
            
            Return your response in this exact JSON format:
            {{
                "narration": [
                    {{"scene": 1, "text": "First scene narration", "duration": {config.max_scene_length}}},
                    {{"scene": 2, "text": "Second scene narration", "duration": {config.max_scene_length}}}
                ],
                "visuals": [
                    {{"scene": 1, "prompt": "Detailed visual prompt for scene 1"}},
                    {{"scene": 2, "prompt": "Detailed visual prompt for scene 2"}}
                ],
                "hashtags": ["tag1", "tag2", "tag3"]
            }}
            
            Make sure:
            1. Each scene duration is EXACTLY {config.max_scene_length} seconds
            2. The narration text can be spoken naturally within {config.max_scene_length} seconds
            3. Each visual prompt should be detailed and suitable for image/video generation
            """
        }
    ]
    
    # Apply the chat template
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids=tokenized_chat,
        **config.text_model_params
    )
    
    decoded_output = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
    print("LLM Response (model generated part only with chat template):\n", decoded_output)
    
    try:
        # Parse the JSON response
        response_data = json.loads(decoded_output)
        
        # Extract and validate the data
        narration_scenes = []
        visual_prompts = []
        hashtags = []
        
        # Sort scenes by scene number to ensure correct order
        narration_data = sorted(response_data.get("narration", []), key=lambda x: x["scene"])
        visuals_data = sorted(response_data.get("visuals", []), key=lambda x: x["scene"])
        
        for scene in narration_data:
            # Force duration to be exactly max_scene_length
            narration_scenes.append({
                "text": scene["text"],
                "duration": float(config.max_scene_length)
            })
            
        for scene in visuals_data:
            visual_prompts.append(scene["prompt"])
            
        hashtags = response_data.get("hashtags", [])
        
        # Validate we have matching number of scenes
        if len(narration_scenes) != len(visual_prompts):
            raise ValueError("Mismatch between number of narration and visual scenes")
            
        # Validate scene count
        if not (config.min_scenes <= len(narration_scenes) <= config.max_scenes):
            raise ValueError(f"Scene count {len(narration_scenes)} outside allowed range [{config.min_scenes}, {config.max_scenes}]")
            
        # Calculate total duration
        total_duration = sum(scene["duration"] for scene in narration_scenes)
        if abs(total_duration - config.target_video_length) > 5:  # Allow 5 second tolerance
            print(f"Warning: Total duration {total_duration}s differs from target {config.target_video_length}s")
            
        return narration_scenes, visual_prompts, hashtags
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        print("Using fallback content...")
        # Fallback content
        narration_scenes = [
            {"text": f"A short segment about {topic}.", "duration": config.max_scene_length},
            {"text": "Concluding thoughts on the future.", "duration": config.max_scene_length}
        ]
        visual_prompts = [
            f"Cinematic shot of {topic}, vibrant colors, high detail, trending on artstation.",
            f"Abstract representation of {topic}, thought-provoking, professional."
        ]
        hashtags = [f"#{topic.replace(' ', '')}", "#AI", "#GeneratedContent"]
        
        return narration_scenes, visual_prompts, hashtags 