# Influencer - AI Instagram Content Generator

A modular AI system that generates Instagram Reels from topics. Combines AI text, image, audio, and video generation.

## Features

- **Topic-to-Reel Generation**: Generate complete Instagram Reels from a single topic
- **Modular Architecture**: Swap different AI models for each generation stage
- **Flexible Configuration**: JSON config files and command-line options
- **Multiple Video Generation Modes**:
  - Image-to-Video (SVD): Generate keyframe images then animate them (SDXL → SVD)
  - Hunyuan Video Framepack: High-quality video generation with first/last frame control
  - Text-to-Video: Direct text-to-video generation (ModelScope)
- **Custom Implementations**: Easily substitute your own implementations for any component

## Installation

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (recommended)
- FFmpeg (required for video processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/influencer.git
   cd influencer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers accelerate bitsandbytes diffusers TTS moviepy Pillow safetensors sentencepiece
   ```
   Note: Make sure the CUDA version in PyTorch matches your system's CUDA Toolkit.

## Usage

### Basic Usage

Generate a Reel about a specific topic:

```bash
python run_influencer.py --topic "benefits of meditation"
```

### Advanced Options

```bash
python run_influencer.py --topic "healthy breakfast ideas" \
    --output-dir my_reels \
    --video-length 45.0 \
    --scene-length 5.0 \
    --min-scenes 3 \
    --max-scenes 5 \
    --video-mode text2vid \
    --speaker-wav my_voice.wav
```

### Configuration Files

Save a configuration for later:

```bash
python run_influencer.py --topic "workout motivation" --save-config my_config.json
```

Use a saved configuration:

```bash
python run_influencer.py --topic "new topic" --config-file my_config.json
```

## Model Selection

The system allows you to select different models for each generation stage:

### Text Generation (Script)
```bash
python run_influencer.py --topic "topic" --text-model "mistralai/Mistral-7B-Instruct-v0.2"
```

### Audio Generation (TTS)
```bash
python run_influencer.py --topic "topic" --tts-model "tts_models/en/ljspeech/tacotron2-DDC"
```

### Image Generation
```bash
python run_influencer.py --topic "topic" --image-model "runwayml/stable-diffusion-v1-5"
```

### Video Generation

#### Stable Video Diffusion (Default)
```bash
python run_influencer.py --topic "topic" --video-mode img2vid --img2vid-model "stabilityai/stable-video-diffusion-img2vid-xt"
```

#### Hunyuan Video Framepack
```bash
python run_influencer.py --topic "topic" --video-mode framepack \
    --framepack-transformer "lllyasviel/FramePackI2V_HY" \
    --framepack-sampling "inverted_anti_drifting"
```

With first and last frame control (requires generating a last frame):
```bash
python run_influencer.py --topic "topic" --video-mode framepack --enable-last-frame
```

#### Text-to-Video (ModelScope)
```bash
python run_influencer.py --topic "topic" --video-mode text2vid --text2vid-model "damo-vilab/text-to-video-ms-1.7b"
```

## Custom Model Implementations

You can provide your own custom implementations for any component of the system. This allows you to:

- Use models not directly supported
- Customize the behavior or parameters of a model
- Implement completely different approaches

### Using Custom Implementations

Specify the path to your custom implementation file:

```bash
python run_influencer.py --topic "topic" \
    --text-model-impl "path/to/my_text_model.py" \
    --tts-model-impl "path/to/my_tts_model.py"
```

### Creating Custom Implementations

Create a Python file with the same function names and signatures as the default implementation:

```python
# my_text_model.py
def load_llm(model_id, device):
    # Your custom implementation...
    return model, tokenizer

def generate_script_and_prompts(topic, model, tokenizer, config):
    # Your custom implementation...
    return narration_scenes, visual_prompts, hashtags
```

See the `examples/` directory for sample implementations.

## Voice Cloning

For XTTS voice cloning, provide a reference audio file:

```bash
python run_influencer.py --topic "topic" --speaker-wav my_voice.wav
```

The reference audio should be a clean WAV file (16-bit PCM), ideally longer than 15 seconds.

## Directory Structure

```
influencer/
├── config.py               # Configuration classes
├── main.py                 # Main application logic
├── cli.py                  # Command-line interface
├── models/                 # Model handlers
│   ├── text/               # Text generation models
│   ├── audio/              # Audio generation models
│   ├── image/              # Image generation models
│   └── video/              # Video generation models
│       ├── img2vid.py      # Stable Video Diffusion
│       ├── text2vid.py     # ModelScope
│       └── framepack.py    # Hunyuan Video Framepack
├── generators/             # Content generators
│   └── video_assembly.py   # Final video assembly
├── utils/                  # Utility functions
│   └── memory.py           # Memory management
└── examples/               # Example custom implementations
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
