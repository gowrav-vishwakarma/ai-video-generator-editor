#!/usr/bin/env python
"""
Influencer - Instagram Content Generation Tool

Usage examples:
python run_influencer.py --topic "benefits of meditation" --output-dir output
python run_influencer.py --topic "healthy breakfast ideas" --video-mode text2vid
python run_influencer.py --topic "workout motivation" --speaker-wav my_voice.wav
"""

import sys
import os

# Add the current directory to sys.path to make the local influencer package importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from influencer.cli import main

if __name__ == "__main__":
    main() 