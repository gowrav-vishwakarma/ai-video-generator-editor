#!/usr/bin/env python3
"""
MP3 to WAV Converter
Converts all MP3 files in the Downloads folder to WAV format.
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment

def convert_mp3_to_wav(downloads_folder="~/Downloads", output_folder=None):
    """
    Convert all MP3 files in the Downloads folder to WAV format.
    
    Args:
        downloads_folder (str): Path to the Downloads folder
        output_folder (str): Path to output folder (defaults to same as input)
    """
    # Expand the tilde to full path
    downloads_path = Path(downloads_folder).expanduser()
    
    if output_folder is None:
        output_path = downloads_path
    else:
        output_path = Path(output_folder).expanduser()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all MP3 files
    mp3_files = list(downloads_path.glob("*.mp3"))
    
    if not mp3_files:
        print("No MP3 files found in the Downloads folder.")
        return
    
    print(f"Found {len(mp3_files)} MP3 file(s) to convert:")
    for mp3_file in mp3_files:
        print(f"  - {mp3_file.name}")
    
    print("\nStarting conversion...")
    
    converted_count = 0
    failed_count = 0
    
    for mp3_file in mp3_files:
        try:
            print(f"Converting: {mp3_file.name}")
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(str(mp3_file))
            
            # Create output filename (replace .mp3 with .wav)
            wav_filename = mp3_file.stem + ".wav"
            wav_path = output_path / wav_filename
            
            # Export as WAV
            audio.export(str(wav_path), format="wav")
            
            print(f"  ✓ Successfully converted to: {wav_filename}")
            converted_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed to convert {mp3_file.name}: {str(e)}")
            failed_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} files")
    if failed_count > 0:
        print(f"Failed conversions: {failed_count} files")

def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MP3 files to WAV format")
    parser.add_argument("--input", "-i", default="~/Downloads", 
                       help="Input folder containing MP3 files (default: ~/Downloads)")
    parser.add_argument("--output", "-o", 
                       help="Output folder for WAV files (default: same as input folder)")
    
    args = parser.parse_args()
    
    try:
        convert_mp3_to_wav(args.input, args.output)
    except KeyboardInterrupt:
        print("\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
