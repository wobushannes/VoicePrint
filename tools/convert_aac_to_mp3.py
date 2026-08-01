#!/usr/bin/env python3
"""
AAC to MP3 Converter
Usage: python convert_aac_to_mp3.py /path/to/folder
"""

import os
import sys
import subprocess
from pathlib import Path


def convert_aac_to_mp3(input_folder):
    """
    Converts all .aac files in the given folder to .mp3 using ffmpeg.
    """
    folder = Path(input_folder)
    
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)
    
    aac_files = list(folder.glob("*.aac")) + list(folder.glob("*.m4a"))
    
    if not aac_files:
        print(f"⚠️  No .aac or .m4a files found in '{input_folder}'.")
        return
    
    print(f"📁 Found {len(aac_files)} file(s) to convert.")
    
    for aac_file in aac_files:
        output_file = aac_file.with_suffix(".mp3")
        
        # Skip if MP3 already exists
        if output_file.exists():
            print(f"⏭️  Skipping {aac_file.name} -> {output_file.name} (already exists)")
            continue
        
        print(f"🔄 Converting {aac_file.name} -> {output_file.name} ...")
        
        try:
            subprocess.run([
                "ffmpeg",
                "-i", str(aac_file),
                "-acodec", "mp3",
                "-ab", "192k",
                "-y",  # overwrite output if exists (but we already checked)
                str(output_file)
            ], check=True, capture_output=True, text=True)
            print(f"✅ Done: {output_file.name}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting {aac_file.name}:")
            print(e.stderr)
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg:")
            print("   macOS: brew install ffmpeg")
            print("   Ubuntu: sudo apt install ffmpeg")
            print("   Windows: download from https://ffmpeg.org/")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_aac_to_mp3.py /path/to/folder")
        sys.exit(1)
    
    convert_aac_to_mp3(sys.argv[1])