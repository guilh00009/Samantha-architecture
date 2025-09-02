#!/usr/bin/env python3
"""
Automatic checkpoint cleanup script for Samantha training
Keeps only the most recent checkpoints to save storage space
"""

import os
import glob
from pathlib import Path

def cleanup_old_checkpoints(max_checkpoints=3):
    """Keep only the most recent N checkpoints"""
    checkpoint_pattern = "samantha-17m-gpt2-rtx3050-step-*.pt"
    checkpoints = glob.glob(checkpoint_pattern)

    if len(checkpoints) <= max_checkpoints:
        print(f"Only {len(checkpoints)} checkpoints found, no cleanup needed")
        return

    # Sort by step number (extract numbers from filenames)
    def get_step_number(filename):
        # Extract number from "samantha-17m-gpt2-rtx3050-step-X"
        parts = filename.split('-')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    checkpoints.sort(key=get_step_number, reverse=True)  # Most recent first

    # Keep the most recent ones, delete the rest
    to_delete = checkpoints[max_checkpoints:]

    total_size_saved = 0
    for checkpoint in to_delete:
        try:
            size = os.path.getsize(checkpoint)
            os.remove(checkpoint)
            total_size_saved += size
            print(f"Deleted: {checkpoint}")
        except Exception as e:
            print(f"Failed to delete {checkpoint}: {e}")

    if total_size_saved > 0:
        print(".2f")

def cleanup_cache_dirs():
    """Clean up temporary cache directories"""
    cache_dirs = [
        './hf_cache_minimal',
        './wandb'
    ]

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                import shutil
                size_before = sum(os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, _, filenames in os.walk(cache_dir)
                                for filename in filenames) if os.path.exists(cache_dir) else 0

                # Only clean if directory is very large (>500MB)
                if size_before > 500 * 1024 * 1024:
                    shutil.rmtree(cache_dir)
                    print(f"Cleaned cache directory: {cache_dir} ({size_before / (1024*1024):.1f} MB)")
            except Exception as e:
                print(f"Failed to clean {cache_dir}: {e}")

if __name__ == "__main__":
    print("Running checkpoint cleanup...")
    cleanup_old_checkpoints(max_checkpoints=2)  # Keep only 2 most recent
    cleanup_cache_dirs()
    print("Cleanup complete!")
