import os
import sys
from pathlib import Path
from tqdm import tqdm

def clean_frames(directory):
    """
    Clean up the frames directory by:
    1. Removing non-image files
    2. Removing empty directories
    3. Verifying image files are valid
    """
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    total_removed = 0
    total_dirs_removed = 0
    
    # First pass: Remove non-image files
    print("Scanning for non-image files...")
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Remove files that aren't images
            if file_ext not in valid_extensions:
                try:
                    os.remove(file_path)
                    print(f"Removed non-image file: {file_path}")
                    total_removed += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    # Second pass: Remove empty directories and count valid images
    print("\nRemoving empty directories...")
    for root, dirs, files in os.walk(directory, topdown=False):
        # Skip the root directory
        if root == directory:
            continue
            
        # Check if directory is empty
        if not os.listdir(root):
            try:
                os.rmdir(root)
                print(f"Removed empty directory: {root}")
                total_dirs_removed += 1
            except Exception as e:
                print(f"Error removing directory {root}: {e}")
    
    # Third pass: Count remaining valid images
    print("\nCounting remaining valid images...")
    valid_images = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                valid_images += 1
    
    print(f"\nCleanup complete!")
    print(f"- Removed {total_removed} non-image files")
    print(f"- Removed {total_dirs_removed} empty directories")
    print(f"- Found {valid_images} valid image files remaining")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_frames.py <frames_directory>")
        sys.exit(1)
    
    frames_dir = sys.argv[1]
    if not os.path.isdir(frames_dir):
        print(f"Error: Directory not found: {frames_dir}")
        sys.exit(1)
    
    print(f"Starting cleanup of: {frames_dir}")
    clean_frames(frames_dir)
