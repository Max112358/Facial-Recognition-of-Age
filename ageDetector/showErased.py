import os
import shutil
from collections import defaultdict

def compare_and_copy_files(dir1, dir2, dest_dir):
    """
    Compare files in two directories (including subdirectories).
    Copy any files present in dir1 but missing from dir2 to dest_dir.
    All copied files are placed in the same destination folder.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Create dictionaries to store filenames and their paths
    dir1_files = defaultdict(list)
    dir2_files = defaultdict(list)

    # Populate the dictionaries with filenames and paths
    for root, _, files in os.walk(dir1):
        for file in files:
            file_path = os.path.join(root, file)
            dir1_files[file].append(file_path)

    for root, _, files in os.walk(dir2):
        for file in files:
            file_path = os.path.join(root, file)
            dir2_files[file].append(file_path)

    # Compare the dictionaries and copy missing files
    for filename, paths in dir1_files.items():
        if filename not in dir2_files:
            for path in paths:
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(path, dest_path)
                print(f"Copied {path} to {dest_path}")


# Example usage
dir1 = "wiki_crop"
dir2 = "catagorized"
dest_dir = "rejected"

compare_and_copy_files(dir1, dir2, dest_dir)