import os
import shutil
import random
import math

# Set the source and destination folders
source_folder = 'larger_categories'
dest_folder = 'larger_categories_unseen'

# Create the destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Traverse the source folder and its subfolders
for root, dirs, files in os.walk(source_folder):
    # Get the relative path of the current folder
    rel_path = os.path.relpath(root, source_folder)
    
    # Create the corresponding subfolder in the destination folder
    dest_subfolder = os.path.join(dest_folder, rel_path)
    os.makedirs(dest_subfolder, exist_ok=True)

    # Get a list of JPG files in the current folder
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    
    # Calculate the number of files to move (2% of the total)
    num_files_to_move = max(1, math.ceil(len(jpg_files) * 0.02)) if jpg_files else 0

    # Randomly select the required number of JPG files
    selected_files = random.sample(jpg_files, num_files_to_move)

    # Move the selected files to the corresponding subfolder in the destination folder
    for file in selected_files:
        src_file = os.path.join(root, file)
        dest_file = os.path.join(dest_subfolder, file)
        shutil.move(src_file, dest_file)
        print(f'Moved {src_file} to {dest_file}')