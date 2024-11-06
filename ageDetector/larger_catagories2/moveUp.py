import os
import shutil

# Set the folder path where you want to search for JPG files
folder_path = "."

# Loop through all subdirectories in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is a JPG
        if file.endswith(".jpg"):
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Construct the destination path (one level up)
            dest_path = os.path.join(os.path.dirname(root), file)
            # Move the file
            shutil.move(file_path, dest_path)