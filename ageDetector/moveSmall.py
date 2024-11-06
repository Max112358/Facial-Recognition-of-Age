import os
import shutil

# Set the source and destination folders
source_folder = "small"
dest_folder = "very_small"

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is a JPG file
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
        # Get the full path of the file
        file_path = os.path.join(source_folder, filename)
        
        # Get the file size in bytes
        file_size = os.path.getsize(file_path)
        
        # Check if the file size is less than 2KB (2048 bytes)
        if file_size < 2048:
            # Move the file to the destination folder
            dest_path = os.path.join(dest_folder, filename)
            shutil.move(file_path, dest_path)
            print(f"Moved {filename} to {dest_folder}")