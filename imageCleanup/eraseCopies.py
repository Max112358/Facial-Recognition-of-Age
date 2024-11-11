import os

# Set the folder to search
folder_path = "data/non_face"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file has "Copy" in its name
    if "Copy" in filename:
        try:
            # Try to remove the file
            os.remove(file_path)
            print(f"Deleted {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")