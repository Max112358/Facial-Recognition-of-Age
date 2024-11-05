import os
import shutil

def extract_age(filename):
    # Split the filename into its components
    parts = filename.split('_')
    
    # Extract the dates from the filename
    date1_str = parts[1]
    date2_str = parts[2][:-4]
    
    #2201_1922-06-19_1955
    date1Parts = date1_str.split('-')
    date1 = date1Parts[0]
    date2 = date2_str
    
    # Calculate the age difference in years
    age = int(date2) - int(date1)
    
    #if int(date2) < 1970:
    #    return -500
    
    return age

def organize_files(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterate through the files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            # Extract the age from the filename
            age = extract_age(filename)
            
            # Create the target subdirectory if it doesn't exist
            target_subdir = os.path.join(target_dir, str(age))
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)
            
            # Copy the file to the target subdirectory
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_subdir, filename)
            shutil.copy(source_file, target_file)
            print(f"Copied {filename} to {target_subdir}")

# Example usage
source_dir = 'face_sorted2'
target_dir = 'catagorized_new'
organize_files(source_dir, target_dir)