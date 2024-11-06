import os
import shutil

# Define the source and destination directories
source_dir = 'catagorized_new'
dest_dir = 'larger_catagories2'

# Define the category ranges
categories = {
    '0-2': range(0, 3),
    '4-6': range(4, 7),
    '8-13': range(8, 14),
    '15-20': range(15, 21),
    '25-32': range(25, 33),
    '38-43': range(38, 44),
    '48-53': range(48, 54),
    '60+': range(60, 101)  # Assuming a maximum age of 100
}

# Create the destination folders if they don't exist
for category in categories:
    dest_category_dir = os.path.join(dest_dir, category)
    if not os.path.exists(dest_category_dir):
        os.makedirs(dest_category_dir)

# Iterate through the source directory
for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)
    if os.path.isdir(item_path):
        try:
            age = int(item)
            for category, age_range in categories.items():
                if age in age_range:
                    dest_category_dir = os.path.join(dest_dir, category)
                    dest_path = os.path.join(dest_category_dir, item)
                    shutil.copytree(item_path, dest_path)
                    print(f"Copied {item} to {dest_path}")
                    break
        except ValueError:
            # Ignore non-numeric folder names
            pass