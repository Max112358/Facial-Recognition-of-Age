import os

def find_highest_numbered_file(directory, prefix, extension):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter out files that don't match the prefix and extension
    matching_files = [f for f in files if f.startswith(prefix) and f.endswith(extension)]

    # Extract the numbers from the filenames
    numbers = [int(f.split('_')[2].split('.')[0]) for f in matching_files]

    # Find the highest number
    if numbers:
        highest_number = max(numbers)
        highest_file = f"{prefix}_{highest_number}.{extension}"
        return os.path.join(directory, highest_file)
    else:
        return None