import os
import subprocess
import sys
import time

repo_dir = os.path.dirname(os.path.realpath(__file__))  # Fixed 'file' to '__file__'
chunk_size = 10  # Adjust this number for smaller or larger batches
max_retries = 3  # Maximum number of retry attempts
retry_delay = 5  # Delay in seconds between retries
size_limit = 100 * 1024 * 1024  # 100 MB in bytes

print(f"Script running in directory: {repo_dir}")

if not os.path.isdir(os.path.join(repo_dir, '.git')):
    print(f"Error: {repo_dir} is not a git repository.")
    sys.exit(1)

os.chdir(repo_dir)
print(f"Changed working directory to: {os.getcwd()}")

def run_git_command(command, retry=False):
    print(f"Running git command: {' '.join(command)}")
    
    for attempt in range(max_retries if retry else 1):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"Command output: {result.stdout.strip()}")
        print(f"Command error (if any): {result.stderr.strip()}")

        if result.returncode == 0:
            return True
        elif retry and attempt < max_retries - 1:
            print(f"Command failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
        else:
            return False

def get_untracked_and_modified_files():
    print("Getting untracked and modified files in the repository...")
    
    untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.splitlines()
    
    modified = subprocess.run(['git', 'ls-files', '--modified'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.splitlines()
    
    # Filter files larger than 100 MB
    files = []
    for file in untracked + modified:
        # Use os.path.normpath to handle paths correctly
        file_path = os.path.normpath(os.path.join(repo_dir, file))
        try:
            if os.path.getsize(file_path) <= size_limit:
                files.append(file)
            else:
                print(f"Ignoring file (exceeds 100MB): {file}")
        except OSError as e:
            print(f"Error accessing file {file_path}: {e}")

    print(f"Found {len(files)} untracked and modified files under 100 MB.")
    return files


def commit_and_push_in_chunks(files, chunk_size):
    total_files = len(files)
    
    for i in range(0, total_files, chunk_size):
        chunk = files[i:i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}: files {i + 1} to {i + len(chunk)}")
        
        for file in chunk:
            run_git_command(['git', 'add', file])
        
        commit_message = f"Adding files {i + 1} to {i + len(chunk)} out of {total_files}"
        
        if not run_git_command(['git', 'commit', '-m', commit_message]):
            print("Failed to commit changes.")
            return False
        
        if not run_git_command(['git', 'push', 'origin', 'main'], retry=True):
            print("Failed to push changes after multiple attempts. You may need to check your network connection or repository status.")
            return False
        
        print(f"Committed and pushed files {i + 1} to {i + len(chunk)} out of {total_files}")

    return True

print("Starting script execution...")
print("Current git status:")
run_git_command(['git', 'status'])

untracked_and_modified_files = get_untracked_and_modified_files()

if untracked_and_modified_files:
    print(f"Found {len(untracked_and_modified_files)} files to upload. Starting pushes...")
    
    if commit_and_push_in_chunks(untracked_and_modified_files, chunk_size):
        print("All files have been successfully committed and pushed in batches.")
    else:
        print("Failed to commit and push all files.")
        sys.exit(1)
else:
    print("No untracked or modified files found to commit.")

print("Script execution completed.")
