# data folders was not organized as desired in the dataset folder.
# script makes validation data organized as desired.


import os
import shutil

# Define the input folder
input_folder = "traffic_data/test"
destination_folder = "traffic_data/test_foldered"

# Get all files in the input folder
files = os.listdir(input_folder)

# Create a dictionary to store file paths based on folder names
folders = {}

# Organize files into folders based on the first three symbols before underscore
for filename in files:
    if "_" in filename:
        prefix = filename.split("_")[0]
        if prefix.isdigit():
            folder_name = str(int(prefix)) # Ensure folder name is 3 digits
            folder_path = os.path.join(destination_folder, folder_name)
            if folder_name not in folders:
                os.makedirs(folder_path, exist_ok=True)
                folders[folder_name] = folder_path
            file_path = os.path.join(input_folder, filename)
            shutil.move(file_path, folder_path)

print("Files organized into folders successfully.")
