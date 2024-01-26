import os
import shutil
import pandas as pd
import argparse

# Parse arguments from the command line
parser = argparse.ArgumentParser(description='Process some directories.')
parser.add_argument('--csv_dir', help='Directory to the CSV file')
parser.add_argument('--data_dir', help='Data directory containing directories')
args = parser.parse_args()

folder_path_root = os.path.expanduser(args.data_dir)
output_folder = os.path.expanduser(folder_path_root + 'not_in_excel/')


images_folder = os.path.expanduser(folder_path_root + 'images/')

# Loop through each directory in the root path
for directory in os.listdir(images_folder):
    dir_path = os.path.join(images_folder, directory)
    if os.path.isdir(dir_path):
        if dir_path.endswith(".1"):
            shutil.move(dir_path, output_folder)
        if dir_path.endswith(".2"):
            shutil.move(dir_path, output_folder)
        if dir_path.endswith(".3"):
            shutil.move(dir_path, output_folder)

        if dir_path.endswith(".svs"):
            # Extract ID (assuming the directory name contains the ID at the start)
            id_number = directory.split(".svs")[0]  # Modify this part as per your directory naming convention
            new_dir_path = os.path.join(images_folder, id_number)
            print(id_number,new_dir_path)
            # Rename directory
            
            os.rename(dir_path, new_dir_path)