import os
import shutil
import pandas as pd
import argparse

# Parse arguments from the command line
parser = argparse.ArgumentParser(description='Process some directories.')
parser.add_argument('--csv_dir', help='Directory to the CSV file')
parser.add_argument('--data_dir', help='Data directory containing directories')
args = parser.parse_args()

# Read data from CSV
excel_data = pd.read_csv(args.csv_dir)
print(excel_data)
excel_ids = set(excel_data['slide_id'].astype(str))
print("Image count before:", len(excel_ids))

# Identify IDs to remove based on a specific condition in the CSV
ids_to_remove = set(excel_data.loc[excel_data['Remove'] == 'DO_NOT_USE_TRAIN', 'slide_id'].astype(str))
excel_ids -= ids_to_remove
print("Image count after remove in csv:", len(excel_ids))

# Define the original folder path
folder_path = os.path.expanduser(args.data_dir)
folder_path_root = os.path.expanduser(args.data_dir)



# List all directories in the specified folder path
dirnames = os.listdir(folder_path)
dir_ids = set()

# Extract IDs and perform operations on directories
for dirname in dirnames:
    dir_path = os.path.join(folder_path, dirname)
    if os.path.isdir(dir_path):
        # Extract ID assuming ID is part of the directory name
        id = dirname.split('.')[0]  # Adjust extraction logic as needed
        dir_ids.add(id)

print(len(dir_ids))

# Reconciling and moving directories based on IDs in Excel
missing_in_excel = dir_ids - excel_ids
print('IDs missing in Excel:', missing_in_excel)

missing_in_dir = excel_ids - dir_ids
print('IDs missing in directory files:', missing_in_dir)

common_ids = dir_ids & excel_ids
print('IDs present in both Excel and directory files:', len(common_ids))


# Moving directories not in common_ids to another folder
output_folder = os.path.expanduser(folder_path_root + 'not_in_excel/')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for dirname in dirnames:
    dir_path = os.path.join(folder_path, dirname)
    if dirname.split('.')[0] not in common_ids:
        shutil.move(dir_path, os.path.join(output_folder, dirname))


# Move the rest of the directories to another folder called "images"
images_folder = os.path.expanduser(folder_path_root + 'images/')
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

for dirname in dirnames:
    dir_path = os.path.join(folder_path, dirname)
    if os.path.isdir(dir_path) and dirname.split('.')[0] in common_ids:
        shutil.move(dir_path, os.path.join(images_folder, dirname))




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

# Filter and save the modified Excel data
filtered_excel_data = excel_data[excel_data['slide_id'].astype(str).isin(common_ids)]
print(filtered_excel_data)
filtered_excel_data.to_csv(folder_path_root + '/filtered_images_clean.csv', index=False)
