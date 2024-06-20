import os
import shutil

def create_folder_structure(src_dir, dst_dir, file_format):
    # Ensure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Get a sorted list of all jpg files in the source directory
    files = sorted([f for f in os.listdir(src_dir) if f.endswith(file_format)])
    
    # Iterate through the files and copy them to the correct folder
    for index, filename in enumerate(files):
        # Determine the folder number
        print(filename.split('.'))
        folder_num = int(filename.split('.')[0])//50
        #folder_num = index // 1000
        folder_name = f'{folder_num:04d}'
        folder_path = os.path.join(dst_dir, folder_name)
        
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Source and destination file paths
        src_file_path = os.path.join(src_dir, filename)
        dst_file_path = os.path.join(folder_path, filename)
        
        # Copy the file
        shutil.copy(src_file_path, dst_file_path)
        print(f'Copied {filename} to {folder_path}')

# Example usage:
src_directory = 'P01_01_framerate_10'
dst_directory = 'sub_images'
create_folder_structure(src_directory, dst_directory, '.jpg')
src_directory = 'dilated_hands_20'
dst_directory = 'sub_masks'
create_folder_structure(src_directory, dst_directory, '.png')

