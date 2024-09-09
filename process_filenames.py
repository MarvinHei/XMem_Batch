import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        # Split filename by "_"
        parts = filename.split("_")
        
        # Search for the split string with at least three "0"
        for i, part in enumerate(parts):
            if part.count("0") >= 3:
                # Remove three "0" from the string
                new_part = part.replace("0", "", 3)
                # Merge parts back into filename
                new_filename = new_part
                
                # Rename the file
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed {filename} to {new_filename}")
                break

# Example usage:
#folder_path = "/home/marvin/EPIC_DATA/P01/P01_01"
#folder_path = "/home/marvin/Pipeline/object_masks"
folder_path = "segmentations/P01_05/hand/both"
folders = os.listdir(folder_path)
for folder in folders:
	new_path = os.path.join(folder_path, folder)
	rename_files(new_path)
