import math
import numpy as np
import os
import shutil

def restructure_folder(dirs, out):
	folders = os.listdir(dirs)
	folders.sort()
	max_len = int(math.floor(len(folders)/4))
	if not os.path.exists(out):
		os.makedirs(out)
	for i in range(max_len):
		for j in range(4):
			folder_name = folders[i * 4 + j]
			folder_path = os.path.join(dirs, folder_name)
			files = os.listdir(folder_path)
			files.sort()
			for k, file in enumerate(files):
				out_folder = os.path.join(out, str(i * 4 * len(files) + k))
				if not os.path.exists(out_folder):
					os.makedirs(out_folder) 
				file_path_in = os.path.join(folder_path, file)
				file_path_out = os.path.join(out_folder, file)
				shutil.copy(file_path_in, file_path_out)


def add_0_to_folder(dirs):
	folders = os.listdir(dirs)
	for folder in folders:
		folder_path = os.path.join(dirs, folder)
		new_name = folder.zfill(7)
		new_path = os.path.join(dirs, new_name)
		os.rename(folder_path, new_path)

def add_raw_to_dir(dirs, raw_dir):
	folders = os.listdir(dirs)
	for folder in folders:
		folder_path = os.path.join(dirs, folder)
		files = os.listdir(folder_path)
		mask_dir = os.path.join(folder_path, "masks")
		new_raw_dir = os.path.join(folder_path, "raw")
		if not os.path.exists(mask_dir):
			os.mkdir(mask_dir)
		if not os.path.exists(new_raw_dir):
			os.mkdir(new_raw_dir)
		files = [f for f in files if f != "masks" and f != "raw"]
		for file in files:
			print(file)
			file_path = os.path.join(folder_path, file)
			mask_path = os.path.join(mask_dir, file)
			shutil.move(file_path, mask_path)
			raw_file_name = file.split('.')[0] + ".jpg"
			raw_path = os.path.join(raw_dir, raw_file_name)
			raw_path_agent = os.path.join(new_raw_dir, raw_file_name)
			shutil.copy(raw_path, raw_path_agent)

dirs = "segmentations/P17_01/hand/both"
out = "agent_data/P17_01"
raw = "../EPIC_DATA/frames/P17/P17_01/"
add_0_to_folder(dirs)
restructure_folder(dirs, out)
add_raw_to_dir(out, raw)
