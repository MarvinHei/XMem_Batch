import os
import cv2
import numpy as np
from PIL import Image
import shutil

def dilate_and_adjust_framerate(dir, out, orig_frame, dst_frame, file_format, dilate_factor=20):
    files = os.listdir(dir)
    if not os.path.exists(out):
        os.mkdir(out)
    frame_fac = int(orig_frame/dst_frame)
    for file in files:
        if int(file.split('.')[0]) % frame_fac == 0:
            new_name = str(int(int(file.split('.')[0])/frame_fac)).zfill(7) + file_format
            mask = cv2.imread(os.path.join(dir, file))
            mask = mask.astype(np.uint8)
            mask = cv2.dilate(
                mask,
                np.ones((dilate_factor, dilate_factor), np.uint8),
                iterations=1
            )
            img_mask = Image.fromarray(mask, 'RGB')
            img_mask.save(os.path.join(out, new_name))
            print("saved mask ", file)

def crop_framerate(dir, out, orig_frame, dst_frame, file_format):
    files = os.listdir(dir)
    if not os.path.exists(out):
        os.mkdir(out)
    if orig_frame < dst_frame:
        return
    frame_fac = int(orig_frame/dst_frame)
    files = os.listdir(dir)
    for i in range(len(files)):
        if files[i].endswith(file_format) and int(files[i].split('.')[0]) % frame_fac == 0:
            new_name = str(int(int(files[i].split('.')[0])/frame_fac)).zfill(7) + file_format
            print(new_name)
            shutil.copy(os.path.join(dir, files[i]), os.path.join(out, new_name))

            


out = "dilated_hands_20"
dir = "workspace/P01_01/masks"
file_format = '.png'
#dilate_and_adjust_framerate(out, dir, 50, 10)
dilate_and_adjust_framerate(dir, out, 50, 10, file_format)




