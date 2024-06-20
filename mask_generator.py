import functools

import os
from os import path
import cv2

import numpy as np
import torch
import shutil
import collections
import cv2
import progressbar
from PIL import Image
try:
    from torch import mps
except:
    print('torch.MPS not available.')

from model.network import XMem

from inference.inference_core import InferenceCore
from inference.interact.s2m_controller import S2MController
from inference.interact.fbrs_controller import FBRSController

from inference.interact.interactive_utils import *
from inference.interact.interaction import *
from inference.interact.resource_manager import ResourceManager
from inference.interact.gui_utils import *

from util.palette import davis_palette

if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image

class MaskGenerator:
    def __init__(self, masks_path, video_path, out_path, size):

        self.masks_path = masks_path
        self.video_path = video_path
        self.out_path = out_path
        self.size = size

        self.imported_mask = None
        self.height = None
        self.width = None
        self.current_mask = None
        self.current_image = None
        self.current_image_torch = None
        self.current_prob = None
        self.image_dir = out_path
        self.num_objects = 20
        self.timestamps = []

        for filename in os.listdir(masks_path):
            # Remove leading 0
            filename = filename.lstrip('0')
            index = filename.split('.')[0]
            self.timestamps.append(int(index))
        self.timestamps.sort()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
    def extract_frames(self, video):
        video = path.join(video_path, video)
        cap = cv2.VideoCapture(video)
        frame_index = 0
        video_sequence_index = 0
        os.mkdir(path.join(self.image_dir, str(video_sequence_index)))
        print(f'Extracting frames from {video} into {self.image_dir}...')
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            if self.size > 0:
                h, w = frame.shape[:2]
                self.height = h
                self.width = w
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
            if frame_index < self.timestamps[video_sequence_index]-50:
                frame_index += 1
                continue        
            cv2.imwrite(path.join(self.image_dir + '/' + str(video_sequence_index) ,f'{frame_index:07d}.jpg'), frame)
            frame_index += 1
            if frame_index == self.timestamps[video_sequence_index]+50:
                if video_sequence_index == len(self.timestamps)-1:
                    break
                else:
                    video_sequence_index +=1
                    os.mkdir(path.join(self.image_dir, str(video_sequence_index)))
            bar.update(frame_index)
        bar.finish()
        print('Done!')

    def read_external_image(self, file_name, size=None):
        image = Image.open(file_name)
        is_mask = image.mode in ['L', 'P']
        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), 
                    resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
        image = np.array(image)
        return image
    
    def get_time_frome_file(self, file_name):
        file_name = file_name.lstrip('0')
        index = file_name.split('.')[0]
        return index
    
    def import_mask(self, file_name):
        mask = self.read_external_image(file_name, size=(self.height, self.width))

        shape_condition = (
            (len(mask.shape) == 2) and
            (mask.shape[-1] == self.width) and 
            (mask.shape[-2] == self.height)
        )

        object_condition = (
            mask.max() <= self.num_objects
        )

        if not shape_condition:
            print(f'Expected ({self.height}, {self.width}). Got {mask.shape} instead.')
        elif not object_condition:
            print(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            print(f'Mask file {file_name} loaded.')
            self.current_image_torch = self.current_prob = None
            self.current_mask = mask
            self.save_current_mask(self.get_time_from_file(file_name))

    def load_current_torch_image_mask(self, no_mask=False):
        if self.current_image_torch is None:
            self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image, self.device)

        if self.current_prob is None and not no_mask:
            self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).to(self.device)

    def on_next_frame(self):
        self.current_frame += 1

    def propagate(self):
        # start to propagate
        self.load_current_torch_image_mask()

        print('Propagation started.')
        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
        # clear
        self.interacted_prob = None
        self.reset_this_interaction()

        propagating = True
        self.clear_mem_button.setEnabled(False)
        # propagate till the end
        while propagating:
            self.propagate_fn()

            self.load_current_image_mask(no_mask=True)
            self.load_current_torch_image_mask(no_mask=True)

            self.current_prob = self.processor.step(self.current_image_torch)
            self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

            self.save_current_mask()

            #self.update_memory_size() TODO: WHAT IS THE PURPOSE

            if self.cursur == 0 or self.cursur == self.num_frames-1:
                break

        propagating = False
    
    def forward_propagate(self):
        return
    
    def backward_propagate(self):
        return
    
    def save_current_mask(self, ti):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(self.current_mask, np.ndarray)

        mask = Image.fromarray(self.current_mask)
        mask.putpalette(self.palette)
        mask.save(path.join(self.masks_path, str(ti)+'.png'))
    
masks_path = 'masks'
video_path = 'videos'
out_path = 'out_2'
size = 480
mask_generator = MaskGenerator(masks_path, video_path, out_path, size)
mask_generator.extract_frames('P01_01.MP4')