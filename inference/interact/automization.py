"""
Based on https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN 
(which is based on https://github.com/seoungwugoh/ivs-demo)
"""
import os

import numpy as np
import shutil
import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')

from model.network import XMem

from inference.inference_core import InferenceCore

from .interactive_utils import *
from .interaction import *
from .resource_manager import ResourceManager
from .gui_utils import *


class Automator():
    def __init__(self, net: XMem, 
                resource_manager: ResourceManager, config, device):
        
        self.num_objects = config['num_objects']
        self.config = config
        self.processor = InferenceCore(net, config)
        self.processor.set_all_labels(list(range(1, self.num_objects+1)))
        self.res_man = resource_manager
        self.device = device

        self.num_frames = len(self.res_man)
        self.height, self.width = self.res_man.h, self.res_man.w

        # current frame info
        self.curr_frame_dirty = False
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8) 
        self.current_image_torch = None
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_prob = torch.zeros((self.num_objects, self.height, self.width), dtype=torch.float).to(self.device)
        
        files = os.listdir(os.path.join(config["workspace"], 'masks'))
        files = [int(file.split('.')[0]) for file in files]
        files.sort()
        self.mask_list = files
        self.cursur = files[0]

        self.alt_workspace = os.path.join("segmentations", config["workspace"][12:])
        print(self.alt_workspace)

        self.propagating = False
        self.out = None

    def automate(self):
        for ti in self.mask_list:
            self.out = os.path.join(self.alt_workspace, str(ti))
            self.cursur = ti
            self.load_current_image_mask()
            self.save_current_mask()
            self.on_forward_propagation(10)
            self.cursur = ti
            self.load_current_image_mask()
            self.save_current_mask()
            self.on_backward_propagation(10)
            self.on_clear_memory()

    def inpaint_prediction(self):
        for ti in self.mask_list:
            self.cursur = ti
            self.load_current_image_mask()
            self.on_forward_propagation(1)
            self.on_clear_memory()

    def automate_whole_video(self):
        for i in range(len(self.mask_list)-1):
            self.cursur = self.mask_list[i]
            self.load_current_image_mask()
            self.on_forward_propagation(self.mask_list[i+1] - self.mask_list[i] -1)
            self.on_clear_memory()
        

    def load_current_image_mask(self, no_mask=False):
        self.current_image = self.res_man.get_image(self.cursur)
        self.current_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.cursur, (self.height, self.width))
            if loaded_mask is None:
                self.current_mask.fill(0)
            else:
                self.current_mask = loaded_mask.copy()
            self.current_prob = None

    def load_current_torch_image_mask(self, no_mask=False):
        if self.current_image_torch is None:
            self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image, self.device)

        if self.current_prob is None and not no_mask:
            self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).to(self.device)

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.cursur, self.current_mask)
        if self.out != None:
            self.res_man.save_mask_custom(self.cursur, self.current_mask, self.out)

    def on_forward_propagation(self, len):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_next_frame
            self.on_propagation(len)

    def on_backward_propagation(self, len):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_prev_frame
            self.on_propagation(len)

    def on_pause(self):
        self.propagating = False
        

    def on_propagation(self, len):
        # start to propagate
        self.load_current_torch_image_mask()

        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

        self.propagating = True
        # propagate till the end
        for i in range(len):
            self.propagate_fn()

            self.load_current_image_mask(no_mask=True)
            self.load_current_torch_image_mask(no_mask=True)

            self.current_prob = self.processor.step(self.current_image_torch)
            self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

            self.save_current_mask()

            if self.cursur == 0 or self.cursur == self.num_frames-1:
                break

        self.propagating = False
        self.curr_frame_dirty = False
        self.on_pause()
    '''
    def on_propagation_whole_video(self):
        # start to propagate
        self.load_current_image_mask()
        self.load_current_torch_image_mask()

        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

        self.propagating = True
        # propagate till the end
        while self.propagating:
            self.on_next_frame()
            if self.cursur in self.mask_list:
                self.on_clear_memory()
                
            print(self.cursur)
            self.load_current_image_mask(no_mask=True)
            self.load_current_torch_image_mask(no_mask=True)

            self.current_prob = self.processor.step(self.current_image_torch)
            self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

            self.save_current_mask()

            if self.cursur == 0 or self.cursur == self.num_frames-1:
                break

        self.propagating = False
        self.curr_frame_dirty = False
        self.on_pause()
    '''

    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1)

    def on_next_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            mps.empty_cache()
