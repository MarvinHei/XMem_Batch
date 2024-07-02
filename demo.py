"""
A simple user interface for XMem
"""

import cv2
import numpy as np
import os
from os import path

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import sys
import shutil
from argparse import ArgumentParser

import torch

from model.network import XMem
from inference.interact.s2m_controller import S2MController
from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from inference.interact.automization import Automator
from inference.interact.resource_manager import ResourceManager
from contextlib import nullcontext
from PIL import Image

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def convert_mask_to_color(file):
    mask = cv2.imread(file)
    mask[np.where((mask ==[255, 255, 255]).all(axis=2))] = (255, 0, 0)
    Image.fromarray(mask).convert('L').save(file)

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename[0] != "P" and filename[0] != "f":
            print("files already configured")
            return
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

def restructure_folder(mask_type):
    workspace_path = os.path.join(config["workspace"], mask_type)
    mask_dst = os.path.join(workspace_path, "masks")
    if not path.exists(mask_dst):
        os.makedirs(mask_dst)
    video_name = config["images"].split('/')[-1]
    mask_path = os.path.join('../EPIC_DATA/segmentations', video_name, mask_type)
    if len(os.listdir(mask_dst)) == 0:
        for file in os.listdir(mask_path):
            shutil.copy(os.path.join(mask_path, file), os.path.join(mask_dst, file))
        rename_files(mask_dst)

    return workspace_path, video_name

def copy_files_from_ws_to_data(video_name, mask_type):
    mask_path = os.path.join(config["workspace"], "masks")
    img_path = os.path.join(config["workspace"], "images")
    shutil.rmtree(img_path)
    dst = os.path.join('../EPIC_DATA/xmem_masks', video_name, mask_type)
    if not os.path.exists(dst):
        os.makedirs(dst)
    for file in os.listdir(mask_path):
        shutil.copy(os.path.join(mask_path, file), os.path.join(dst, file))

if __name__ == '__main__':
    # Arguments parsing
    parser = ArgumentParser()
    parser.add_argument('--model', default='./saves/XMem.pth')
    parser.add_argument('--s2m_model', default='saves/s2m.pth')
    parser.add_argument('--fbrs_model', default='saves/fbrs.pth')

    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default=None)

    parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)

    parser.add_argument('--num_objects', type=int, default=1)

    # Long-memory options
    # Defaults. Some can be changed in the GUI.
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time',
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', type=int, default=10)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
    parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
    parser.add_argument('--size', default=480, type=int,
            help='Resize the shorter side to this size. -1 to use original resolution. ')
    args = parser.parse_args()

    # create temporary workspace if not specified
    config = vars(args)
    config['enable_long_term'] = True
    config['enable_long_term_count_usage'] = True

    if config["workspace"] is None:
        if config["images"] is not None:
            basename = path.basename(config["images"])
        elif config["video"] is not None:
            basename = path.basename(config["video"])[:-4]
        else:
            raise NotImplementedError(
                'Either images, video, or workspace has to be specified')

        config["workspace"] = path.join('./workspace', basename)

    with torch.cuda.amp.autocast(enabled=not args.no_amp) if device.type == 'cuda' else nullcontext():
        # Load our checkpoint
        network = XMem(config, args.model, map_location=device).to(device).eval()

        # Loads the S2M model
        if args.s2m_model is not None:
            s2m_saved = torch.load(args.s2m_model, map_location=device)
            s2m_model = S2M().to(device).eval()
            s2m_model.load_state_dict(s2m_saved)
        else:
            s2m_model = None

        s2m_controller = S2MController(s2m_model, args.num_objects, ignore_class=255, device=device)
        if args.fbrs_model is not None:
            fbrs_controller = FBRSController(args.fbrs_model, device=device)
        else:
            fbrs_controller = None

        # Manages most IO
        '''
        basepath = config["workspace"]
        folders = os.listdir(basepath)
        for folder in folders:
            config["workspace"] = os.path.join(basepath, folder)
            resource_manager = ResourceManager(config)
            ex = Automator(network, resource_manager, config, device)
            ex.inpaint_prediction()
        '''
        """ 
        resource_manager = ResourceManager(config)
        ex = Automator(network, resource_manager, config, device)
        ex.automate_whole_video()
        """

        mask_types = ["hand/left", "hand/right", "object/left", "object/right"]
        #mask_types = ["hand/both"]
        workspace = config["workspace"]

        for mask_type in mask_types:
            config["workspace"] = workspace
            workspace_path, video_name = restructure_folder(mask_type)
            config["workspace"] = workspace_path
            resource_manager = ResourceManager(config)
            ex = Automator(network, resource_manager, config, device)
            ex.automate()
            #ex.automate_whole_video()
        copy_files_from_ws_to_data(video_name, mask_type)
        mask_type = "hand/both"
        config["num_objects"] = 2
        config["workspace"] = workspace
        workspace_path, video_name = restructure_folder(mask_type)
        config["workspace"] = workspace_path
        resource_manager = ResourceManager(config)
        ex = Automator(network, resource_manager, config, device)
        ex.automate()
        #ex.automate_whole_video()
        copy_files_from_ws_to_data(video_name, mask_type)
