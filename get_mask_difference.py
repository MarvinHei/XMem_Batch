from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import os

def draw_affordance(rgb, output_path, affordance, alpha=0.5):
    
    aff = np.array(affordance)
    aff[np.where(aff != 0)] = 1

    # Creating kernel 
    kernel = np.ones((5, 5), np.uint8) 
    
    # Using cv2.erode() method  
    aff = cv2.erode(aff, kernel)  
    '''
    kernel = np.ones((5,5),np.float32)/25
    aff = cv2.filter2D(aff,-1,kernel)
    '''
    '''
    aff_min = aff.min()
    aff_max = aff.max()
    aff = (aff - aff_min) / (aff_max - aff_min)
    '''

    # Calculate the density of ones in each region of the binary mask
    density_map = np.zeros_like(aff, dtype=float)
    height, width = aff.shape
    for i in range(height):
        for j in range(width):
            if aff[i, j] == 1:
                for x in range(max(0, i-1), min(height, i+2)):
                    for y in range(max(0, j-1), min(width, j+2)):
                        density_map[x, y] += 1
    
    # Normalize the density map
    max_density = np.max(density_map)
    min_density = np.min(density_map)
    density_map = (density_map - min_density) / (max_density - min_density)
    aff = cv2.filter2D(aff,-1,kernel)
    '''
    rgb_gray = rgb.copy()
    rgb_gray[:, :, 0] = rgb.mean(axis=2)
    rgb_gray[:, :, 1] = rgb.mean(axis=2)
    rgb_gray[:, :, 2] = rgb.mean(axis=2)
    '''

    heatmap_img = cv2.applyColorMap((density_map * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:, :, ::-1]
    vis = cv2.addWeighted(heatmap_img, alpha, rgb, 1-alpha, 0)
    # vis = cv2.addWeighted(heatmap_img, alpha, rgb_gray, 1-alpha, 0)
    Image.fromarray(vis).save(output_path)
    #Image.fromarray(heatmap_img).save(output_path_3)

def delete_intersection(mask1_path, mask2_path, output_path):
    # Open the mask images
    mask1 = Image.open(mask1_path).convert('L')  # Convert to grayscale
    mask2 = Image.open(mask2_path).convert('L')  # Convert to grayscale

    # Convert images to numpy arrays
    mask1_array = np.array(mask1)
    mask2_array = np.array(mask2)

    mask1_array[np.where(mask1_array != 0)] = 1
    mask2_array[np.where(mask2_array != 0)] = 1
    

    # Find intersection of masks
    intersection = np.abs(mask2_array - mask1_array)

    # Delete intersection from both masks

    # Convert back to PIL images
    result_mask = Image.fromarray(intersection)
    
    # Save the result mask
    # result_mask.save(output_path)

    return result_mask

# Example usage:
'''
mask1_path = "workspace/0003682/masks/0000001.png"
mask2_path = "workspace/0003682/masks/0000002.png"
output_path = "affordances/0003682.png"
output_path_2 = "affordances/0003682_affordance.png"
output_path_3 = "affordances/0003682_affordance_only.png"
'''


basepath = "workspace"
folders = os.listdir(basepath)
#rgb = "workspace/P01_01/images/0003682.jpg"

for folder in folders:

    workspace_path = os.path.join(basepath, folder)
    rgb = os.path.join(workspace_path, 'images/0000001.jpg')
    rgb = Image.open(rgb)
    rgb = np.array(rgb)
    mask1_path = os.path.join(workspace_path, 'masks/0000000.png')
    mask2_path = os.path.join(workspace_path, 'masks/0000001.png')
    output_path = "affordances/" + folder + ".png"
    result_mask = delete_intersection(mask1_path, mask2_path, output_path)
    draw_affordance(rgb, output_path, result_mask)