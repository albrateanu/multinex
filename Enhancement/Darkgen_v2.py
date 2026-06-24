import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def synthesize_low_light(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    img = img.astype(np.float32) / 255.0
    
    # 1. Gamma Correction
    gamma = np.random.uniform(1.2, 2.2)
    img = np.power(img, gamma)
    
    # 2. Linear Darkening
    dark_val = np.random.uniform(0.3, 0.6)
    img = img * dark_val
    img = img * 255.0
    
    # 3. Add Gaussian Noise
    noise_std = np.random.uniform(7, 15)
    noise = np.random.normal(0, noise_std, img.shape)
    img_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, img_noisy)

# Set the single folder path where your images currently are, 
# and where you want the new dark ones to go.
single_folder_dir = 'data/Sewer_ValTest'

# Find all files in that folder
image_paths = glob(os.path.join(single_folder_dir, '*.*'))

# Optional: Filter out images that ALREADY have _DARK so you don't double-darken them if you run this twice
image_paths = [p for p in image_paths if '_DARK' not in p]

print(f"Generating {len(image_paths)} synthetic dark images...")

for path in tqdm(image_paths):
    # 1. Get just the filename (e.g., 'crack_001.jpg')
    base_filename = os.path.basename(path)
    
    # 2. Split into name and extension (e.g., 'crack_001' and '.jpg')
    name, ext = os.path.splitext(base_filename)
    
    # 3. Build the new filename (e.g., 'crack_001_DARK.jpg')
    new_filename = f"{name}_DARK{ext}"
    
    # 4. Construct the full output path back into your single folder
    output_path = os.path.join(single_folder_dir, new_filename)
    
    # 5. Generate and save the image
    synthesize_low_light(path, output_path)