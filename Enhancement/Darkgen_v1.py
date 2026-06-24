import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def synthesize_low_light(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    img = img.astype(np.float32) / 255.0
    
    # 1. Gamma Correction (darkens mid-tones non-linearly)
    gamma = np.random.uniform(2.0, 3.5)
    img = np.power(img, gamma)
    
    # 2. Linear Darkening (simulates low exposure)
    dark_val = np.random.uniform(0.15, 0.4)
    img = img * dark_val
    img = img * 255.0
    
    # 3. Add Gaussian Noise (simulates ISO sensor grain)
    noise_std = np.random.uniform(10, 25)
    noise = np.random.normal(0, noise_std, img.shape)
    img_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, img_noisy)

# Set your paths
gt_dir = 'data/Sewer_ValTest/GT'
lq_dir = 'data/Sewer_ValTest/Dark'
os.makedirs(lq_dir, exist_ok=True)

image_paths = glob(os.path.join(gt_dir, '*.*'))
print(f"Generating {len(image_paths)} synthetic dark images...")

for path in tqdm(image_paths):
    synthesize_low_light(path, os.path.join(lq_dir, os.path.basename(path)))