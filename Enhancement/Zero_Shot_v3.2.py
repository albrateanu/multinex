import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from skimage import img_as_ubyte

# Import Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader

# Ensure multinex modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

# --- 1. Custom Dataset ---
class SewerInferenceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image (No padding here yet)
        img = np.float32(utils.load_img(img_path)) / 255.
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        
        _, h, w = img_t.shape
        return img_t, img_path, h, w

# --- 2. Custom Collation Function (The Magic Fix) ---
def dynamic_pad_collate(batch):
    """
    Takes a batch of mixed-resolution images, finds the largest dimensions,
    and dynamically pads the smaller images to form a uniform tensor cube.
    """
    factor = 2 # Required by the model architecture
    
    # Find the maximum height and width in this specific batch
    max_h = max([item[2] for item in batch])
    max_w = max([item[3] for item in batch])
    
    # Ensure the max dimensions are divisible by the model's stride factor
    H = ((max_h + factor - 1) // factor) * factor
    W = ((max_w + factor - 1) // factor) * factor
    
    padded_imgs = []
    paths = []
    orig_hs = []
    orig_ws = []
    
    for img, path, h, w in batch:
        padh = H - h
        padw = W - w
        
        # We use 'constant' padding (black pixels) instead of 'reflect'.
        # Reflect crashes if the padding needed is larger than the image itself (e.g., 360p to 1080p)
        img_padded = F.pad(img, (0, padw, 0, padh), mode='constant', value=0)
        
        padded_imgs.append(img_padded)
        paths.append(path)
        orig_hs.append(h)
        orig_ws.append(w)
        
    return torch.stack(padded_imgs), paths, orig_hs, orig_ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for GPU processing.')
    parser.add_argument('--num_workers', type=int, default=8, help='Background CPU threads.')
    args = parser.parse_args()

    # Directories
    input_dir = 'data/Sewer_samples'  # Directory containing input images
    out_dir = 'Results/Sewer_Results_NanoLOLv2real'
    os.makedirs(out_dir, exist_ok=True)
    
    # Configs
    opt_path = 'Options/multinexNano_lol-v2-real.yaml' 
    weights_path = 'pretrained_weights/Multinex-Nano_LOL_v2_real.pth' 
    
    print("Loading Custom Trained Multinex Model...")
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    
    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(weights_path)
    
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)
        
    model_restoration.cuda().eval()
    total_params = sum(p.numel() for p in model_restoration.parameters())
    print(f"Model loaded! Total Parameters: {total_params:,}")

    # GPU Warm-up
    print("Warming up the GPU...")
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    with torch.inference_mode():
        for _ in range(10):
            _ = model_restoration(dummy_input)
    torch.cuda.synchronize()
    
    # Process Images
    image_paths = glob(os.path.join(input_dir, '*.*'))
    num_images = len(image_paths)
    if num_images == 0:
        print(f"No images found in {input_dir}.")
        return
        
    # Setup DataLoader with Custom Collate
    dataset = SewerInferenceDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=dynamic_pad_collate # <--- This handles the mixed resolutions!
    )
    
    print(f"Starting inference (Batch Size: {args.batch_size}, Workers: {args.num_workers})...")
    
    total_inference_time = 0.0

    for batch_imgs, batch_paths, orig_hs, orig_ws in dataloader:
        
        # Move the padded batch to GPU
        batch_imgs = batch_imgs.cuda(non_blocking=True)
        
        # --- START BENCHMARK ---
        torch.cuda.synchronize() 
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            restored_batch = model_restoration(batch_imgs)
            
        torch.cuda.synchronize() 
        end_time = time.perf_counter()
        # --- END BENCHMARK ---
        
        total_inference_time += (end_time - start_time)
            
        # Post-Process: Unpad and save each image individually
        restored_batch = torch.clamp(restored_batch, 0, 1).cpu().detach()
        
        for i in range(len(batch_paths)):
            img_path = batch_paths[i]
            h = orig_hs[i]
            w = orig_ws[i]
            
            # Slice out the valid image (removing the black padding)
            restored_img = restored_batch[i, :, :h, :w].permute(1, 2, 0).numpy()
            
            save_path = os.path.join(out_dir, os.path.basename(img_path))
            utils.save_img(save_path, img_as_ubyte(restored_img))

    # Generate Report
    avg_time_per_image = total_inference_time / num_images
    fps = 1.0 / avg_time_per_image
    
    print("\n" + "="*45)
    print(" 📊 REAL-WORLD DATASET BENCHMARK REPORT")
    print("="*45)
    print(f"Hardware Constraint    : CPU/OS Bottleneck Overcome")
    print(f"Batch Size             : {args.batch_size}")
    print(f"Total Images Processed : {num_images} (Mixed Resolutions)")
    print(f"Total Inference Time   : {total_inference_time:.4f} seconds")
    print(f"Average Time / Image   : {avg_time_per_image:.4f} seconds/img")
    print(f"HONEST INFERENCE SPEED : {fps:.2f} Frames Per Second")
    print("="*45)

if __name__ == '__main__':
    main()