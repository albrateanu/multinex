import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time # Added for benchmarking
from glob import glob
from skimage import img_as_ubyte

# Ensure multinex modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

def main():
    # 1. Setup Input/Output Directories
    input_dir = 'sewer_samples' 
    out_dir = 'sewer_results'
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Use your Custom Config and Weights
    opt_path = 'Options/multinexnano_lol-v2-real.yaml' 
    weights_path = 'pretrained_weights/multinex-nano_lol_v2_real.pth' 
    
    # 3. Initialize Model
    print("Loading Custom Trained Multinex Model...")
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    
    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(weights_path)
    
    # Handle DataParallel state_dict prefix if necessary
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)
        
    model_restoration.cuda().eval()
    
    # Calculate Model Parameters (Great for the report!)
    total_params = sum(p.numel() for p in model_restoration.parameters())
    print(f"Model loaded successfully! Total Parameters: {total_params:,}")
    
    # 4. GPU Warm-up (Crucial for accurate FPS)
    print("Warming up the GPU...")
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    with torch.inference_mode():
        for _ in range(10):
            _ = model_restoration(dummy_input)
    torch.cuda.synchronize()
    
    # 5. Process Images and Benchmark
    image_paths = glob(os.path.join(input_dir, '*.*'))
    num_images = len(image_paths)
    if num_images == 0:
        print(f"No images found in {input_dir}.")
        return
        
    print(f"Found {num_images} images. Starting zero-shot inference and benchmarking...")
    factor = 2 
    
    # Tracking variables
    total_inference_time = 0.0
    
    for img_path in image_paths:
        # Load and format image (Not counted in FPS)
        img = np.float32(utils.load_img(img_path)) / 255.
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img_t.unsqueeze(0).cuda()
        
        # Pad image
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        
        # --- START BENCHMARK ---
        torch.cuda.synchronize() # Wait for all previous ops to finish
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            restored = model_restoration(input_)
            
        torch.cuda.synchronize() # Wait for inference to fully finish
        end_time = time.perf_counter()
        # --- END BENCHMARK ---
        
        total_inference_time += (end_time - start_time)
            
        # Unpad and save (Not counted in FPS)
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        save_path = os.path.join(out_dir, os.path.basename(img_path))
        utils.save_img(save_path, img_as_ubyte(restored))

    # 6. Generate Supervisor Report
    avg_time_per_image = total_inference_time / num_images
    fps = 1.0 / avg_time_per_image
    
    print("\n" + "="*40)
    print(" 📊 INFERENCE BENCHMARK REPORT")
    print("="*40)
    print(f"Total Images Processed : {num_images}")
    print(f"Total Parameters       : {total_params:,}")
    print(f"Total Inference Time   : {total_inference_time:.4f} seconds")
    print(f"Average Time / Image   : {avg_time_per_image:.4f} seconds/img")
    print(f"Inference Speed (FPS)  : {fps:.2f} Frames Per Second")
    print("="*40)
    print(f"Images saved to: '{out_dir}'\n")

if __name__ == '__main__':
    main()