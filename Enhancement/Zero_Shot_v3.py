import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from skimage import img_as_ubyte

# Ensure multinex modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

def forward_ensemble(net, x):
    """
    Applies 8 geometric transformations to the input, passes each through the network,
    applies the inverse transformations, and averages the results for maximum quality.
    """
    def _transform(img, mode):
        if mode == 0: return img
        elif mode == 1: return img.flip(2)
        elif mode == 2: return img.flip(3)
        elif mode == 3: return img.flip(2).flip(3)
        elif mode == 4: return img.transpose(2, 3)
        elif mode == 5: return img.transpose(2, 3).flip(2)
        elif mode == 6: return img.transpose(2, 3).flip(3)
        elif mode == 7: return img.transpose(2, 3).flip(2).flip(3)

    def _inverse_transform(img, mode):
        if mode == 0: return img
        elif mode == 1: return img.flip(2)
        elif mode == 2: return img.flip(3)
        elif mode == 3: return img.flip(2).flip(3)
        elif mode == 4: return img.transpose(2, 3)
        elif mode == 5: return img.flip(2).transpose(2, 3)
        elif mode == 6: return img.flip(3).transpose(2, 3)
        elif mode == 7: return img.flip(3).flip(2).transpose(2, 3)

    res = []
    for i in range(8):
        out = net(_transform(x, i))
        res.append(_inverse_transform(out, i))
        
    return torch.stack(res).mean(0)


def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_ensemble', action='store_true', help='Use x8 self-ensemble for maximum image quality.')
    args = parser.parse_args()

    # 1. Setup Input/Output Directories
    input_dir = 'data/Sewer_ValTest'  
    out_dir = 'Results/Sewer_Results_LOLv2real' + ('ENS' if args.self_ensemble else '')
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Use your Custom Config and Weights
    opt_path = 'Options/multinex_lol-v2-real.yaml' 
    weights_path = 'pretrained_weights/Multinex_LOL-v2-real.pth' 
    
    # 3. Initialize Model
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
    
    if args.self_ensemble:
        print("⚠️ SELF-ENSEMBLE ENABLED: Maximizing Quality (FPS will drop significantly)")
    else:
        print("⚡ STANDARD INFERENCE: Maximizing Speed")

    # 4. GPU Warm-up
    print("Warming up the GPU...")
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    with torch.inference_mode():
        for _ in range(10):
            _ = model_restoration(dummy_input)
    torch.cuda.synchronize()
    
    # 5. Process Images
    image_paths = glob(os.path.join(input_dir, '*.*'))
    num_images = len(image_paths)
    if num_images == 0:
        print(f"No images found in {input_dir}.")
        return
        
    factor = 2 
    total_inference_time = 0.0
    
    for img_path in image_paths:
        # Load and pad
        img = np.float32(utils.load_img(img_path)) / 255.
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img_t.unsqueeze(0).cuda()
        
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        
        # --- START BENCHMARK ---
        torch.cuda.synchronize() 
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            if args.self_ensemble:
                restored = forward_ensemble(model_restoration, input_)
            else:
                restored = model_restoration(input_)
            
        torch.cuda.synchronize() 
        end_time = time.perf_counter()
        # --- END BENCHMARK ---
        
        total_inference_time += (end_time - start_time)
            
        # Unpad and save
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        save_path = os.path.join(out_dir, os.path.basename(img_path))
        utils.save_img(save_path, img_as_ubyte(restored))

    # 6. Generate Report
    avg_time_per_image = total_inference_time / num_images
    fps = 1.0 / avg_time_per_image
    
    mode_str = "MAX QUALITY (x8 Ensemble)" if args.self_ensemble else "REAL-TIME (Standard)"
    
    print("\n" + "="*40)
    print(" 📊 INFERENCE BENCHMARK REPORT")
    print("="*40)
    print(f"Testing Mode           : {mode_str}")
    print(f"Total Images Processed : {num_images}")
    print(f"Total Parameters       : {total_params:,}")
    print(f"Total Inference Time   : {total_inference_time:.4f} seconds")
    print(f"Average Time / Image   : {avg_time_per_image:.4f} seconds/img")
    print(f"Inference Speed (FPS)  : {fps:.2f} Frames Per Second")
    print("="*40)

if __name__ == '__main__':
    main()