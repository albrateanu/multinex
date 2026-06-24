import os
import sys
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

def main():
    # 1. Setup Input/Output Directories
    # Updated to point to your specific test dataset input directory
    # input_dir = 'data/Sewer/Test/input'
    input_dir = 'sewer_samples' 
    out_dir = 'sewer_resultsv2'
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Use your newly trained Custom Config and Weights
    opt_path = 'Options/Multinex_Sewer.yaml' 
    weights_path = 'experiments/Multinex_Sewer/models/net_g_latest.pth' 
    
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
    
    # 4. Process Images
    image_paths = glob(os.path.join(input_dir, '*.*'))
    if len(image_paths) == 0:
        print(f"No images found in {input_dir}. Please add some sewer images!")
        return
        
    print(f"Found {len(image_paths)} images. Starting zero-shot inference...")
    factor = 2 # Padding factor required by the architecture
    
    for img_path in image_paths:
        # Load and format image
        img = np.float32(utils.load_img(img_path)) / 255.
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img_t.unsqueeze(0).cuda()
        
        # Pad image dimensions to fit model stride requirements
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        
        # Run Inference
        with torch.inference_mode():
            restored = model_restoration(input_)
            
        # Unpad and convert back to image format
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        # Save output
        save_path = os.path.join(out_dir, os.path.basename(img_path))
        utils.save_img(save_path, img_as_ubyte(restored))
        # print(f"Enhanced and saved: {save_path}")

    print(f"\nDone! Check the '{out_dir}' folder for your enhanced sewer images.")

if __name__ == '__main__':
    main()