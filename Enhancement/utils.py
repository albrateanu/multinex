

import numpy as np
import os
import cv2
import math
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio


from basicsr.metrics.psnr_ssim import calculate_psnr
def PSNR(pred, gt, use_gtmean=False):
    """pred, gt ∈ [0,1]  H×W×C"""
    return calculate_psnr(pred, gt, crop_border=0, gtmean=use_gtmean)


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(pred,
                   gt,
                   border: int = 0,
                   gtmean: bool = False) -> float:
    """
    pred, gt : H×W×C, float or uint8, range [0-1] *or* [0-255]
               (prediction is the FIRST argument – same order as training)

    border   : pixels to crop away before computing SSIM
    gtmean   : if True, scale *pred* so its grey-level mean matches *gt*,
               exactly like `calculate_psnr(..., gtmean=True)` in Basicsr.
    """
    if pred.shape != gt.shape:
        raise ValueError('Input images must have the same dimensions.')

    # 1.  Optionally align global brightness of PREDICTION to GT
    if gtmean:
        # work in float32, regardless of original range
        p = pred.astype(np.float32).copy()
        g = gt.astype(np.float32)
        max_val = 1.0 if p.max() <= 1.0 else 255.0

        # grey means
        p_gray = p.mean(axis=2)
        g_gray = g.mean(axis=2)
        p_mean = p_gray.mean()
        g_mean = g_gray.mean()

        # avoid divide-by-zero
        scale = g_mean / (p_mean)
        pred = np.clip(p * scale, 0, max_val).astype(pred.dtype)

    # 2.  Crop border if requested
    h, w = pred.shape[:2]
    pred = pred[border:h - border, border:w - border]
    gt   = gt  [border:h - border, border:w - border]

    # 3.  Plain SSIM (per-channel then average)
    if pred.ndim == 2:            # single channel
        return _ssim(pred, gt)
    elif pred.ndim == 3:
        if pred.shape[2] == 3:    # RGB -> mean of per-channel SSIMs
            return np.mean([_ssim(pred[:, :, i], gt[:, :, i]) for i in range(3)])
        elif pred.shape[2] == 1:
            return _ssim(pred.squeeze(), gt.squeeze())

    raise ValueError('Unsupported image dimensions or number of channels.')


def _ssim(img1, img2):
    max_val = 1.0 if img1.max() <= 1.0 else 255.0
    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img, compression_level=0):
    """
    Saves an image in PNG format with compression.

    Args:
        filepath (str): Path to save the image.
        img (numpy.ndarray): Image in RGB format.
        compression_level (int, optional): PNG compression level (0-9).
            - 0 = No compression (largest file, fastest)
            - 9 = Maximum compression (smallest file, slowest)
    """
    # Convert RGB to BGR (OpenCV uses BGR format)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Save PNG with specified compression level
    cv2.imwrite(filepath, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)


def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)
