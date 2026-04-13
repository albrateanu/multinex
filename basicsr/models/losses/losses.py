import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from pytorch_msssim import ms_ssim, ssim
import torchvision.models as models

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]
        self.loss_model = vgg.to(device).eval()
        for p in self.loss_model.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, y_true, y_pred):
        device = next(self.loss_model.parameters()).device
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)

        y_true = self.normalize(y_true)
        y_pred = self.normalize(y_pred)

        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ssim(y_true, y_pred, data_range=max_val, size_average=True)

class HybridLoss(nn.Module):
    def __init__(self, w_pixloss=1.0, w_perc=0.01, w_msssim=0.2):
        super(HybridLoss, self).__init__()
        if w_pixloss != 0:
            self.pixel_loss = MSELoss()
        else:
            print('not using MSE Loss')
        
        if w_perc != 0:
            self.perceptual_loss_model = VGGPerceptualLoss
        else:
            print('not using Perc Loss')

        if w_msssim != 0:
            self.ms_ssim_loss = multiscale_ssim_loss
        else:
            print('not using SSIM Loss')
        

        self.w_pixloss = w_pixloss
        self.w_perc = w_perc
        self.w_msssim = w_msssim

    def forward(self, y_true, y_pred):
        total_loss = 0

        if self.w_pixloss != 0:
            pixel_l = self.pixel_loss(y_true, y_pred)
            total_loss += self.w_pixloss * pixel_l
        
        if self.w_perc != 0:
            if self.perceptual_loss_model is None:
                self.perceptual_loss_model = VGGPerceptualLoss(y_true.device).to(y_true.device)
            perc_l = self.perceptual_loss_model(y_true, y_pred)
            total_loss += self.w_perc*perc_l 
        
        if self.w_msssim != 0:
            ms_ssim_l = self.ms_ssim_loss(y_true, y_pred)
            total_loss += self.w_msssim*ms_ssim_l

        return torch.mean(total_loss)

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss



