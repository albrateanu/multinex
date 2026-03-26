import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
from typing import Optional, Dict

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


# ---------- tiny utils ----------
def make_act(act):
    import torch.nn as nn
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    if isinstance(act, str):
        name = act.lower()
        if name in ('silu', 'swish'):
            return nn.SiLU()
        if name in ('relu',):
            return nn.ReLU(inplace=True)
        if name in ('gelu',):
            return nn.GELU()
        if name in ('lrelu', 'leakyrelu'):
            return nn.LeakyReLU(0.1, inplace=True)
        if name in ('prelu',):
            return nn.PReLU()
        return nn.SiLU()  # default
    if isinstance(act, type):
        return act()  # a class like nn.SiLU
    return nn.SiLU()

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act='SiLU', bn=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = make_act(act)                     # <-- change here
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act='SiLU', bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = make_act(act)                     # <-- change here
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------- illumination stack (single-op maps) ----------

class IlluminationExtractor(nn.Module):
    """
    Returns K single-op illumination maps stacked along channel dim.
    Toggle which maps to use via 'use_flags'.
    """
    def __init__(self,
                 use_flags: Dict[str, bool] = None,
                 eps: float = 1e-6):
        super().__init__()
        # default: enable all six
        default = dict(mean=True, rec709=True, vmax=True, lightness=True, ycgco=True, l2norm=True)
        self.use = default if use_flags is None else {**default, **use_flags}
        self.eps = eps
        # fixed weights for Rec.709
        self.register_buffer("w_rec709", torch.tensor([0.2126, 0.7152, 0.0722]).view(1,3,1,1))
        # fixed weights for YCgCo Y = 1/4 R + 1/2 G + 1/4 B
        self.register_buffer("w_ycgco", torch.tensor([0.25, 0.5, 0.25]).view(1,3,1,1))

        # list order to keep channel order deterministic
        self.order = [k for k, v in self.use.items() if v]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB (B,3,H,W)"
        maps = []
        R, G, Bc = x[:,0:1], x[:,1:2], x[:,2:3]

        if self.use.get("mean", False):
            maps.append((R + G + Bc) / 3.0)
        if self.use.get("rec709", False):
            maps.append((x * self.w_rec709).sum(dim=1, keepdim=True))
        if self.use.get("vmax", False):
            maps.append(torch.maximum(R, torch.maximum(G, Bc)))
        if self.use.get("lightness", False):
            mx = torch.maximum(R, torch.maximum(G, Bc))
            mn = torch.minimum(R, torch.minimum(G, Bc))
            maps.append((mx + mn) / 2.0)
        if self.use.get("ycgco", False):
            maps.append((x * self.w_ycgco).sum(dim=1, keepdim=True))
        if self.use.get("l2norm", False):
            maps.append(torch.sqrt(R*R + G*G + Bc*Bc + self.eps))

        return torch.cat(maps, dim=1)  # (B, K, H, W)

# ---------- per-illum pooling conv attention ----------

class PoolConvAttention(nn.Module):
    """
    For each single-channel illum map:
      - pool (avg or max)
      - depthwise 3x3
      - upsample
      - sigmoid
    Process K maps independently using grouped convs for efficiency.
    """
    def __init__(self, K: int, pool_kernel: int = 8, use_max_pool: bool = False):
        super().__init__()
        self.K = K
        self.pool_kernel = pool_kernel
        self.use_max_pool = use_max_pool
        # Grouped depthwise conv (groups=K) over K channels
        self.dw = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, groups=K, bias=True)

    def forward(self, illum_stack: torch.Tensor) -> torch.Tensor:
        B, K, H, W = illum_stack.shape
        if self.use_max_pool:
            pooled = F.max_pool2d(illum_stack, kernel_size=self.pool_kernel, stride=self.pool_kernel, ceil_mode=True)
        else:
            pooled = F.avg_pool2d(illum_stack, kernel_size=self.pool_kernel, stride=self.pool_kernel, ceil_mode=True)
        att_coarse = self.dw(pooled)  # (B,K,h,w)
        att = F.interpolate(att_coarse, size=(H, W), mode='bilinear', align_corners=False)
        return att     # (B,K,H,W)

# ---------- main net ----------

# ---------- MSEF -------------
class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)
# 
    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
class SEBlock_AGAP(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, gap_size=(1, 1)):
        super(SEBlock_AGAP, self).__init__()
        self.gap_size = gap_size  # can be (1,1), (2,2), etc.
        reduced = max(1, input_channels // reduction_ratio)

        # Linear layers stay channel-wise
        self.fc1 = nn.Linear(input_channels * gap_size[0] * gap_size[1], reduced)
        self.fc2 = nn.Linear(reduced, input_channels * gap_size[0] * gap_size[1])

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        # adaptive pooling to (gap_H, gap_W)
        y = F.adaptive_avg_pool2d(x, self.gap_size)
        # flatten all but batch
        y = y.reshape(batch_size, -1)
        # MLP
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        # reshape back to (B, C, gap_H, gap_W)
        y = y.reshape(batch_size, num_channels, *self.gap_size)
        # upsample if GAP > 1
        if self.gap_size != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return x * y


class MSEFBlock(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super(MSEFBlock, self).__init__()
        # self.layer_norm = LayerNormalization(ch)
        self.depthwise_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.se_attn = SEBlock(ch, reduction_ratio)
        # self.se_attn = SEBlock_AGAP(ch, reduction_ratio, gap_size=(reduction_ratio,reduction_ratio))

    def forward(self, x):
        # x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x)
        x2 = self.se_attn(x)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out

# ---------- MSEF -------------

# ---- MDTA ----

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# --------

class ChrominanceExtractor(nn.Module):
    """
    Returns K_C chrominance maps stacked along channel dim.
    Toggle which maps to use via 'use_flags'.
    All maps are formula-based & fast.
    """
    def __init__(self, use_flags: Dict[str, bool] = None, eps: float = 1e-6):
        super().__init__()
        default = dict(
            yuv_uv=True,       # U, V (BT.601)
            ycbcr_cbcr=True,   # Cb, Cr (BT.601)
            opponent=True,     # O1, O2
            chroma_rg=True,    # r, g (chromaticity)
            hsv_s=True,         # saturation S
            # litags=True,
        )
        self.use = default if use_flags is None else {**default, **use_flags}
        self.eps = eps

        # Fixed matrices (BT.601)
        # YUV (Y = 0.299R + 0.587G + 0.114B), we only use U,V
        self.register_buffer("yuv_U", torch.tensor([-0.14713, -0.28886, 0.436]).view(1,3,1,1))
        self.register_buffer("yuv_V", torch.tensor([ 0.61500, -0.51499, -0.10001]).view(1,3,1,1))

        # YCbCr luma coeffs for Y (reuse from Illum if needed); for Cb, Cr we can use direct linear forms:
        self.register_buffer("ycbcr_Cb", torch.tensor([-0.168736, -0.331264, 0.5]).view(1,3,1,1))
        self.register_buffer("ycbcr_Cr", torch.tensor([ 0.5,      -0.418688, -0.081312]).view(1,3,1,1))

        # Opponent color space
        self.register_buffer("opp_O1", torch.tensor([ 1.0, -1.0,  0.0]).view(1,3,1,1) / (2**0.5))
        self.register_buffer("opp_O2", torch.tensor([ 1.0,  1.0, -2.0]).view(1,3,1,1) / (6**0.5))

        # LITA-GS Structure Prior
        # self.extraction_model = CIConv2d('W', k=3, scale=0.8)
        # self.extraction_model = self.extraction_model.cuda()
        

        # Keep deterministic channel order
        self.order = []
        if self.use.get('yuv_uv', False):
            self.order += ['U','V']
        if self.use.get('ycbcr_cbcr', False):
            self.order += ['Cb','Cr']
        if self.use.get('opponent', False):
            self.order += ['O1','O2']
        if self.use.get('chroma_rg', False):
            self.order += ['r','g']
        if self.use.get('hsv_s', False):
            self.order += ['S']
        # if self.use.get('litags', False):
        #     print('Chrominance Extractor: Leveraging LITA-GS Structure Prior')
        #     self.order +=['litags']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB (B,3,H,W)"
        R, G, Bc = x[:,0:1], x[:,1:2], x[:,2:3]
        maps = []

        # YUV U,V (no Y)
        if self.use.get('yuv_uv', False):
            maps.append((x * self.yuv_U).sum(1, keepdim=True))  # U
            maps.append((x * self.yuv_V).sum(1, keepdim=True))  # V

        # YCbCr Cb, Cr (linear forms)
        if self.use.get('ycbcr_cbcr', False):
            maps.append((x * self.ycbcr_Cb).sum(1, keepdim=True))  # Cb
            maps.append((x * self.ycbcr_Cr).sum(1, keepdim=True))  # Cr

        # Opponent O1,O2
        if self.use.get('opponent', False):
            maps.append((x * self.opp_O1).sum(1, keepdim=True))    # O1
            maps.append((x * self.opp_O2).sum(1, keepdim=True))    # O2

        # Chromaticity r,g
        if self.use.get('chroma_rg', False):
            denom = (R + G + Bc).clamp_min(self.eps)
            maps.append(R / denom)  # r
            maps.append(G / denom)  # g

        # HSV Saturation S = 1 - min/max
        if self.use.get('hsv_s', False):
            mx = torch.maximum(R, torch.maximum(G, Bc))
            mn = torch.minimum(R, torch.minimum(G, Bc))
            S = (mx - mn) / (mx.clamp_min(self.eps))  # 1 - mn/mx
            maps.append(S)
        
        # if self.use.get('litags', False):
        #     maps.append(self.extraction_model(x)) # LITAGS Structure Prior

        return torch.cat(maps, dim=1) if maps else torch.zeros(B,0,H,W, device=x.device, dtype=x.dtype)


# -------------------------------
# Retinex-style model
# -------------------------------
from mmdet.models.builder import BACKBONES  # <--- This is the missing line

@BACKBONES.register_module()
class MultinexNano(nn.Module):
    """
    Multinex with defaults updated to match 'network_g' YAML configuration.
    """
    def __init__(self,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 
                 # --- trunk width/depth ---
                 base_channels: int = 3,          # YAML: 40
                 width_mult: float = 1.0,          # YAML: 2.0
                 
                 # --- conv type ---
                 use_depthwise: bool = True,
                 act=nn.ReLU,                      # YAML: ReLU
                 
                 # --- attention projections ---
                 per_illum_proj: bool = False,     # YAML: false
                 per_chroma_proj: bool = False,    # YAML: false
                 
                 # --- attention toggles ---
                 use_illum_attn: bool = True,
                 use_chroma_attn: bool = True,
                 
                 # --- block counts ---
                 illum_mid: int = 1,               # YAML: 3
                 chroma_mid: int = 1,              # YAML: 3
                 # --- block counts ---
                #  illum_mid: int = 3,               # YAML: 3
                #  chroma_mid: int = 3,              # YAML: 3
                 
                 # --- MSEF ---
                 reduction_ratio: int = 3,         # YAML: 8
                 
                 # --- budget ---
                 target_params: Optional[int] = 300, # YAML: 45000
                # target_params: Optional[int] = 45000, # YAML: 45000
                 
                 # --- flags (dictionaries handled in logic below) ---
                 illum_flags: Optional[Dict[str, bool]] = None,
                 chroma_flags: Optional[Dict[str, bool]] = None,

                 # --- heads & fusion ---
                 luma_head_act: Optional[str] = None,    # YAML: null
                 chroma_head_act: Optional[str] = None,  # YAML: null
                 retinex_residual: bool = True,
                 eps: float = 1e-9,                      # YAML: 1.0e-9
                 
                 # --- MMDetection Args ---
                 pretrained=None, 
                 init_cfg=None
                 ):
        super().__init__()

        # --- Handle Flags Defaults matching YAML ---
        if illum_flags is None:
            illum_flags = {
                'mean': False, 
                'rec709': True, 
                'vmax': True, 
                'lightness': True, 
                'ycgco': False, 
                'l2norm': True
            }
        
        if chroma_flags is None:
            chroma_flags = {
                'yuv_uv': False, 
                'ycbcr_cbcr': True, 
                'opponent': False, 
                'chroma_rg': True, 
                'hsv_s': True
            }

        # --- MMDet Init ---
        self.pretrained = pretrained
        if init_cfg is not None:
            self.init_cfg = init_cfg

        # --- Standard Logic ---
        self.use_depthwise = use_depthwise
        self.per_illum_proj = per_illum_proj
        self.per_chroma_proj = per_chroma_proj
        self.reduction_ratio = reduction_ratio
        self.base_channels = base_channels
        self.width_mult = width_mult
        self.act = act
        self.eps = eps
        self.luma_head_act = luma_head_act
        self.chroma_head_act = chroma_head_act
        self.retinex_residual = retinex_residual
        self.use_illum_attn = use_illum_attn
        self.use_chroma_attn = use_chroma_attn

        # Initialize Extractors with the specific flags
        self.illum_extractor = IlluminationExtractor(illum_flags)
        self.chroma_extractor = ChrominanceExtractor(chroma_flags)

        # Calculate K
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 8, 8)
            K_L = self.illum_extractor(dummy).shape[1]
            K_C = self.chroma_extractor(dummy).shape[1]
        self.K_L, self.K_C = K_L, K_C

        # Calculate C based on Width Mult
        C = max(3, int(round(self.base_channels * self.width_mult)))
        self.C = C

        # Stems
        self.illum_stem  = nn.Conv2d(max(1, K_L), C, kernel_size=1, bias=True)
        self.chroma_stem = nn.Conv2d(max(1, K_C), C, kernel_size=1, bias=True)

        # Attention Convs
        self.illum_att = nn.Conv2d(max(1, K_L), max(1, K_L), kernel_size=7, stride=1, padding=3, groups=max(1, K_L), bias=True)
        self.chroma_att = nn.Conv2d(max(1, K_C), max(1, K_C), kernel_size=7, stride=1, padding=3, groups=max(1, K_C), bias=True)
        
        # Attention Projections
        if self.per_illum_proj:
            self.illum_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,K_L))])
        else:
            self.illum_att_proj = nn.Conv2d(max(1,K_L), C, 1, 1, 0, bias=True)

        if self.per_chroma_proj:
            self.chroma_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,K_C))])
        else:
            self.chroma_att_proj = nn.Conv2d(max(1,K_C), C, 1, 1, 0, bias=True)

        # Blocks (Depth determined by illum_mid / chroma_mid)
        # Note: Pre-seq logic in original code was tied to mid_seq length, assuming symmetric pre/mid or just mid.
        # Based on your YAML, we just have 'mid' depth. 
        # I will apply the depth count to both PRE and MID sequences as per the structure in your previous snippet.
        # self.illum_pre_seq = self._make_stack_blocks(C, C, depth=illum_mid)
        self.illum_mid_seq  = self._make_stack_blocks(C, C, depth=illum_mid)
        # self.chroma_pre_seq  = self._make_stack_blocks(C, C, depth=chroma_mid)
        self.chroma_mid_seq  = self._make_stack_blocks(C, C, depth=chroma_mid)

        # Heads
        self.head_luma   = nn.Conv2d(C, 1, kernel_size=1, bias=True)
        self.head_chroma = nn.Conv2d(C, out_ch, kernel_size=1, bias=True)
        # self.det_head = nn.Conv2d(2*39, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.head_luma.weight)
        nn.init.zeros_(self.head_luma.bias)

        nn.init.zeros_(self.head_chroma.weight)
        nn.init.zeros_(self.head_chroma.bias)
        # Auto-shrink parameters if budget is set
        if target_params is not None:
            self._fit_param_budget(target_params, in_ch, out_ch)
        print('total parameters:', sum(param.numel() for param in self.parameters()))

    # ---- helpers (unchanged except we don’t build BN/act gates now)
    def _make_block(self, c_in, c_out):
        if self.use_depthwise:
            return nn.Sequential(
                # MSEFBlock(c_in, self.reduction_ratio),
                DWSeparableConv(c_in, c_out, k=3, s=1, p=1, act=self.act),
                MSEFBlock(c_out, self.reduction_ratio),
            )
        else:
            return nn.Sequential(
                MSEFBlock(c_in, self.reduction_ratio),
                ConvBNAct(c_in, c_out, k=3, s=1, p=1, act=self.act),
                MSEFBlock(c_out, self.reduction_ratio),
            )

    def _make_stack_blocks(self, c_in, c_out, depth=1):
        return nn.Sequential(*[self._make_block(c_in, c_out) for _ in range(depth)]) if depth > 0 else nn.Identity()

    def _att_project_to_C(self, att, proj):
        # att: (B,K,H,W) ; proj: ModuleList or Conv2d
        if isinstance(proj, nn.ModuleList):
            out = 0.0
            for i in range(att.shape[1]):
                out = out + proj[i](att[:, i:i+1])
            return out
        else:
            return proj(att)

    def _apply_head_act(self, x, kind: Optional[str]):
        if kind is None:
            return x
        k = kind.lower() if isinstance(kind, str) else None
        if k == 'sigmoid': return torch.sigmoid(x)
        if k == 'tanh':    return torch.tanh(x)
        if k == 'relu':    return F.relu(x, inplace=False)
        return x

    def _fit_param_budget(self, target_params: int, in_ch: int, out_ch: int):
        # Rebuild all C-dependent parts when shrinking width
        lo, hi = 0.25, self.width_mult
        best = self.width_mult
        for _ in range(10):
            mid = (lo + hi) / 2
            self.width_mult = mid
            C = max(3, int(round(self.base_channels * self.width_mult)))
            self.C = C
            print(f'self.C = {self.C}')

            # stems
            self.illum_stem  = nn.Conv2d(max(1,self.K_L), C, 1, 1, 0, bias=True)
            self.chroma_stem = nn.Conv2d(max(1,self.K_C), C, 1, 1, 0, bias=True)

            # blocks
            # self.illum_pre_seq  = self._make_stack_blocks(C, C, depth=self.illum_mid_seq.__len__() if isinstance(self.illum_mid_seq, nn.Sequential) else 0)
            self.illum_mid_seq  = self._make_stack_blocks(C, C, depth=self.illum_mid_seq.__len__() if isinstance(self.illum_mid_seq, nn.Sequential) else 0)

            # self.chroma_pre_seq  = self._make_stack_blocks(C, C, depth=self.chroma_mid_seq.__len__() if isinstance(self.illum_mid_seq, nn.Sequential) else 0)
            self.chroma_mid_seq  = self._make_stack_blocks(C, C, depth=self.chroma_mid_seq.__len__() if isinstance(self.chroma_mid_seq, nn.Sequential) else 0)

            # heads
            self.head_luma   = nn.Conv2d(C, 1, kernel_size=1, bias=True)
            self.head_chroma = nn.Conv2d(C, out_ch, kernel_size=1, bias=True)

            # attn projections K->C
            if self.per_illum_proj:
                self.illum_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,self.K_L))])
            else:
                self.illum_att_proj = nn.Conv2d(max(1,self.K_L), C, 1, 1, 0, bias=True)

            if self.per_chroma_proj:
                self.chroma_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,self.K_C))])
            else:
                self.chroma_att_proj = nn.Conv2d(max(1,self.K_C), C, 1, 1, 0, bias=True)

            p = count_params(self)
            if p <= target_params:
                best = mid
                lo = mid
            else:
                hi = mid
        self.width_mult = best

    # ---- forward (with direct attention application) ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        rgb_in = x
        factor = 4
        x = F.max_pool2d(x, kernel_size=3, stride=factor, ceil_mode=True)
        # Stacks
        L_stack = self.illum_extractor(x)   # (B,K_L,H,W) or (B,0,.,.)
        if L_stack.shape[1] == 0:
            L_stack = x.new_zeros(B,1,H,W)
        C_stack = self.chroma_extractor(x)  # (B,K_C,H,W) or (B,0,.,.)
        if C_stack.shape[1] == 0:
            C_stack = x.new_zeros(B,1,H,W)

        # Stems
        fL = self.illum_stem(L_stack)       # (B,C,H,W)
        fC = self.chroma_stem(C_stack)      # (B,C,H,W)

        # fL = self.illum_pre_seq(fL)
        # fC = self.chroma_pre_seq(fC)

        if self.use_illum_attn:
            L_att = self.illum_att(L_stack)                       # (B,K_L,H,W)
            # save_tensor_channels(L_att, 'debug_output', prefix=f"L_att_")
            L_mask = torch.sigmoid(self._att_project_to_C(L_att, self.illum_att_proj))  # (B,C,H,W)
            # L_mask = self._att_project_to_C(L_att, self.illum_att_proj)
            fL = fL * L_mask
        fL = self.illum_mid_seq(fL)

        if self.use_chroma_attn:
            C_att = self.chroma_att(C_stack)                      # (B,K_C,H,W)
            # save_tensor_channels(C_att, 'debug_output', prefix=f"C_att_")
            C_mask = torch.sigmoid(self._att_project_to_C(C_att, self.chroma_att_proj))  # (B,C,H,W)
            fC = fC * C_mask
        fC = self.chroma_mid_seq(fC)

        # Heads
        L_hat = self.head_luma(fL)                                # (B,1,H,W)
        C_hat = self.head_chroma(fC)                              # (B,out_ch,H,W)

        # # Optional head activations (range controls)
        L_hat = self._apply_head_act(L_hat, self.luma_head_act)
        C_hat = self._apply_head_act(C_hat, self.chroma_head_act)

        delta = L_hat * C_hat
        delta = F.interpolate(input=delta, scale_factor=factor)

        x_det = torch.clamp(rgb_in + 0.1 * delta, 1e-6, 1.0)

        return x_det, delta

    def param_count(self) -> int:
        return count_params(self)
    


import os
from torchvision.transforms.functional import to_pil_image

def save_tensor_channels(tensor, save_dir, prefix):
    """
    Save a (B, C, H, W) tensor as per-channel PNGs for the first batch element.
    """
    if tensor is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    # Take first item in batch
    t = tensor[0]  # (C, H, W)
    C = t.size(0)

    for c in range(C):
        img = t[c]  # (H, W)
        # Normalize to [0,1] for visualization
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        to_pil_image(img_norm).save(
            os.path.join(save_dir, f"{prefix}_ch{c}.png")
        )
