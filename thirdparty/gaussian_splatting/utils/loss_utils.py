#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import img_as_float, filters, color
from skimage.metrics import structural_similarity
from skimage.feature import canny
from typing import Tuple
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def rel_loss(network_output, gt):
    return torch.abs((network_output - gt) / (gt + 0.001)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssimmap(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssimmap(img1, img2, window, window_size, channel, size_average)


def _ssimmap(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def ssim_score(img1: np.ndarray,
               img2: np.ndarray,
               win_size: int = 11,
               gaussian_weights: bool = True,
               sigma: float = 1.5) -> float:
    """
    Compute the scalar SSIM index between two images.

    Args:
        img1, img2: Arrays of shape (H, W) or (H, W, C). dtype uint8 or float.
        win_size: Sliding window size.
        gaussian_weights: Whether to weight the window with a Gaussian.
        sigma: Sigma for the Gaussian window.

    Returns:
        SSIM index (float in [-1, 1], higher is more similar).
    """
    f1 = img_as_float(img1)
    f2 = img_as_float(img2)
    multichannel = (f1.ndim == 3)

    score = structural_similarity(
        f1, f2,
        win_size=win_size,
        gaussian_weights=gaussian_weights,
        sigma=sigma,
        data_range=1.0,
        multichannel=multichannel
    )
    return score

def ssim_map(img1: np.ndarray,
             img2: np.ndarray,
             win_size: int = 11,
             gaussian_weights: bool = True,
             sigma: float = 1.5) -> np.ndarray:
    """
    Compute the full SSIM map between two images.

    Args:
        img1, img2: Arrays of shape (H, W) or (H, W, C). dtype uint8 or float.
        win_size: Sliding window size.
        gaussian_weights: Whether to weight the window with a Gaussian.
        sigma: Sigma for the Gaussian window.

    Returns:
        SSIM map array of same shape as input (or single-channel).
    """
    f1 = img_as_float(img1)
    f2 = img_as_float(img2)
    multichannel = (f1.ndim == 3)

    _, s_map = structural_similarity(
        f1, f2,
        win_size=win_size,
        gaussian_weights=gaussian_weights,
        sigma=sigma,
        data_range=1.0,
        multichannel=multichannel,
        full=True
    )
    return s_map

def compute_edge_density(img: np.ndarray,
                         method: str = 'sobel',
                         sigma: float = 1.0) -> float:
    """
    Compute a global edge density of an image in [0, 1].

    Args:
        img: H×W or H×W×C array.
        method: 'sobel' or 'canny'.
        sigma: smoothing for Sobel, ignored by Canny.

    Returns:
        Fraction of pixels marked as edge.
    """
    gray = color.rgb2gray(img) if img.ndim == 3 else img
    if method == 'canny':
        edges = canny(gray)
    else:
        # Sobel gradient magnitude then threshold at mean
        grad = filters.sobel(gray, sigma=sigma)
        edges = grad > grad.mean()
    return float(edges.mean())

def adaptive_window_size(base_size: int,
                         edge_density: float,
                         min_size: int = 3) -> int:
    """
    Shrink the window size where edges are dense.

    window_size = max(min_size, int(base_size * (1 - edge_density)))

    Ensures window_size is odd.
    """
    size = max(min_size, int(base_size * (1.0 - edge_density)))
    return size + 1 if size % 2 == 0 else size

def ssim_score_edge_aware(img1: np.ndarray,
                          img2: np.ndarray,
                          base_win_size: int = 11,
                          gaussian_weights: bool = True,
                          sigma: float = 1.5,
                          edge_method: str = 'sobel') -> float:
    """
    Compute an edge-aware SSIM index:
      • Detect edges on img1 → edge_density in [0,1]
      • Adapt window size: smaller windows if many edges
      • Call structural_similarity with that window
    """
    # to [0,1] floats
    f1 = img_as_float(img1)
    f2 = img_as_float(img2)
    # 1) edge density
    ed = compute_edge_density(f1, method=edge_method, sigma=sigma)
    # 2) adaptive window size
    win_size = adaptive_window_size(base_win_size, ed)
    # 3) SSIM
    multichannel = (f1.ndim == 3)
    score = structural_similarity(
        f1, f2,
        win_size=win_size,
        gaussian_weights=gaussian_weights,
        sigma=sigma,
        data_range=1.0,
        multichannel=multichannel
    )
    return score

def ssim_map_edge_aware(img1: np.ndarray,
                        img2: np.ndarray,
                        base_win_size: int = 11,
                        gaussian_weights: bool = True,
                        sigma: float = 1.5,
                        edge_method: str = 'sobel') -> np.ndarray:
    """
    Same as above, but returns the per-pixel SSIM map.
    """
    f1 = img_as_float(img1)
    f2 = img_as_float(img2)
    ed = compute_edge_density(f1, method=edge_method, sigma=sigma)
    win_size = adaptive_window_size(base_win_size, ed)
    multichannel = (f1.ndim == 3)
    _, s_map = structural_similarity(
        f1, f2,
        win_size=win_size,
        gaussian_weights=gaussian_weights,
        sigma=sigma,
        data_range=1.0,
        multichannel=multichannel,
        full=True
    )
    return s_map
