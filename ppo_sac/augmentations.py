"""
This module provides image augmentation methods for reinforcement learning.
It includes CUDA-accelerated implementations of common image transformations used in RL training pipelines.

All methods expect a float tensor between 0 and 1 (normalized images).
"""
import math
import torch
import torch.nn.functional as F


def crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (
        w1 is not None and h1 is not None
    ), "must either specify both w1 and h1 or neither of them"

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        raise ValueError("Crop size is larger than the image size")

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(
        x.shape
    ), "window_shape must be a tuple with same number of dimensions as x"

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3),
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    return crop(imgs, size=h)


def channel_shuffle(x):
    """Shuffle the channels of an image tensor"""
    return x[:, torch.randperm(x.size(1)), :, :]


def blur(x, max_kernel_size=3):
    """Apply a random blur to an image tensor"""
    # Randomly choose kernel size between 1 and kernel_size
    k = torch.randint(1, max_kernel_size, (1,)).item()
    if k % 2 == 0:  # Ensure odd kernel size for symmetric padding
        k += 1
    result = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return result


def brightness(x, max_scale=0.2):
    """Apply a random brightness to an image tensor"""
    scaled = x * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * (max_scale * 2) + (1-max_scale))
    return torch.clamp(scaled, 0, 1)


def gaussian_noise(x, max_scale=0.1):
    """Apply gaussian noise to an image tensor"""
    x += torch.randn_like(x) * max_scale
    return torch.clamp(x, 0, 1)


def rotation(x, max_degrees=10):
    """Apply random rotation to an image tensor"""
    # Generate random angles between -max_degrees and max_degrees for each image in batch
    angles = torch.rand(x.size(0), device=x.device) * 2 * max_degrees - max_degrees
    
    # Convert angles to radians
    angles = angles * math.pi / 180
    
    # Calculate rotation matrix components
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    zeros = torch.zeros_like(angles)
    ones = torch.ones_like(angles)
    
    # Create rotation matrices for each image in batch
    rotation_matrices = torch.stack([
        torch.stack([cos_angles, -sin_angles, zeros], dim=1),
        torch.stack([sin_angles, cos_angles, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=2)
    
    # Create affine grid and apply rotation
    grid = F.affine_grid(rotation_matrices[:,:2], x.size(), align_corners=False)
    rotated = F.grid_sample(x, grid, align_corners=False)
    
    return rotated


def color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
    """Apply random color jittering to an image tensor"""
    # Apply random brightness adjustment
    if brightness > 0:
        brightness_factor = 1.0 + torch.rand(1, device=x.device) * brightness * 2 - brightness
        x = x * brightness_factor

    # Apply random contrast adjustment
    if contrast > 0:
        contrast_factor = 1.0 + torch.rand(1, device=x.device) * contrast * 2 - contrast
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast_factor + mean

    # Apply random saturation adjustment
    if saturation > 0:
        saturation_factor = 1.0 + torch.rand(1, device=x.device) * saturation * 2 - saturation
        grayscale = x.mean(dim=1, keepdim=True)
        x = x * saturation_factor + grayscale * (1 - saturation_factor)

    # Apply random hue adjustment
    if hue > 0:
        hue_factor = torch.rand(1, device=x.device) * hue * 2 - hue
        x = torch.clamp(x + hue_factor, 0, 1)

    return x


def grayscale(x, p=0.2):
    """Randomly convert image to grayscale with probability p"""
    mask = torch.rand(x.size(0), device=x.device) < p
    gray = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    result = torch.where(mask.view(-1, 1, 1, 1), gray, x)
    return result


def cutout(x, size=20):
    """Apply random cutout to an image tensor"""
    b, c, h, w = x.shape
    mask = torch.ones_like(x, device=x.device)
    y = torch.randint(0, h - size, (b,))
    x_pos = torch.randint(0, w - size, (b,))

    for i in range(b):
        mask[i, :, y[i] : y[i] + size, x_pos[i] : x_pos[i] + size] = 0
    return x * mask


def convolution(x):
    """Apply random convolution filter to an image tensor"""
    kernel = torch.randn(3, 3, device=x.device)
    kernel = kernel / kernel.sum()  # Normalize
    kernel = kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    return torch.nn.functional.conv2d(x, kernel, padding=1, groups=x.size(1))


def inversion(x, p=0.2):
    """Randomly invert colors with probability p"""
    mask = (torch.rand(x.size(0), device=x.device) < p).view(-1, 1, 1, 1)
    return torch.where(mask, 1 - x, x)
