"""
Simplified utility functions for image processing and data loading.
"""
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format (H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load a mask from file path.
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        Mask as numpy array (H, W) with values 0-255
    """
    mask = Image.open(mask_path).convert("L")
    return np.array(mask)


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        img: Image array (H, W, C)
        target_size: Target size (height, width)
        
    Returns:
        Resized image
    """
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    return np.array(img_resized)


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a mask to target size.
    
    Args:
        mask: Mask array (H, W)
        target_size: Target size (height, width)
        
    Returns:
        Resized mask
    """
    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
    return np.array(mask_resized)


def normalize_image(img: np.ndarray, mean: Optional[np.ndarray] = None, 
                    std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize image with mean and std.
    
    Args:
        img: Image array (H, W, C) in range [0, 255]
        mean: Mean values for each channel (C,)
        std: Std values for each channel (C,)
        
    Returns:
        Normalized image in range [0, 1] or normalized by mean/std
    """
    img = img.astype(np.float32)
    
    if mean is not None and std is not None:
        # Normalize with ImageNet stats or custom stats
        mean = np.array(mean) * 255.0
        std = np.array(std) * 255.0
        img = (img - mean) / std
    else:
        # Simple normalization to [0, 1]
        img = img / 255.0
    
    return img


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize mask to [0, 1] range.
    
    Args:
        mask: Mask array (H, W) in range [0, 255]
        
    Returns:
        Normalized mask in range [0, 1]
    """
    return (mask / 255.0).clip(0, 1)


def process_bbox(bbox: list) -> Tuple[np.ndarray, float]:
    """
    Process bounding box to get center and size.
    
    Args:
        bbox: Bounding box as [x, y, w, h]
        
    Returns:
        center: Center coordinates (cx, cy)
        size: Maximum of width and height
    """
    bbox = np.array(bbox, dtype=np.float32)
    center = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
    size = max(bbox[2], bbox[3])
    return center, size


def normalize_keypoints_2d(keypoints_2d: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Normalize 2D keypoints to [-0.5, 0.5] range based on image size.
    
    Args:
        keypoints_2d: Keypoints array (N, 3) where last dim is [x, y, visibility]
        img_size: Image size (height, width)
        
    Returns:
        Normalized keypoints (N, 3)
    """
    keypoints = keypoints_2d.copy().astype(np.float32)
    # Normalize x and y coordinates
    keypoints[:, 0] = keypoints[:, 0] / img_size[1] - 0.5  # x / width - 0.5
    keypoints[:, 1] = keypoints[:, 1] / img_size[0] - 0.5  # y / height - 0.5
    return keypoints


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.
    
    Args:
        array: Numpy array
        
    Returns:
        Torch tensor
    """
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(array)


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert image from HWC to CHW format and to tensor.
    
    Args:
        img: Image array (H, W, C)
        
    Returns:
        Tensor (C, H, W)
    """
    if len(img.shape) == 2:
        # Grayscale image
        img = img[..., None]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return to_tensor(img)

