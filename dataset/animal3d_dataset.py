"""
Animal3D Dataset loader from HuggingFace.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from PIL import Image
import io

# Import HuggingFace datasets with explicit name to avoid conflict
try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    # Fallback if datasets is not installed
    hf_load_dataset = None

from .utils import (
    load_image, load_mask, resize_image, resize_mask,
    normalize_image, normalize_mask, process_bbox,
    normalize_keypoints_2d, image_to_tensor
)


class Animal3DDataset(Dataset):
    """
    Animal3D Dataset loaded from HuggingFace.
    
    This dataset loads images, masks, keypoints, and annotations from the
    Animal3D dataset stored on HuggingFace.
    """
    
    def __init__(
        self,
        split: str = "train",
        token: Optional[str] = None,
        image_size: int = 256,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Animal3D dataset.
        
        Args:
            split: Dataset split ('train', 'test', 'val', etc.)
            token: HuggingFace token for private repository access
            image_size: Target image size for resizing
            mean: Mean values for normalization (default: ImageNet)
            std: Std values for normalization (default: ImageNet)
            cache_dir: Directory to cache the dataset
        """
        self.split = split
        self.image_size = image_size
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        
        # Load dataset from HuggingFace
        repo_id = "WatermelonAnh/WattiesMammals"
        data_path = f"data/animal3d/{split}.json"
        # data_path = f"{split}.json"
        
        # Store repo_id and token for loading images later
        self.repo_id = repo_id
        self.token = token
        print(f"Repo ID: {repo_id}")
        print(f"Token: {'***' if token else None}")  # Mask token for security
        print(f"Data path: {data_path}")
        print(f"Cache dir: {cache_dir}")
        
        try:
            # Load JSON file from HuggingFace
            # The JSON file contains the full structure with 'data' key
            if hf_load_dataset is None:
                raise ImportError("HuggingFace datasets library is not installed. "
                                "Install it with: pip install datasets")
            
            if token:
                dataset_dict = hf_load_dataset(
                    repo_id,
                    data_files={split: data_path},
                    # repo_id=repo_id,
                    token=token,
                    cache_dir=cache_dir,
                )
            else:
                # Try without token first (will fail if private)
                dataset_dict = hf_load_dataset(
                    repo_id,
                    data_files={split: data_path},
                    # repo_id=repo_id,
                    cache_dir=cache_dir,
                )
            
            # Get the split dataset
            self.dataset = dataset_dict[split] if split in dataset_dict else dataset_dict.get('train', None)
            
            if self.dataset is None:
                raise ValueError(f"Split '{split}' not found in dataset")
            
        except Exception as e:
            print(f"Error: {e}")
            raise RuntimeError(
                f"Failed to load dataset from HuggingFace. "
                f"Make sure you have access to the repository and provide a token if it's private. "
                f"Error: {str(e)}"
            )
        
        # The JSON structure: when loaded, each row might be a dict or the whole file
        # Check if the dataset has the expected structure
        if len(self.dataset) > 0:
            first_item = self.dataset[0]
            
            # Check if first item is the root dict with 'data' key
            if isinstance(first_item, dict) and 'data' in first_item:
                # Single row containing the full JSON structure
                self.data = first_item
                self.samples = self.data.get('data', [])
                self.metadata = {
                    'flength': self.data.get('flength', 1000.0),
                    'supercategories': self.data.get('supercategories', []),
                    'categories': self.data.get('categories', []),
                    'keypoint_vids': self.data.get('keypoint_vids', []),
                    'info': self.data.get('info', '')
                }
            else:
                # Each row might be a sample, or we need to aggregate
                # Try to find metadata in any row
                self.samples = []
                self.metadata = {}
                
                # Collect all samples
                for item in self.dataset:
                    if isinstance(item, dict):
                        # Check if this is a metadata row or a sample row
                        if 'data' in item:
                            # This is the root structure
                            self.data = item
                            self.samples = item.get('data', [])
                            self.metadata = {
                                'flength': item.get('flength', 1000.0),
                                'supercategories': item.get('supercategories', []),
                                'categories': item.get('categories', []),
                                'keypoint_vids': item.get('keypoint_vids', []),
                                'info': item.get('info', '')
                            }
                            break
                        elif 'img_path' in item:
                            # This is a sample
                            self.samples.append(item)
                
                # If we didn't find the root structure, set default metadata
                if not self.metadata:
                    self.metadata = {
                        'flength': 1000.0,
                        'supercategories': [],
                        'categories': [],
                        'keypoint_vids': [],
                        'info': ''
                    }
        else:
            self.samples = []
            self.metadata = {
                'flength': 1000.0,
                'supercategories': [],
                'categories': [],
                'keypoint_vids': [],
                'info': ''
            }
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        if self.metadata.get('info'):
            print(f"Dataset info: {self.metadata['info']}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - img: Image tensor (C, H, W)
                - mask: Mask tensor (H, W)
                - keypoints_2d: 2D keypoints (N, 3) normalized
                - keypoints_3d: 3D keypoints (N, 3)
                - bbox: Bounding box [x, y, w, h]
                - category: Category index
                - supercategory: Supercategory index
        """
        sample = self.samples[idx]
        
        # Load image
        img_path = f"data/animal3d/{sample.get('img_path', '')}"
        img = None
        
        if img_path:
            try:
                # Try to load image from HuggingFace using huggingface_hub
                from huggingface_hub import hf_hub_download
                
                # Download image from repo
                local_img_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=img_path,
                    token=self.token,
                    repo_type="dataset"
                )
                img = load_image(local_img_path)
            except Exception as e1:
                try:
                    # Fallback: try loading as local path if already downloaded
                    img = load_image(img_path)
                except Exception as e2:
                    print(f"Warning: Could not load image from {img_path}: {e1}, {e2}")
                    img = None
        
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Load mask
        mask_path = f"data/animal3d/{sample.get('mask_path', '')}"
        mask = None
        
        if mask_path:
            try:
                # Try to load mask from HuggingFace
                from huggingface_hub import hf_hub_download
                
                local_mask_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=mask_path,
                    token=self.token,
                    repo_type="dataset"
                )
                mask = load_mask(local_mask_path)
            except Exception as e1:
                try:
                    # Fallback: try loading as local path
                    mask = load_mask(mask_path)
                except Exception as e2:
                    print(f"Warning: Could not load mask from {mask_path}: {e1}, {e2}")
                    mask = None
        
        if mask is None:
            mask = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
        
        # Resize image and mask
        original_size = (img.shape[0], img.shape[1])
        img = resize_image(img, (self.image_size, self.image_size))
        mask = resize_mask(mask, (self.image_size, self.image_size))
        
        # Normalize image
        img_normalized = normalize_image(img, self.mean, self.std)
        
        # Normalize mask
        mask_normalized = normalize_mask(mask)
        
        # Convert to tensors
        img_tensor = image_to_tensor(img_normalized)
        mask_tensor = torch.from_numpy(mask_normalized).float()
        
        # Process keypoints
        keypoint_2d = np.array(sample.get('keypoint_2d', []), dtype=np.float32)
        if len(keypoint_2d) == 0:
            keypoint_2d = np.zeros((26, 3), dtype=np.float32)
        elif keypoint_2d.shape[1] == 2:
            # Add visibility if missing
            visibility = np.ones((keypoint_2d.shape[0], 1), dtype=np.float32)
            keypoint_2d = np.concatenate([keypoint_2d, visibility], axis=1)
        
        # Normalize keypoints based on original image size
        keypoints_2d_normalized = normalize_keypoints_2d(keypoint_2d, original_size)
        
        # Process 3D keypoints - ensure (N, 4) format with confidence
        keypoint_3d = np.array(sample.get('keypoint_3d', []), dtype=np.float32)
        if len(keypoint_3d) == 0:
            keypoint_3d = np.zeros((26, 4), dtype=np.float32)
        elif keypoint_3d.shape[1] == 3:
            # Add confidence dimension if missing
            confidence = np.ones((keypoint_3d.shape[0], 1), dtype=np.float32)
            keypoint_3d = np.concatenate([keypoint_3d, confidence], axis=1)
        
        # Process pose (105 floats = 35 joints * 3 for axis-angle or similar)
        pose = np.array(sample.get('pose', []), dtype=np.float32)
        if len(pose) == 0:
            # Default: 35 joints * 3 = 105 (or adjust based on actual format)
            pose = np.zeros(105, dtype=np.float32)
        
        # Process shape (use shape_extra if available, else shape)
        # shape_extra has 21 floats, shape has 20 floats
        shape = np.array(sample.get('shape_extra', sample.get('shape', [])), dtype=np.float32)
        if len(shape) == 0:
            # Default to 41 for SMAL (20 base + 21 extra)
            shape = np.zeros(41, dtype=np.float32)
        elif len(shape) == 20:
            # If only base shape, pad to 41 with zeros
            shape_padded = np.zeros(41, dtype=np.float32)
            shape_padded[:20] = shape
            shape = shape_padded
        
        # Get focal length from metadata (per-sample or global)
        focal_length = self.metadata.get('flength', 1000.0)
        
        # Process camera parameters
        # Camera format: [scale, tx, ty] typically
        # Try to get from sample, otherwise use default
        trans = np.array(sample.get('trans', [0.0, 0.0, 0.0]), dtype=np.float32)
        if len(trans) == 3:
            # Construct camera from translation (scale=1.0, tx=trans[0], ty=trans[1])
            camera = np.array([1.0, trans[0], trans[1]], dtype=np.float32)
        else:
            camera = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Bounding box
        bbox = np.array(sample.get('bbox', [0, 0, 0, 0]), dtype=np.float32)
        
        # Categories
        category = int(sample.get('category', 0))
        supercategory = int(sample.get('supercategory', 0))
        
        return {
            'img': img_tensor,
            'mask': mask_tensor,
            'keypoints_2d': torch.from_numpy(keypoints_2d_normalized).float(),
            'keypoints_3d': torch.from_numpy(keypoint_3d).float(),  # (N, 4) with confidence
            'bbox': torch.from_numpy(bbox).float(),
            'category': torch.tensor(category, dtype=torch.long),
            'supercategory': torch.tensor(supercategory, dtype=torch.long),
            'img_path': img_path,
            'original_size': torch.tensor(original_size, dtype=torch.int32),
            # Add required fields for loss computation
            'pose': torch.from_numpy(pose).float(),
            'shape': torch.from_numpy(shape).float(),
            'focal_length': torch.tensor([focal_length, focal_length], dtype=torch.float32),
            'camera': torch.from_numpy(camera).float(),
            'smal_params': {
                'pose': torch.from_numpy(pose).float(),
                'betas': torch.from_numpy(shape).float(),
            },
            'has_smal_params': {
                'pose': torch.ones(1, dtype=torch.bool),
                'betas': torch.ones(1, dtype=torch.bool),
            }
        }

