"""
Inference script for mesh and silhouette generation.
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf

from models.model import CategoryRoutedModel
from dataset import Animal3DDataset
from dataset.utils import (
    load_image, resize_image, normalize_image, image_to_tensor
)
from models.utils.geometry import perspective_projection


def setup_device(cfg):
    """Setup device."""
    if cfg.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_model(cfg, device):
    """Create model from config."""
    # Build config dict for model
    model_cfg = {
        'image_size': cfg.model.image_size,
        'num_categories': cfg.model.num_categories,
        'backbone': OmegaConf.to_container(cfg.model.backbone, resolve=True),
        'cls_header': OmegaConf.to_container(cfg.model.get('cls_header', {}), resolve=True),
        'classifier': OmegaConf.to_container(cfg.model.classifier, resolve=True),
        'regressors': OmegaConf.to_container(cfg.model.regressors, resolve=True),
        'router': OmegaConf.to_container(cfg.model.get('router', {}), resolve=True),
        'smal': OmegaConf.to_container(cfg.model.get('smal', {}), resolve=True),
        'huggingface': OmegaConf.to_container(cfg.get('huggingface', {}), resolve=True),
    }
    
    model = CategoryRoutedModel(model_cfg)
    model = model.to(device)
    
    return model


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """Load checkpoint and restore model state."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model state from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def load_image_from_path(image_path: str, image_size: int = 256, 
                         mean: list = [0.485, 0.456, 0.406],
                         std: list = [0.229, 0.224, 0.225]) -> Dict[str, Any]:
    """
    Load and preprocess image from file path.
    
    Args:
        image_path: Path to image file
        image_size: Target image size
        mean: Mean values for normalization
        std: Std values for normalization
    
    Returns:
        Dictionary with preprocessed image tensor and original image
    """
    # Load image
    img = load_image(image_path)
    original_img = img.copy()
    
    # Resize
    img = resize_image(img, (image_size, image_size))
    
    # Normalize
    img = normalize_image(img, mean=np.array(mean), std=np.array(std))
    
    # Convert to tensor (CHW format)
    img_tensor = image_to_tensor(img)
    
    return {
        'img': img_tensor,
        'original_img': original_img,
        'img_path': image_path
    }


def load_image_from_dataset(dataset: Animal3DDataset, index: int, 
                            image_size: int = 256) -> Dict[str, Any]:
    """
    Load image from dataset by index.
    
    Args:
        dataset: Animal3D dataset instance
        index: Sample index
        image_size: Target image size (should match dataset)
    
    Returns:
        Dictionary with preprocessed image tensor and original image
    """
    if index >= len(dataset):
        raise ValueError(f"Index {index} out of range (dataset size: {len(dataset)})")
    
    sample = dataset[index]
    
    # Get original image path if available
    img_path = sample.get('img_path', f'dataset_sample_{index}')
    
    # Denormalize image for visualization
    img_tensor = sample['img']  # Already preprocessed (C, H, W)
    img_np = img_tensor.numpy()
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_vis = img_np.copy()
    for i in range(3):
        img_vis[i] = img_vis[i] * std[i] + mean[i]
    img_vis = np.clip(img_vis, 0, 1)
    img_vis = np.transpose(img_vis, (1, 2, 0))  # CHW -> HWC
    img_vis = (img_vis * 255).astype(np.uint8)
    
    return {
        'img': img_tensor,
        'original_img': img_vis,
        'img_path': img_path
    }


def save_mesh_obj(vertices: np.ndarray, faces: np.ndarray, output_path: str):
    """
    Save mesh to OBJ format.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices (0-indexed)
        output_path: Path to save OBJ file
    """
    with open(output_path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Saved mesh to {output_path}")


def project_vertices_to_2d(vertices: torch.Tensor, cam_t: torch.Tensor, 
                           focal_length: torch.Tensor, image_size: int) -> np.ndarray:
    """
    Project 3D vertices to 2D image coordinates.
    
    Args:
        vertices: (N, 3) 3D vertices
        cam_t: (3,) camera translation
        focal_length: (2,) focal length
        image_size: Image size for normalization
    
    Returns:
        (N, 2) 2D projected points in pixel coordinates
    """
    # Reshape for batch processing
    vertices = vertices.unsqueeze(0)  # (1, N, 3)
    cam_t = cam_t.unsqueeze(0)  # (1, 3)
    focal_length = focal_length.unsqueeze(0)  # (1, 2)
    
    # Project using perspective projection
    # Note: perspective_projection expects focal_length normalized by image_size
    projected = perspective_projection(
        vertices,
        translation=cam_t,
        focal_length=focal_length / image_size
    )
    
    # Convert from normalized [-0.5, 0.5] to pixel coordinates [0, image_size]
    projected_pixel = (projected[0] + 0.5) * image_size
    
    return projected_pixel.cpu().numpy()


def render_silhouette(vertices_2d: np.ndarray, faces: np.ndarray, 
                     image_shape: tuple) -> np.ndarray:
    """
    Render silhouette mask from projected vertices and faces.
    
    Args:
        vertices_2d: (N, 2) 2D projected vertices
        faces: (M, 3) face indices (0-indexed)
        image_shape: (H, W) output image shape
    
    Returns:
        (H, W) binary silhouette mask
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Clip vertices to image bounds
    h, w = image_shape
    vertices_2d = np.clip(vertices_2d, [0, 0], [w-1, h-1])
    
    # Render each face
    for face in faces:
        pts = vertices_2d[face].astype(np.int32)
        # Check if all points are valid (within bounds)
        if np.all(pts >= 0) and np.all(pts[:, 0] < w) and np.all(pts[:, 1] < h):
            cv2.fillPoly(mask, [pts], 255)
    
    return mask


def create_silhouette_overlay(original_img: np.ndarray, silhouette_mask: np.ndarray,
                              overlay_color: tuple = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay image with silhouette on original image.
    
    Args:
        original_img: (H, W, 3) original image
        silhouette_mask: (H, W) binary mask
        overlay_color: RGB color for overlay
        alpha: Transparency of overlay
    
    Returns:
        (H, W, 3) overlay image
    """
    # Resize mask if needed
    if silhouette_mask.shape[:2] != original_img.shape[:2]:
        silhouette_mask = cv2.resize(silhouette_mask, 
                                   (original_img.shape[1], original_img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = np.zeros_like(original_img)
    mask_bool = silhouette_mask > 0
    overlay[mask_bool] = overlay_color
    
    # Blend with original image
    result = (1 - alpha) * original_img.astype(np.float32) + alpha * overlay.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def run_inference(model: torch.nn.Module, img_tensor: torch.Tensor, 
                 device: torch.device, image_size: int = 256) -> Dict[str, Any]:
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        img_tensor: (C, H, W) preprocessed image tensor
        device: Device to run inference on
        image_size: Image size
    
    Returns:
        Model output dictionary
    """
    model.eval()
    
    # Prepare batch
    img_batch = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    # Create batch dict
    batch = {
        'img': img_batch
    }
    
    # Run inference
    with torch.no_grad():
        output = model(batch, training=False)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Inference script for mesh and silhouette generation")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory containing checkpoint files")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file (default: configs/config.yaml)")
    parser.add_argument("--output-dir", type=str, default="./inference_outputs",
                       help="Directory to save outputs (default: ./inference_outputs)")
    
    # Image input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-path", type=str,
                            help="Path to input image file")
    input_group.add_argument("--dataset-index", type=int,
                            help="Index of sample from validation/test dataset")
    
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "val"],
                       help="Dataset split to use when using --dataset-index (default: test)")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt",
                       choices=["best_model.pt", "latest_checkpoint.pt"],
                       help="Checkpoint file to load (default: best_model.pt)")
    parser.add_argument("--overlay-color", type=int, nargs=3, default=[0, 255, 0],
                       help="RGB color for silhouette overlay (default: 0 255 0)")
    parser.add_argument("--overlay-alpha", type=float, default=0.5,
                       help="Alpha transparency for overlay (default: 0.5)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    print("Loading config...")
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        # Try with Hydra config path (relative to configs directory)
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        try:
            with hydra.initialize(config_path="configs", version_base=None):
                cfg = hydra.compose(config_name=config_name)
        except Exception as e:
            print(f"Warning: Could not load config with Hydra: {e}")
            print("Trying to load default config...")
            with hydra.initialize(config_path="configs", version_base=None):
                cfg = hydra.compose(config_name="config")
    
    # Setup device
    device = setup_device(cfg)
    
    # Create model
    print("Creating model...")
    model = create_model(cfg, device)
    
    # Check if SMAL is loaded
    if not model.has_smal():
        raise RuntimeError(
            "ERROR: SMAL model is not loaded. Cannot generate mesh. "
            "Please check model_path in configs/model/smal.yaml"
        )
    print("SMAL model loaded successfully.")
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    load_checkpoint(checkpoint_path, model, device)
    
    # Load image
    print("Loading image...")
    if args.image_path:
        # Load from file path
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        
        data = load_image_from_path(
            args.image_path,
            image_size=cfg.model.image_size,
            mean=cfg.data.dataset.get('mean', [0.485, 0.456, 0.406]),
            std=cfg.data.dataset.get('std', [0.229, 0.224, 0.225])
        )
        identifier = os.path.splitext(os.path.basename(args.image_path))[0]
    else:
        # Load from dataset
        print(f"Loading sample {args.dataset_index} from {args.split} split...")
        dataset = Animal3DDataset(
            split=args.split,
            token=cfg.get('huggingface', {}).get('token', None),
            image_size=cfg.model.image_size,
            cache_dir=cfg.get('huggingface', {}).get('cache_dir', None)
        )
        data = load_image_from_dataset(dataset, args.dataset_index, cfg.model.image_size)
        identifier = f"{args.split}_sample_{args.dataset_index}"
    
    img_tensor = data['img']
    original_img = data['original_img']
    img_path = data['img_path']
    
    print(f"Image loaded: {img_path}")
    print(f"Original image shape: {original_img.shape}")
    
    # Run inference
    print("Running inference...")
    output = run_inference(model, img_tensor, device, cfg.model.image_size)
    
    # Extract results
    pred_vertices = output['pred_vertices'][0].cpu().numpy()  # (N, 3)
    pred_cam_t = output['pred_cam_t'][0].cpu()  # (3,)
    focal_length = output['focal_length'][0].cpu()  # (2,)
    faces = model.smal.faces.cpu().numpy()  # (M, 3)
    
    print(f"Generated mesh with {len(pred_vertices)} vertices and {len(faces)} faces")
    
    # Save mesh
    mesh_path = os.path.join(args.output_dir, f"mesh_{identifier}.obj")
    save_mesh_obj(pred_vertices, faces, mesh_path)
    
    # Project vertices to 2D
    print("Projecting mesh to 2D...")
    vertices_2d = project_vertices_to_2d(
        torch.from_numpy(pred_vertices),
        pred_cam_t,
        focal_length,
        cfg.model.image_size
    )
    
    # Render silhouette
    print("Rendering silhouette...")
    # Resize original image to match model input size if needed
    if original_img.shape[:2] != (cfg.model.image_size, cfg.model.image_size):
        original_img_resized = resize_image(original_img, 
                                           (cfg.model.image_size, cfg.model.image_size))
    else:
        original_img_resized = original_img.copy()
    
    silhouette_mask = render_silhouette(
        vertices_2d,
        faces,
        (cfg.model.image_size, cfg.model.image_size)
    )
    
    # Save silhouette mask
    mask_path = os.path.join(args.output_dir, f"silhouette_mask_{identifier}.png")
    cv2.imwrite(mask_path, silhouette_mask)
    print(f"Saved silhouette mask to {mask_path}")
    
    # Create and save overlay
    overlay = create_silhouette_overlay(
        original_img_resized,
        silhouette_mask,
        overlay_color=tuple(args.overlay_color),
        alpha=args.overlay_alpha
    )
    overlay_path = os.path.join(args.output_dir, f"silhouette_overlay_{identifier}.png")
    # Convert RGB to BGR for OpenCV
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)
    print(f"Saved silhouette overlay to {overlay_path}")
    
    # Also save original image for reference
    original_path = os.path.join(args.output_dir, f"original_{identifier}.png")
    if len(original_img_resized.shape) == 3 and original_img_resized.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        original_bgr = cv2.cvtColor(original_img_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, original_bgr)
    else:
        cv2.imwrite(original_path, original_img_resized)
    print(f"Saved original image to {original_path}")
    
    print("\n" + "="*60)
    print("Inference completed successfully!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"  - Mesh: mesh_{identifier}.obj")
    print(f"  - Silhouette mask: silhouette_mask_{identifier}.png")
    print(f"  - Silhouette overlay: silhouette_overlay_{identifier}.png")
    print(f"  - Original image: original_{identifier}.png")


if __name__ == "__main__":
    main()

