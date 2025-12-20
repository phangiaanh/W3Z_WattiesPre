import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import einops
import pickle
import os

from .backbones.vit import vith
from .heads.cls_header import ClassTokenHead
from .heads.category_classifier import CategoryClassifier
from .heads.category_regressors import CategoryRegressors
from .category_router import CategoryRouter
from .utils.geometry import perspective_projection
from .smal_wrapper import SMAL

# Import HuggingFace data loader (handle both relative and absolute imports)
try:
    from utils.hf_data_loader import resolve_path
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.hf_data_loader import resolve_path


class CategoryRoutedModel(nn.Module):
    """Main model with category routing for pose/shape/camera regression."""
    
    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration dict with model parameters
        """
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.get('image_size', 256)
        self.num_categories = cfg.get('num_categories', 5)
        
        # Get HuggingFace config
        hf_cfg = cfg.get('huggingface', {})
        # Backbone and SMAL data are stored in WatermelonAnh/Watties_Data
        hf_data_repo_id = hf_cfg.get('data_repo_id', 'WatermelonAnh/Watties_Data')
        hf_token = hf_cfg.get('token', os.getenv('HUGGINGFACE_TOKEN', None))
        cache_dir = hf_cfg.get('cache_dir', None)
        
        # Create ViT backbone
        backbone_cfg = cfg.get('backbone', {})
        
        # Load pretrained backbone weights if specified
        pretrained_weights = backbone_cfg.get('pretrained_weights', None)
        pretrained_path = None
        if pretrained_weights:
            # Resolve path (local or HuggingFace)
            # Backbone weights are stored in WatermelonAnh/Watties_Data
            is_hf_path = not os.path.isabs(pretrained_weights) and not os.path.exists(pretrained_weights)
            pretrained_path = resolve_path(
                pretrained_weights,
                repo_id=hf_data_repo_id,
                token=hf_token,
                cache_dir=cache_dir,
                is_hf_path=is_hf_path
            )
            backbone_cfg['pretrained_weights'] = pretrained_path
        
        self.backbone = vith(cfg=backbone_cfg)
        
        # Load pretrained weights if path was provided
        if pretrained_path:
            print(f'Loading backbone weights from {pretrained_path}')
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'backbone.' prefix if present
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f'Missing keys when loading backbone: {len(missing_keys)} keys')
            if unexpected_keys:
                print(f'Unexpected keys when loading backbone: {len(unexpected_keys)} keys')
        
        # Create CLS header
        cls_header_cfg = cfg.get('cls_header', {})
        self.cls_header = ClassTokenHead(**cls_header_cfg)
        
        # Create category classifier
        classifier_cfg = cfg.get('classifier', {})
        classifier_cfg.setdefault('input_dim', cls_header_cfg.get('output_dim', 256))
        classifier_cfg.setdefault('num_categories', self.num_categories)
        self.category_classifier = CategoryClassifier(**classifier_cfg)
        
        # Create category regressors
        regressor_cfg = cfg.get('regressors', {})
        # Pass backbone embed_dim as context_dim for cross-attention
        self.regressors = CategoryRegressors(
            regressor_cfg, 
            num_categories=self.num_categories,
            context_dim=self.backbone.embed_dim
        )
        
        # Create category router
        router_cfg = cfg.get('router', {})
        self.router = CategoryRouter(**router_cfg)
        
        # Instantiate SMAL model
        smal_cfg = cfg.get('smal', {})
        smal_model_path = smal_cfg.get('model_path', None)
        if smal_model_path:
            # Resolve SMAL model path (local or HuggingFace)
            # SMAL data is stored in WatermelonAnh/Watties_Data
            is_hf_path = not os.path.isabs(smal_model_path) and not os.path.exists(smal_model_path)
            smal_path = resolve_path(
                smal_model_path,
                repo_id=hf_data_repo_id,
                token=hf_token,
                cache_dir=cache_dir,
                is_hf_path=is_hf_path
            )
            
            print(f'Loading SMAL model from {smal_path}')
            with open(smal_path, 'rb') as f:
                smal_data = pickle.load(f, encoding="latin1")
            self.smal = SMAL(**smal_data)
        else:
            self.smal = None
            print('Warning: SMAL model path not provided. Keypoints/vertices will not be generated.')
        
        # Get embed dimension from backbone
        self.embed_dim = self.backbone.embed_dim
    
    def forward(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing:
                - img: (B, 3, H, W) input images
                - supercategory: (B,) ground truth supercategory indices (optional, for training)
                - focal_length: (B, 2) focal lengths (optional)
            training: Whether in training mode
        
        Returns:
            Dictionary containing:
                - cls_token: (B, embed_dim) CLS token
                - cls_feats: (B, output_dim) CLS features
                - category_logits: (B, num_categories) category logits
                - category_probs: (B, num_categories) category probabilities
                - pred_category: (B,) predicted category indices
                - pred_pose: (B, npose) pose predictions
                - pred_shape: (B, shape_dim) shape predictions
                - pred_camera: (B, 3) camera predictions
        """
        # Extract input
        img = batch['img']
        batch_size = img.shape[0]
        device = img.device
        
        # Get ground truth supercategory if available
        gt_category = batch.get('supercategory', None)
        
        # Forward through backbone
        # Note: Original AniMer crops image, but we'll use full image for now
        # You can add cropping if needed: img[:, :, :, 32:-32]
        features, cls_token = self.backbone(img)
        
        # Extract CLS features
        cls_feats = self.cls_header(cls_token)
        
        # Classify category
        category_logits, category_probs = self.category_classifier(cls_feats)
        pred_category = torch.argmax(category_probs, dim=-1)
        
        # Get routing indices (use GT during training if configured)
        routing_indices = self.router.get_routing_indices(
            pred_category, gt_category, training=training
        )
        
        # Convert features to token format (B, H*W, C)
        features_tokens = einops.rearrange(features, 'b c h w -> b (h w) c')
        
        # Route and regress
        pred_pose, pred_shape, pred_camera = self.regressors(features_tokens, routing_indices)
        
        # Convert pose to rotation matrices
        pose_dict = self.regressors.convert_pose_to_rotmat(pred_pose)
        
        # Prepare output
        output = {
            'cls_token': cls_token,
            'cls_feats': cls_feats,
            'category_logits': category_logits,
            'category_probs': category_probs,
            'pred_category': pred_category,
            'pred_pose': pred_pose,
            'pred_shape': pred_shape,
            'pred_camera': pred_camera,
            'pred_smal_params': {
                'global_orient': pose_dict['global_orient'],
                'pose': pose_dict['pose'],
                'betas': pred_shape,
            }
        }
        
        # Compute camera translation if focal length is provided
        if 'focal_length' in batch:
            focal_length = batch['focal_length']
            pred_cam_t = torch.stack([
                pred_camera[:, 1],
                pred_camera[:, 2],
                2 * focal_length[:, 0] / (self.image_size * pred_camera[:, 0] + 1e-9)
            ], dim=-1)
            output['pred_cam_t'] = pred_cam_t
            output['focal_length'] = focal_length
        else:
            # Default focal length if not provided
            focal_length = torch.tensor([[self.image_size, self.image_size]], 
                                       device=device, dtype=torch.float32).expand(batch_size, -1)
            pred_cam_t = torch.stack([
                pred_camera[:, 1],
                pred_camera[:, 2],
                2 * focal_length[:, 0] / (self.image_size * pred_camera[:, 0] + 1e-9)
            ], dim=-1)
            output['pred_cam_t'] = pred_cam_t
            output['focal_length'] = focal_length
        
        # Forward through SMAL to get keypoints and vertices
        if self.smal is not None:
            # Reshape pose parameters for SMAL
            pred_smal_params_reshaped = {
                'global_orient': pose_dict['global_orient'].reshape(batch_size, -1, 3, 3),
                'pose': pose_dict['pose'].reshape(batch_size, -1, 3, 3),
                'betas': pred_shape.reshape(batch_size, -1),
            }
            
            # Forward through SMAL
            smal_output = self.smal(**pred_smal_params_reshaped, pose2rot=False)
            
            pred_keypoints_3d = smal_output.joints
            pred_vertices = smal_output.vertices
            output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
            output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
            
            # Project to 2D
            pred_cam_t = output['pred_cam_t'].reshape(-1, 3)
            focal_length = output['focal_length'].reshape(-1, 2)
            pred_keypoints_2d = perspective_projection(
                pred_keypoints_3d,
                translation=pred_cam_t,
                focal_length=focal_length / self.image_size
            )
            output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        
        return output
    

