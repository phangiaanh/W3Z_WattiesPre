import torch
import torch.nn as nn
import einops
from ..components.pose_transformer import TransformerDecoder
from ..utils.geometry import rot6d_to_rotmat, aa_to_rotmat


class SingleRegressorHead(nn.Module):
    """Single regressor head using transformer decoder architecture."""
    
    def __init__(self, cfg, output_dim, regressor_type='pose'):
        """
        Args:
            cfg: Configuration dict with regressor parameters
            output_dim: Output dimension (e.g., npose for pose, 41 for shape, 3 for camera)
            regressor_type: Type of regressor ('pose', 'shape', 'camera')
        """
        super().__init__()
        self.cfg = cfg
        self.regressor_type = regressor_type
        self.joint_rep_type = cfg.get('joint_rep', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        
        if regressor_type == 'pose':
            npose = self.joint_rep_dim * (cfg.get('num_joints', 34) + 1)
            self.npose = npose
            output_dim = npose
        elif regressor_type == 'shape':
            output_dim = cfg.get('shape_dim', 41)
        elif regressor_type == 'camera':
            output_dim = 3
        
        self.input_is_mean_shape = cfg.get('transformer_input', 'zero') == 'mean_shape'
        
        transformer_args = dict(
            num_tokens=1,
            token_dim=(output_dim + 10 + 3) if self.input_is_mean_shape else 1,
            dim=cfg.get('transformer_dim', 1024),
            depth=cfg.get('transformer_depth', 1),
            heads=cfg.get('transformer_heads', 8),
            mlp_dim=cfg.get('transformer_mlp_dim', 2048),
            dim_head=cfg.get('transformer_dim_head', 64),
            dropout=cfg.get('transformer_dropout', 0.1),
            emb_dropout=cfg.get('transformer_emb_dropout', 0.0),
        )
        
        # Override with any specific config values
        transformer_args.update(cfg.get('transformer_decoder', {}))
        
        self.transformer = TransformerDecoder(**transformer_args)
        dim = transformer_args['dim']
        
        self.decoder = nn.Linear(dim, output_dim)
        
        # Initialize decoder weights
        if cfg.get('init_decoder_xavier', False):
            nn.init.xavier_uniform_(self.decoder.weight, gain=0.01)
        
        # Initialize with zeros or mean values
        if regressor_type == 'pose':
            init_value = torch.zeros(size=(1, output_dim), dtype=torch.float32)
        elif regressor_type == 'shape':
            init_value = torch.zeros(size=(1, output_dim), dtype=torch.float32)
        elif regressor_type == 'camera':
            init_value = torch.tensor([[0.9, 0, 0]], dtype=torch.float32)
        
        self.register_buffer('init_value', init_value)
        self.ief_iters = cfg.get('ief_iters', 3)
    
    def forward(self, features):
        """
        Args:
            features: (B, H*W, C) feature maps from backbone
        Returns:
            output: (B, output_dim) regressed parameters
        """
        batch_size = features.shape[0]
        
        # Initialize with zero or mean shape
        pred = self.init_value.expand(batch_size, -1)
        pred_list = []
        
        for i in range(self.ief_iters):
            # Input token to transformer
            if self.input_is_mean_shape:
                # For pose/shape, concatenate with other params if needed
                token = pred[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(features.device)
            
            # Pass through transformer
            token_out = self.transformer(token, context=features)
            token_out = token_out.squeeze(1)  # (B, C)
            
            # Readout and residual connection
            pred = self.decoder(token_out) + pred
            pred_list.append(pred)
        
        return pred, pred_list


class CategoryRegressors(nn.Module):
    """Category-specific regressors: 5 pose + 5 shape + 1 shared camera."""
    
    def __init__(self, cfg, num_categories=5):
        """
        Args:
            cfg: Configuration dict
            num_categories: Number of categories (5)
        """
        super().__init__()
        self.num_categories = num_categories
        self.cfg = cfg
        
        # Create 5 pose regressors (one per category)
        self.pose_regressors = nn.ModuleList([
            SingleRegressorHead(cfg, output_dim=None, regressor_type='pose')
            for _ in range(num_categories)
        ])
        
        # Create 5 shape regressors (one per category)
        self.shape_regressors = nn.ModuleList([
            SingleRegressorHead(cfg, output_dim=None, regressor_type='shape')
            for _ in range(num_categories)
        ])
        
        # Create 1 shared camera regressor
        self.camera_regressor = SingleRegressorHead(cfg, output_dim=None, regressor_type='camera')
        
        # Get output dimensions
        joint_rep_type = cfg.get('joint_rep', '6d')
        joint_rep_dim = {'6d': 6, 'aa': 3}[joint_rep_type]
        self.npose = joint_rep_dim * (cfg.get('num_joints', 34) + 1)
        self.shape_dim = cfg.get('shape_dim', 41)
        self.joint_rep_type = joint_rep_type
    
    def forward(self, features, category_indices):
        """
        Args:
            features: (B, H*W, C) feature maps from backbone
            category_indices: (B,) category indices for routing
        Returns:
            pred_pose: (B, npose) pose predictions
            pred_shape: (B, shape_dim) shape predictions
            pred_camera: (B, 3) camera predictions
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Route to category-specific pose and shape regressors
        # Process each category separately for efficiency
        pred_pose_list = []
        pred_shape_list = []
        
        # Group samples by category
        for cat_idx in range(self.num_categories):
            mask = (category_indices == cat_idx)
            if mask.any():
                cat_features = features[mask]
                
                # Get pose and shape from category-specific regressors
                pose_pred, _ = self.pose_regressors[cat_idx](cat_features)
                shape_pred, _ = self.shape_regressors[cat_idx](cat_features)
                
                pred_pose_list.append((mask, pose_pred))
                pred_shape_list.append((mask, shape_pred))
        
        # Reassemble in original order
        pred_pose = torch.zeros(batch_size, self.npose, device=device)
        pred_shape = torch.zeros(batch_size, self.shape_dim, device=device)
        
        for mask, pred in pred_pose_list:
            pred_pose[mask] = pred
        for mask, pred in pred_shape_list:
            pred_shape[mask] = pred
        
        # Camera is shared - process all at once
        camera_pred, _ = self.camera_regressor(features)
        
        return pred_pose, pred_shape, camera_pred
    
    def convert_pose_to_rotmat(self, pred_pose):
        """Convert pose representation to rotation matrices."""
        batch_size = pred_pose.shape[0]
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]
        
        pred_pose_rotmat = joint_conversion_fn(pred_pose)
        pred_pose_rotmat = pred_pose_rotmat.view(batch_size, self.cfg.get('num_joints', 34) + 1, 3, 3)
        
        return {
            'global_orient': pred_pose_rotmat[:, [0]],
            'pose': pred_pose_rotmat[:, 1:],
        }

