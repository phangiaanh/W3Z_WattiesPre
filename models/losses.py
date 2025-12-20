import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for category classification."""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred_logits, gt_category):
        """
        Args:
            pred_logits: (B, num_categories) category logits
            gt_category: (B,) ground truth category indices
        Returns:
            loss: scalar classification loss
        """
        return self.loss_fn(pred_logits, gt_category.long())


class Keypoint2DLoss(nn.Module):
    """2D keypoint loss."""
    
    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1' or 'l2'
        """
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
    
    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_keypoints_2d: (B, N, 2) predicted 2D keypoints
            gt_keypoints_2d: (B, N, 3) ground truth 2D keypoints with confidence
        Returns:
            loss: scalar 2D keypoint loss
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1)
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1, 2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):
    """3D keypoint loss."""
    
    def __init__(self, loss_type: str = 'l1', pelvis_id: int = 0):
        """
        Args:
            loss_type: 'l1' or 'l2'
            pelvis_id: Index of pelvis joint for alignment
        """
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
        self.pelvis_id = pelvis_id
    
    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_keypoints_3d: (B, N, 3) predicted 3D keypoints
            gt_keypoints_3d: (B, N, 4) ground truth 3D keypoints with confidence
        Returns:
            loss: scalar 3D keypoint loss
        """
        # Align to pelvis
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, self.pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d_aligned = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, self.pelvis_id, :-1].unsqueeze(dim=1)
        
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1)
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d_aligned)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):
    """Parameter loss for pose/shape/camera."""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, 
                has_param: torch.Tensor = None):
        """
        Args:
            pred_param: (B, ...) predicted parameters
            gt_param: (B, ...) ground truth parameters
            has_param: (B,) mask indicating which samples have ground truth
        Returns:
            loss: scalar parameter loss
        """
        if has_param is not None:
            mask = has_param.type(pred_param.dtype).view(-1, *([1] * (len(pred_param.shape) - 1)))
            loss = (mask * self.loss_fn(pred_param, gt_param)).sum()
        else:
            loss = self.loss_fn(pred_param, gt_param).sum()
        return loss


class CombinedLoss(nn.Module):
    """Combined loss for the full model."""
    
    def __init__(self, loss_weights: dict):
        """
        Args:
            loss_weights: Dictionary with loss weights:
                - classification: weight for classification loss
                - keypoints_2d: weight for 2D keypoint loss
                - keypoints_3d: weight for 3D keypoint loss
                - pose: weight for pose parameter loss
                - shape: weight for shape parameter loss
                - camera: weight for camera parameter loss
        """
        super().__init__()
        self.loss_weights = loss_weights
        
        self.classification_loss = ClassificationLoss()
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1', pelvis_id=0)
        self.parameter_loss = ParameterLoss()
    
    def forward(self, output: dict, batch: dict) -> dict:
        """
        Compute all losses.
        
        Args:
            output: Model output dictionary
            batch: Input batch dictionary
        
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Classification loss
        if 'category' in batch and self.loss_weights.get('classification', 0) > 0:
            cls_loss = self.classification_loss(output['category_logits'], batch['category'])
            losses['classification'] = cls_loss
            total_loss += self.loss_weights['classification'] * cls_loss
        
        # 2D keypoint loss
        if 'pred_keypoints_2d' in output and 'keypoints_2d' in batch and self.loss_weights.get('keypoints_2d', 0) > 0:
            kp2d_loss = self.keypoint_2d_loss(output['pred_keypoints_2d'], batch['keypoints_2d'])
            losses['keypoints_2d'] = kp2d_loss
            total_loss += self.loss_weights['keypoints_2d'] * kp2d_loss
        
        # 3D keypoint loss
        if 'pred_keypoints_3d' in output and 'keypoints_3d' in batch and self.loss_weights.get('keypoints_3d', 0) > 0:
            kp3d_loss = self.keypoint_3d_loss(output['pred_keypoints_3d'], batch['keypoints_3d'])
            losses['keypoints_3d'] = kp3d_loss
            total_loss += self.loss_weights['keypoints_3d'] * kp3d_loss
        
        # Parameter losses (if ground truth available)
        if 'smal_params' in batch:
            smal_params = batch['smal_params']
            has_params = batch.get('has_smal_params', {})
            
            # Pose loss
            if 'pose' in smal_params and self.loss_weights.get('pose', 0) > 0:
                # Convert pose to same format
                gt_pose = smal_params['pose']
                pred_pose = output['pred_pose']
                has_pose = has_params.get('pose', torch.ones(len(pred_pose), device=pred_pose.device))
                pose_loss = self.parameter_loss(pred_pose, gt_pose, has_pose)
                losses['pose'] = pose_loss
                total_loss += self.loss_weights['pose'] * pose_loss
            
            # Shape loss
            if 'betas' in smal_params and self.loss_weights.get('shape', 0) > 0:
                gt_shape = smal_params['betas']
                pred_shape = output['pred_shape']
                has_shape = has_params.get('betas', torch.ones(len(pred_shape), device=pred_shape.device))
                shape_loss = self.parameter_loss(pred_shape, gt_shape, has_shape)
                losses['shape'] = shape_loss
                total_loss += self.loss_weights['shape'] * shape_loss
        
        # Camera loss
        if 'camera' in batch and self.loss_weights.get('camera', 0) > 0:
            gt_camera = batch['camera']
            pred_camera = output['pred_camera']
            camera_loss = self.parameter_loss(pred_camera, gt_camera)
            losses['camera'] = camera_loss
            total_loss += self.loss_weights['camera'] * camera_loss
        
        losses['total'] = total_loss
        return losses

