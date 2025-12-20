import torch
import numpy as np
from typing import Dict, List, Union


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrustes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """
    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1 ** 2).sum(dim=(1, 2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1.float(), X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K.float())
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1).float()
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U.float(), Vh.float()).float()))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(R.float(), mu1.float())

    # 7. Error:
    S1_hat = scale * torch.matmul(R.float(), S1.float()).float() + t

    return S1_hat.permute(0, 2, 1)


def compute_pck(pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor,
                threshold: float, normalize_by: str = 'image_size', 
                image_size: int = 256, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute PCK (Percentage of Correct Keypoints) metric.
    
    Args:
        pred_keypoints_2d: (B, N, 2) predicted 2D keypoints
        gt_keypoints_2d: (B, N, 3) ground truth 2D keypoints with confidence
        threshold: PCK threshold (0.1, 0.15, etc.)
        normalize_by: 'image_size' or 'head_size' or 'mask'
        image_size: Image size for normalization
        mask: (B, H, W) segmentation mask for head size computation
    
    Returns:
        pck: (B,) PCK scores per sample
    """
    conf = gt_keypoints_2d[:, :, -1]  # (B, N)
    gt_kp = gt_keypoints_2d[:, :, :-1]  # (B, N, 2)
    
    # Compute distances
    dist = torch.norm(pred_keypoints_2d - gt_kp, dim=-1)  # (B, N)
    
    # Normalize by appropriate factor
    if normalize_by == 'image_size':
        norm_factor = image_size
    elif normalize_by == 'head_size':
        # Compute head size from mask
        if mask is not None:
            seg_area = torch.sum(mask.reshape(mask.shape[0], -1), dim=-1).unsqueeze(-1)  # (B, 1)
            norm_factor = torch.sqrt(seg_area)
        else:
            norm_factor = image_size
    elif normalize_by == 'mask':
        if mask is not None:
            seg_area = torch.sum(mask.reshape(mask.shape[0], -1), dim=-1).unsqueeze(-1)  # (B, 1)
            norm_factor = torch.sqrt(seg_area)
        else:
            norm_factor = image_size
    else:
        norm_factor = image_size
    
    # Normalize distances
    if isinstance(norm_factor, torch.Tensor):
        normalized_dist = dist / norm_factor
    else:
        normalized_dist = dist / norm_factor
    
    # Compute hits
    hits = normalized_dist < threshold
    
    # Weight by confidence
    total_visible = torch.sum(conf, dim=-1)  # (B,)
    correct = torch.sum(hits.float() * conf, dim=-1)  # (B,)
    
    # Avoid division by zero
    total_visible = torch.clamp(total_visible, min=1.0)
    pck = correct / total_visible
    
    return pck


def compute_pck_at_threshold(pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor,
                            threshold: float, image_size: int = 256, 
                            mask: torch.Tensor = None) -> float:
    """Compute P@threshold (e.g., P@0.1, P@0.15)."""
    pck = compute_pck(pred_keypoints_2d, gt_keypoints_2d, threshold, 
                     normalize_by='image_size', image_size=image_size, mask=mask)
    return pck.mean().item()


def compute_pck_at_head(pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor,
                        threshold: float, mask: torch.Tensor, image_size: int = 256) -> float:
    """Compute P@H (PCK@HTH) - PCK normalized by head size."""
    pck = compute_pck(pred_keypoints_2d, gt_keypoints_2d, threshold,
                     normalize_by='head_size', image_size=image_size, mask=mask)
    return pck.mean().item()


def compute_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """
    Compute PA-MPJPE (Procrustes-aligned mean per-joint position error).
    
    Args:
        pred_joints: (B, N, 3) predicted 3D joints
        gt_joints: (B, N, 3) ground truth 3D joints
    
    Returns:
        pa_mpjpe: Scalar PA-MPJPE in mm
    """
    # Align predictions to ground truth using Procrustes
    S1_hat = compute_similarity_transform(pred_joints, gt_joints)
    
    # Compute mean per-joint error
    pa_mpjpe = torch.sqrt(((S1_hat - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1)  # (B,)
    pa_mpjpe = pa_mpjpe.mean().cpu().numpy() * 1000  # Convert to mm
    
    return float(pa_mpjpe)


def compute_pa_mpvpe(pred_vertices: torch.Tensor, gt_vertices: torch.Tensor) -> float:
    """
    Compute PA-MPVPE (Procrustes-aligned mean per-vertex position error).
    
    Args:
        pred_vertices: (B, V, 3) predicted 3D vertices
        gt_vertices: (B, V, 3) ground truth 3D vertices
    
    Returns:
        pa_mpvpe: Scalar PA-MPVPE in mm
    """
    # Align predictions to ground truth using Procrustes
    S1_hat = compute_similarity_transform(pred_vertices, gt_vertices)
    
    # Compute mean per-vertex error
    pa_mpvpe = torch.sqrt(((S1_hat - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1)  # (B,)
    pa_mpvpe = pa_mpvpe.mean().cpu().numpy() * 1000  # Convert to mm
    
    return float(pa_mpvpe)


class Evaluator:
    """Evaluator for computing all specified metrics."""
    
    def __init__(self, image_size: int = 256, pelvis_ind: int = 0, 
                 pck_thresholds: List[float] = None, pck_head_threshold: float = 0.5):
        """
        Args:
            image_size: Image size for normalization
            pelvis_ind: Index of pelvis joint for alignment
            pck_thresholds: List of PCK thresholds (e.g., [0.1, 0.15])
            pck_head_threshold: Threshold for PCK@H (head-normalized PCK)
        """
        self.image_size = image_size
        self.pelvis_ind = pelvis_ind
        self.pck_thresholds = pck_thresholds if pck_thresholds is not None else [0.1, 0.15]
        self.pck_head_threshold = pck_head_threshold
    
    def evaluate(self, output: Dict, batch: Dict) -> Dict[str, float]:
        """
        Evaluate model output and compute all metrics.
        
        Args:
            output: Model output dictionary
            batch: Input batch dictionary
        
        Returns:
            Dictionary with metric names and values
        """
        metrics = {}
        
        # 2D keypoint metrics
        if 'pred_keypoints_2d' in output and 'keypoints_2d' in batch:
            pred_kp2d = output['pred_keypoints_2d']
            gt_kp2d = batch['keypoints_2d']
            mask = batch.get('mask', None)
            
            # PCK at configured thresholds
            for threshold in self.pck_thresholds:
                metric_name = f'P@{threshold}'
                metrics[metric_name] = compute_pck_at_threshold(
                    pred_kp2d, gt_kp2d, threshold=threshold, image_size=self.image_size, mask=mask
                )
            
            # P@H (PCK@HTH) - requires mask
            if mask is not None:
                metrics['P@H'] = compute_pck_at_head(
                    pred_kp2d, gt_kp2d, threshold=self.pck_head_threshold, 
                    mask=mask, image_size=self.image_size
                )
        
        # 3D keypoint metrics
        if 'pred_keypoints_3d' in output and 'keypoints_3d' in batch:
            pred_kp3d = output['pred_keypoints_3d']
            gt_kp3d = batch['keypoints_3d'][:, :, :-1]  # Remove confidence
            
            # Align to pelvis
            pred_kp3d_aligned = pred_kp3d - pred_kp3d[:, self.pelvis_ind, :].unsqueeze(1)
            gt_kp3d_aligned = gt_kp3d - gt_kp3d[:, self.pelvis_ind, :].unsqueeze(1)
            
            # PAJ (PA-MPJPE)
            metrics['PAJ'] = compute_pa_mpjpe(pred_kp3d_aligned, gt_kp3d_aligned)
        
        # Vertex metrics
        if 'pred_vertices' in output and 'vertices' in batch:
            pred_verts = output['pred_vertices']
            gt_verts = batch['vertices']
            
            # PAV (PA-MPVPE)
            metrics['PAV'] = compute_pa_mpvpe(pred_verts, gt_verts)
        
        return metrics

