import torch


class CategoryRouter:
    """Routes features to category-specific regressors."""
    
    def __init__(self, use_gt_for_routing=True):
        """
        Args:
            use_gt_for_routing: If True, use ground truth category during training.
                               If False, use predicted category.
        """
        self.use_gt_for_routing = use_gt_for_routing
    
    def get_routing_indices(self, predicted_category, gt_category=None, training=True):
        """
        Get category indices for routing.
        
        Args:
            predicted_category: (B,) predicted category indices
            gt_category: (B,) ground truth category indices (optional)
            training: Whether in training mode
        
        Returns:
            routing_indices: (B,) category indices to use for routing
        """
        if training and self.use_gt_for_routing and gt_category is not None:
            return gt_category
        else:
            return predicted_category
    
    def route(self, features, category_indices):
        """
        Route features based on category indices.
        This is a placeholder - actual routing happens in CategoryRegressors.
        
        Args:
            features: (B, H*W, C) feature maps
            category_indices: (B,) category indices
        
        Returns:
            features: Same features (routing handled by regressors)
            category_indices: Category indices to use
        """
        return features, category_indices

