import torch
import torch.nn as nn


class CategoryClassifier(nn.Module):
    """Simple MLP classifier for 5 animal categories."""
    
    def __init__(self, input_dim=256, hidden_dim=512, num_categories=5, dropout=0.1):
        """
        Args:
            input_dim: Dimension of CLS features (from ClassTokenHead)
            hidden_dim: Hidden dimension of MLP
            num_categories: Number of categories (5)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_categories = num_categories
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories)
        )
    
    def forward(self, cls_features):
        """
        Args:
            cls_features: (B, input_dim) CLS token features
        Returns:
            logits: (B, num_categories) category logits
            probs: (B, num_categories) category probabilities
        """
        logits = self.classifier(cls_features)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

