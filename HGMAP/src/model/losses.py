import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MicrobiomeLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha  # Weight for focal loss
        self.beta = beta   # Weight for taxonomic loss
        self.gamma = gamma # Weight for compositional loss
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        
    def focal_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        bce_loss = F.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.focal_gamma
        
        if self.class_weights is not None:
            focal_term = focal_term * self.class_weights
            
        return (focal_term * bce_loss).mean()
    
    def taxonomic_loss(
        self,
        features: torch.Tensor,
        y_true: torch.Tensor,
        taxonomy_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Loss term incorporating taxonomic relationships"""
        # Calculate pairwise similarities
        sim_matrix = torch.mm(features, features.t())
        
        # Weight similarities by taxonomic relationships
        weighted_sim = sim_matrix * taxonomy_matrix
        
        # Calculate loss based on class relationships
        same_class = (y_true.unsqueeze(0) == y_true.unsqueeze(1)).float()
        diff_class = 1 - same_class
        
        pos_loss = -torch.log(weighted_sim + 1e-6) * same_class
        neg_loss = -torch.log(1 - weighted_sim + 1e-6) * diff_class
        
        return (pos_loss.sum() + neg_loss.sum()) / (len(y_true) ** 2)
    
    def compositional_loss(
        self,
        abundances: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Loss term for compositional nature of microbiome data"""
        # Center log-ratio transform
        clr_abundances = torch.log(abundances + 1e-6)
        clr_abundances = clr_abundances - clr_abundances.mean(dim=1, keepdim=True)
        
        # Correlation between predictions and abundances
        corr_matrix = torch.corrcoef(
            torch.stack([y_pred.squeeze(), clr_abundances.mean(dim=1)])
        )
        compositional_term = 1 - torch.abs(corr_matrix[0, 1])
        
        return compositional_term
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        features: torch.Tensor,
        abundances: torch.Tensor,
        taxonomy_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Combined loss function"""
        focal = self.focal_loss(y_pred, y_true)
        taxonomic = self.taxonomic_loss(features, y_true, taxonomy_matrix)
        compositional = self.compositional_loss(abundances, y_pred)
        
        return (
            self.alpha * focal +
            self.beta * taxonomic +
            self.gamma * compositional
        ) 