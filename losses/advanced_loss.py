import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedCODLoss(nn.Module):
    """Advanced loss function combining multiple objectives"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

        # Loss weights optimized for camouflage
        self.w_bce = 0.2
        self.w_focal = 0.25
        self.w_iou = 0.25
        self.w_boundary = 0.2
        self.w_deep = 0.1  # Deep supervision weight

        # Sobel kernels for boundary detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for handling hard examples"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1 - pt) ** gamma * bce
        return focal_loss.mean()

    def iou_loss(self, pred, target):
        """IoU loss with positive weighting"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Weight positive samples more
        weights = torch.where(target > 0.5,
                              torch.ones_like(target) * 2.0,
                              torch.ones_like(target))

        intersection = (pred * target * weights).sum()
        union = (pred * weights).sum() + (target * weights).sum() - intersection

        return 1 - (intersection + 1e-6) / (union + 1e-6)

    def boundary_loss(self, pred, target):
        """
        Boundary-aware loss using Sobel edge detection
        FIXED: Use MSE loss instead of BCE for continuous edge values
        """
        # Compute edges using Sobel filters
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-8)

        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-8)

        # FIXED: Normalize both to [0, 1] for stable comparison
        pred_edges_norm = pred_edges / (pred_edges.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        target_edges_norm = target_edges / (target_edges.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

        # Use MSE loss (works for any value range, more stable than BCE for edges)
        boundary_loss = F.mse_loss(pred_edges_norm, target_edges_norm)

        return boundary_loss

    def deep_supervision_loss(self, deep_outputs, target):
        """Loss for intermediate predictions"""
        total_loss = 0
        for output in deep_outputs:
            total_loss += self.bce(output, target)
        return total_loss / len(deep_outputs) if len(deep_outputs) > 0 else 0

    def forward(self, pred, target, aux_loss=0, deep_outputs=None):
        """
        Args:
            pred: Main prediction
            target: Ground truth
            aux_loss: Auxiliary loss from MoE routing
            deep_outputs: List of intermediate predictions (optional)
        """
        # Main losses
        bce_loss = self.bce(pred, target)
        focal_loss = self.focal_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        boundary_loss = self.boundary_loss(pred, target)

        # Deep supervision
        deep_loss = 0
        if deep_outputs is not None and len(deep_outputs) > 0:
            deep_loss = self.deep_supervision_loss(deep_outputs, target)

        # Combined loss
        total_loss = (
                self.w_bce * bce_loss +
                self.w_focal * focal_loss +
                self.w_iou * iou_loss +
                self.w_boundary * boundary_loss +
                self.w_deep * deep_loss +
                0.01 * aux_loss
        )

        return total_loss, {
            'bce': bce_loss.item(),
            'focal': focal_loss.item(),
            'iou': iou_loss.item(),
            'boundary': boundary_loss.item(),
            'deep': deep_loss if isinstance(deep_loss, float) else deep_loss.item(),
            'aux': aux_loss if isinstance(aux_loss, float) else aux_loss.item()
        }