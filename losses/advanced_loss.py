"""
Advanced Loss Function for Camouflaged Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedCODLoss(nn.Module):
    """
    Combined loss for COD:
    - BCE with Logits (safe with AMP)
    - IoU Loss
    - Edge-aware Loss
    - Auxiliary MoE Loss
    """

    def __init__(self, bce_weight=1.0, iou_weight=1.0, edge_weight=0.5, aux_weight=0.1):
        super().__init__()

        # Use BCEWithLogitsLoss (safe with autocast)
        self.bce = nn.BCEWithLogitsLoss()

        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight

    def iou_loss(self, pred, target):
        """IoU loss"""
        # Clamp logits to prevent NaN in mixed precision
        pred = torch.clamp(pred, min=-15, max=15)
        pred = torch.sigmoid(pred)  # Apply sigmoid here

        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)

        return 1 - iou.mean()

    def edge_loss(self, pred, target):
        """Edge-aware loss"""
        # Clamp logits to prevent NaN in mixed precision
        pred = torch.clamp(pred, min=-15, max=15)
        pred = torch.sigmoid(pred)  # Apply sigmoid here

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # Compute edges
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        # Ensure non-negative before sqrt
        pred_edge = torch.sqrt(torch.clamp(pred_edge_x ** 2 + pred_edge_y ** 2, min=0) + 1e-5)

        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-5)

        return F.mse_loss(pred_edge, target_edge)

    def forward(self, pred, target, aux_loss=None, deep_outputs=None):
        """
        Args:
            pred: Main prediction (logits, before sigmoid)
            target: Ground truth masks
            aux_loss: Auxiliary MoE loss
            deep_outputs: Deep supervision outputs (list of logits)
        """

        # Clamp main prediction logits to prevent NaN in mixed precision
        pred = torch.clamp(pred, min=-15, max=15)

        # Main losses (pred is logits)
        bce = self.bce(pred, target)
        iou = self.iou_loss(pred, target)  # IoU handles sigmoid internally
        edge = self.edge_loss(pred, target)  # Edge handles sigmoid internally

        total_loss = (
                self.bce_weight * bce +
                self.iou_weight * iou +
                self.edge_weight * edge
        )

        loss_dict = {
            'bce': bce.item(),
            'iou': iou.item(),
            'edge': edge.item()
        }

        # Add auxiliary MoE loss
        if aux_loss is not None:
            total_loss += self.aux_weight * aux_loss
            loss_dict['aux'] = aux_loss.item()

        # Add deep supervision losses
        if deep_outputs is not None:
            deep_loss = 0
            for i, deep_pred in enumerate(deep_outputs):
                # Clamp deep supervision logits to prevent NaN
                deep_pred = torch.clamp(deep_pred, min=-15, max=15)

                # Resize target to match deep_pred size
                if deep_pred.shape[2:] != target.shape[2:]:
                    target_resized = F.interpolate(target, size=deep_pred.shape[2:],
                                                   mode='bilinear', align_corners=False)
                else:
                    target_resized = target

                deep_loss += self.bce(deep_pred, target_resized)

            deep_loss /= len(deep_outputs)
            total_loss += 0.4 * deep_loss
            loss_dict['deep'] = deep_loss.item()

        return total_loss, loss_dict