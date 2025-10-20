# camoxpert_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class StructureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_edges(self, x):
        edges_x = F.conv2d(x, self.sobel_x, padding=1)
        edges_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)

    def forward(self, pred, target):
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        edge_loss = F.mse_loss(pred_edges, target_edges)
        pred_mean, target_mean = pred.mean(), target.mean()
        pred_std, target_std = pred.std(), target.std()
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * pred_mean * target_mean + c1) * (2 * covariance + c2)) / \
               ((pred_mean ** 2 + target_mean ** 2 + c1) * (pred_std ** 2 + target_std ** 2 + c2))
        region_loss = 1 - ssim
        return edge_loss + region_loss

class CamoXpertLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.3, structure_weight=0.4):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.structure_loss = StructureLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.structure_weight = structure_weight

    def forward(self, pred, target, aux_loss=0):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        structure = self.structure_loss(pred, target)
        total_loss = (self.bce_weight * bce +
                      self.dice_weight * dice +
                      self.structure_weight * structure +
                      aux_loss)
        return total_loss, {
            'total': total_loss.item(),
            'bce': bce.item(),
            'dice': dice.item(),
            'structure': structure.item(),
            'aux': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        }