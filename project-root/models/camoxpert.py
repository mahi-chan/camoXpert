import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import EdgeNeXtBackbone
from experts import MoELayer
from fusion import BiLevelFusion
from segmentation_head import SegmentationHead


class CamoXpert(nn.Module):
    """
    CamoXpert: Dynamic Neural Network for Adaptive Camouflaged Object Detection

    Architecture:
    1. EdgeNeXt Backbone for multi-scale feature extraction
    2. MoE layers for adaptive processing
    3. Bi-level fusion for combining features
    4. Segmentation head for final prediction
    """

    def __init__(self, in_channels=3, num_classes=1, depths=[3, 3, 9, 3], dims=[48, 96, 160, 256]):
        super().__init__()

        # Backbone
        self.backbone = EdgeNeXtBackbone(in_channels, depths, dims)

        # MoE layers for each stage
        self.moe_layers = nn.ModuleList([
            MoELayer(dim) for dim in dims
        ])

        # Feature fusion
        self.fusion = BiLevelFusion(dims)

        # Segmentation head
        self.seg_head = SegmentationHead(in_dim=64, num_classes=num_classes)

    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)

        # Apply MoE to each feature level
        enhanced_features = []
        total_aux_loss = 0
        for i, (feat, moe) in enumerate(zip(features, self.moe_layers)):
            enhanced_feat, aux_loss = moe(feat)
            enhanced_features.append(enhanced_feat)
            total_aux_loss += aux_loss

        # Fuse features
        fused = self.fusion(enhanced_features)

        # Generate final prediction
        mask = self.seg_head(fused)

        # Apply sigmoid for binary segmentation
        mask = torch.sigmoid(mask)

        return mask, total_aux_loss