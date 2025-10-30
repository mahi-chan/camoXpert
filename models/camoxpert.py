"""
CamoXpert: Advanced Camouflaged Object Detection Model

Architecture:
- EdgeNeXt backbone for feature extraction
- SDTA (Selective Dual-axis Temporal Attention) enhancement
- Mixture of Experts (MoE) for specialized feature processing
- Progressive decoder with skip connections
- Deep supervision for better training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class CamoXpert(nn.Module):
    """
    CamoXpert: State-of-the-art Camouflaged Object Detection
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        backbone: str = 'edgenext_small',
        num_experts: int = 5
    ):
        """
        Initialize CamoXpert

        Args:
            in_channels: Input image channels (default: 3 for RGB)
            num_classes: Output classes (default: 1 for binary segmentation)
            pretrained: Use pretrained backbone weights
            backbone: Backbone architecture ('edgenext_small', 'edgenext_base', 'edgenext_base_usi')
            num_experts: Number of MoE experts to create (3-7)
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_classes = num_classes

        # Backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        self.feature_dims = self._detect_feature_dims()

        # SDTA Enhancement Blocks
        self.sdta_blocks = nn.ModuleList([
            SDTABlock(dim) for dim in self.feature_dims
        ])

        # Mixture of Experts Layers
        from models.experts import MoELayer
        top_k = max(2, num_experts // 2)

        self.moe_layers = nn.ModuleList([
            MoELayer(dim, num_experts=num_experts, top_k=top_k)
            for dim in self.feature_dims
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.feature_dims[3], self.feature_dims[2]),
            DecoderBlock(self.feature_dims[2], self.feature_dims[1]),
            DecoderBlock(self.feature_dims[1], self.feature_dims[0]),
            DecoderBlock(self.feature_dims[0], 64),
        ])

        # Deep Supervision Heads
        self.deep_heads = nn.ModuleList([
            nn.Conv2d(self.feature_dims[2], num_classes, kernel_size=1),
            nn.Conv2d(self.feature_dims[1], num_classes, kernel_size=1),
            nn.Conv2d(self.feature_dims[0], num_classes, kernel_size=1),
        ])

        # Final Prediction Head
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        print(f"✓ CamoXpert: {backbone} | {num_experts} experts (top-{top_k}) | {self.feature_dims}")

    def _create_backbone(self, backbone: str, pretrained: bool):
        """Create EdgeNeXt backbone"""
        try:
            import timm
            model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
            return model
        except Exception as e:
            print(f"⚠️  Error loading backbone: {e}")
            print("Attempting alternative backbone loading...")

            # Fallback
            import timm
            if 'small' in backbone:
                model = timm.create_model('edgenext_small', pretrained=pretrained, features_only=True)
            else:
                model = timm.create_model('edgenext_base', pretrained=pretrained, features_only=True)
            return model

    def _detect_feature_dims(self):
        """Dynamically detect feature dimensions by running a dummy forward pass"""
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dims = [f.shape[1] for f in features]
        self.backbone.train()
        return feature_dims

    def forward(self, x, return_deep_supervision=False):
        """Forward pass - returns logits (no sigmoid)"""
        B, _, H, W = x.shape

        # Encoder: Extract multi-scale features
        encoder_features = self.backbone(x)

        # Enhancement: Apply SDTA + MoE
        enhanced_features = []
        total_aux_loss = 0.0

        for feat, sdta, moe in zip(encoder_features, self.sdta_blocks, self.moe_layers):
            feat = sdta(feat)
            feat, aux_loss, _ = moe(feat)
            if aux_loss is not None:
                total_aux_loss += aux_loss
            enhanced_features.append(feat)

        # Decoder: Progressive upsampling with deep supervision
        deep_outputs = [] if return_deep_supervision else None

        x4 = enhanced_features[3]
        x3 = self.decoder_blocks[0](x4, enhanced_features[2])
        if return_deep_supervision:
            deep_outputs.append(self.deep_heads[0](x3))

        x2 = self.decoder_blocks[1](x3, enhanced_features[1])
        if return_deep_supervision:
            deep_outputs.append(self.deep_heads[1](x2))

        x1 = self.decoder_blocks[2](x2, enhanced_features[0])
        if return_deep_supervision:
            deep_outputs.append(self.deep_heads[2](x1))

        x0 = self.decoder_blocks[3](x1, None)
        x0_up = F.interpolate(x0, size=(H, W), mode='bilinear', align_corners=False)

        # Final prediction (logits)
        pred = self.final_conv(x0_up)

        return pred, total_aux_loss, deep_outputs


class SDTABlock(nn.Module):
    """
    Optimized Selective Dual-axis Temporal Attention
    Memory-efficient with depthwise separable convolutions
    """

    def __init__(self, dim, reduction=8):
        super().__init__()

        # Channel attention (lightweight)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 8), dim, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention (depthwise for efficiency)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa

        # Residual connection
        return x + identity * 0.1


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connections
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Fusion of upsampled and skip features
        if out_channels == 64:
            # Last decoder block (no skip connection)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # With skip connection
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, skip=None):
        """
        Args:
            x: Input features from previous decoder stage
            skip: Skip connection features from encoder (can be None for last stage)
        """
        x = self.upsample(x)

        if skip is not None:
            # Resize if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == '__main__':
    # Test model
    print("Testing CamoXpert...")

    model = CamoXpert(
        in_channels=3,
        num_classes=1,
        pretrained=False,
        backbone='edgenext_small',
        num_experts=5
    )

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total/1e6:.2f}M")
    print(f"Trainable parameters: {trainable/1e6:.2f}M")

    # Test forward pass
    x = torch.randn(2, 3, 288, 288)

    print("\nTesting forward pass...")
    pred, aux_loss, deep = model(x, return_deep_supervision=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Aux loss: {aux_loss}")
    print(f"Deep supervision outputs: {len(deep) if deep else 0}")

    if deep:
        for i, d in enumerate(deep):
            print(f"  Deep {i+1}: {d.shape}")

    print("\n✓ Model test successful!")

    # Verify outputs are logits (not sigmoid)
    print(f"\nOutput range check:")
    print(f"  Min: {pred.min().item():.4f}")
    print(f"  Max: {pred.max().item():.4f}")
    print(f"  Mean: {pred.mean().item():.4f}")
    print(f"  (Should be unbounded, not [0, 1])")