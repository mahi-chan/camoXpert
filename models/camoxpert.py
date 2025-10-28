import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.experts import MoELayer  # Import the CORRECT MoELayer with specialized experts


class SDTABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Add channel attention
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.use_attention:
            x = x * self.channel_attention(x)

        return x


class CamoXpert(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True,
                 backbone='edgenext_base_usi', num_experts=4):
        super().__init__()

        self.backbone_name = backbone

        # Create backbone
        print(f"Creating backbone: {backbone}")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        self.feature_dims = [f['num_chs'] for f in self.backbone.feature_info]
        print(f"Feature dimensions: {self.feature_dims}")

        # FIXED: Use specialized MoE layers with 4 experts
        print(f"\nInitializing {num_experts} specialized experts per layer...")
        self.sdta_blocks = nn.ModuleList([SDTABlock(dim) for dim in self.feature_dims])
        self.moe_layers = nn.ModuleList([
            MoELayer(dim, num_experts=num_experts) for dim in self.feature_dims
        ])

        # Decoder
        decoder_channels = [512, 256, 128, 64]
        self.decoder_blocks = nn.ModuleList()

        # First decoder block
        self.decoder_blocks.append(
            DecoderBlock(self.feature_dims[-1], decoder_channels[0], use_attention=True)
        )

        # Subsequent decoder blocks with skip connections
        for i in range(1, len(decoder_channels)):
            skip_idx = len(self.feature_dims) - 1 - i
            if skip_idx >= 0:
                in_channels = decoder_channels[i - 1] + self.feature_dims[skip_idx]
            else:
                in_channels = decoder_channels[i - 1]

            self.decoder_blocks.append(
                DecoderBlock(in_channels, decoder_channels[i], use_attention=True)
            )

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        # Deep supervision heads
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(decoder_channels[i], num_classes, 1)
            for i in range(len(decoder_channels))
        ])

        print(f"\nCamoXpert initialized with:")
        print(f"  - Backbone: {backbone}")
        print(f"  - 4 Specialized Experts per layer:")
        print(f"    1. TextureExpert (multi-scale patterns)")
        print(f"    2. AttentionExpert (global context)")
        print(f"    3. HybridExpert (local-global fusion)")
        print(f"    4. FrequencyExpert (frequency analysis)")
        print(f"  - {len(self.feature_dims)} encoder stages")
        print(f"  - {len(decoder_channels)} decoder stages")
        print(f"  - Deep supervision: enabled\n")

    def forward(self, x, return_deep_supervision=False):
        input_size = x.shape[2:]

        # Extract multi-scale features from backbone
        features = self.backbone(x)

        # Enhance features with SDTA and specialized MoE
        enhanced_features = []
        total_aux_loss = 0
        for i, (feat, sdta, moe) in enumerate(zip(features, self.sdta_blocks, self.moe_layers)):
            # Apply SDTA enhancement
            feat = sdta(feat)

            # Apply MoE with 4 specialized experts
            feat, aux_loss = moe(feat)

            enhanced_features.append(feat)
            total_aux_loss += aux_loss

        # Decode with skip connections
        x = enhanced_features[-1]
        x = self.decoder_blocks[0](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        deep_outputs = []

        for i in range(1, len(self.decoder_blocks)):
            skip_idx = len(enhanced_features) - 1 - i

            if 0 <= skip_idx < len(enhanced_features):
                skip = enhanced_features[skip_idx]
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = self.decoder_blocks[i](x)

            # Deep supervision
            if return_deep_supervision and i < len(self.deep_supervision):
                deep_out = self.deep_supervision[i](x)
                deep_out = F.interpolate(deep_out, size=input_size, mode='bilinear', align_corners=False)
                deep_outputs.append(torch.sigmoid(deep_out))

            if i < len(self.decoder_blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Final prediction
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        final_output = torch.sigmoid(x)

        if return_deep_supervision:
            return final_output, total_aux_loss, deep_outputs

        return final_output, total_aux_loss


# Utility function for model analysis
def analyze_model(model, input_size=(1, 3, 416, 416), device='cuda'):
    """
    Analyze the CamoXpert model structure and expert usage.

    Args:
        model: CamoXpert model instance
        input_size: Input tensor size
        device: Device to run on
    """
    from models.experts import visualize_expert_routing

    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    print("\n" + "=" * 70)
    print("MODEL ANALYSIS")
    print("=" * 70)

    # Forward pass
    with torch.no_grad():
        output, aux_loss = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Aux loss:     {aux_loss.item():.6f}")

    # Analyze expert routing at each stage
    features = model.backbone(dummy_input)
    print(f"\n Expert Routing Analysis:")
    print("-" * 70)

    for i, (feat, sdta, moe) in enumerate(zip(features, model.sdta_blocks, model.moe_layers)):
        print(f"\nStage {i + 1} (dim={feat.shape[1]}):")
        feat = sdta(feat)
        visualize_expert_routing(moe, feat)

    print("\n" + "=" * 70)