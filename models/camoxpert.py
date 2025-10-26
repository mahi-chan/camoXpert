import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4):  # Increased to 4 experts
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, padding=1),
                nn.BatchNorm2d(dim * 2),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim, 3, padding=1),
                nn.BatchNorm2d(dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x_pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        router_logits = self.router(x_pooled)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        router_probs = router_probs.view(B, self.num_experts, 1, 1, 1)
        output = (expert_outputs * router_probs).sum(dim=1)

        mean_routing = router_probs.squeeze(-1).squeeze(-1).squeeze(-1).mean(dim=0)
        target_dist = torch.ones_like(mean_routing) / self.num_experts
        aux_loss = F.mse_loss(mean_routing, target_dist) * 0.01

        return output, aux_loss


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

        # Create backbone with edgenext_base_usi
        print(f"Creating backbone: {backbone}")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        self.feature_dims = [f['num_chs'] for f in self.backbone.feature_info]
        print(f"Feature dimensions: {self.feature_dims}")

        # Enhanced SDTA and MoE layers
        self.sdta_blocks = nn.ModuleList([SDTABlock(dim) for dim in self.feature_dims])
        self.moe_layers = nn.ModuleList([MoELayer(dim, num_experts) for dim in self.feature_dims])

        # Larger decoder for base models
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

        # Enhanced final conv with deep supervision
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        # Deep supervision heads for intermediate outputs
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(decoder_channels[i], num_classes, 1)
            for i in range(len(decoder_channels))
        ])

    def forward(self, x, return_deep_supervision=False):
        input_size = x.shape[2:]

        # Extract multi-scale features
        features = self.backbone(x)

        # Enhance features with SDTA and MoE
        enhanced_features = []
        total_aux_loss = 0
        for feat, sdta, moe in zip(features, self.sdta_blocks, self.moe_layers):
            feat = sdta(feat)
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