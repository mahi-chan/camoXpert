import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=3):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class CamoXpert(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True, depths=None, dims=None):
        super().__init__()

        self.backbone = timm.create_model(
            'edgenext_small',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        self.feature_dims = [f['num_chs'] for f in self.backbone.feature_info]

        self.sdta_blocks = nn.ModuleList([SDTABlock(dim) for dim in self.feature_dims])
        self.moe_layers = nn.ModuleList([MoELayer(dim) for dim in self.feature_dims])

        decoder_channels = [256, 128, 64, 32]
        self.decoder_blocks = nn.ModuleList()
        in_ch = self.feature_dims[-1]
        for out_ch in decoder_channels:
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch))
            in_ch = out_ch

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(feat_dim, dec_ch, 1)
            for feat_dim, dec_ch in zip(self.feature_dims[:-1][::-1], decoder_channels[1:])
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)

        enhanced_features = []
        total_aux_loss = 0
        for feat, sdta, moe in zip(features, self.sdta_blocks, self.moe_layers):
            feat = sdta(feat)
            feat, aux_loss = moe(feat)
            enhanced_features.append(feat)
            total_aux_loss += aux_loss

        x = enhanced_features[-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(self.skip_convs):
                skip = self.skip_convs[i](enhanced_features[-(i + 2)])
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip

        x = self.final_conv(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(x), total_aux_loss