import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x

class EdgeNeXtBackbone(nn.Module):
    def __init__(self, in_channels=3, depths=[3, 3, 9, 3], dims=[48, 96, 160, 256], drop_path_rate=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(*[nn.Conv2d(dims[i], dims[i], kernel_size=3, padding=1) for _ in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.num_stages = len(depths)
        self.dims = dims

    def forward(self, x):
        features = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class BiLevelFusion(nn.Module):
    def __init__(self, dims=[48, 96, 160, 256], out_dim=64):
        super().__init__()
        self.low_fusion = nn.Sequential(
            nn.Conv2d(dims[0] + dims[1], dims[0], 1),
            LayerNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )
        self.high_fusion = nn.Sequential(
            nn.Conv2d(dims[2] + dims[3], dims[2], 1),
            LayerNorm2d(dims[2]),
            nn.GELU(),
            nn.Conv2d(dims[2], dims[2], 3, padding=1),
            LayerNorm2d(dims[2]),
            nn.GELU()
        )
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(dims[0] + dims[2], out_dim * 2, 3, padding=1),
            LayerNorm2d(out_dim * 2),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
            LayerNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, features):
        f1, f2, f3, f4 = features
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        low_fused = self.low_fusion(torch.cat([f1, f2_up], dim=1))
        high_fused = self.high_fusion(torch.cat([f3, f4_up], dim=1))
        high_up = F.interpolate(high_fused, size=low_fused.shape[2:], mode='bilinear', align_corners=False)
        fused = self.cross_fusion(torch.cat([low_fused, high_up], dim=1))
        return fused

class SegmentationHead(nn.Module):
    def __init__(self, in_dim=64, num_classes=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            LayerNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        return self.decoder(x)

class CamoXpert(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, depths=[3, 3, 9, 3], dims=[48, 96, 160, 256]):
        super().__init__()
        self.backbone = EdgeNeXtBackbone(in_channels, depths, dims)
        self.fusion = BiLevelFusion(dims)
        self.seg_head = SegmentationHead(in_dim=64, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        fused = self.fusion(features)
        mask = self.seg_head(fused)
        mask = torch.sigmoid(mask)
        return mask