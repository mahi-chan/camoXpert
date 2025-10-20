import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import LayerNorm2d
from models.backbone import SDTAEncoder  # Assuming SDTAEncoder is defined in `backbone.py`

class TextureExpert(nn.Module):
    """
    Texture-focused expert using multi-scale dilated convolutions.
    Captures fine-grained patterns critical for camouflage detection.
    """
    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, dilation=1),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=2, dilation=2),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=3, dilation=3),
            nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=4, dilation=4),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        multi_scale = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.fusion(multi_scale)
        return out + x


class AttentionExpert(nn.Module):
    """
    Attention-based expert for global context understanding.
    Uses efficient self-attention to capture long-range dependencies.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = SDTAEncoder(
            dim=dim,
            num_heads=num_heads,
            drop_path=0.1
        )

    def forward(self, x):
        return self.attention(x)


class HybridExpert(nn.Module):
    """
    Hybrid expert combining local and global processing.
    Balances efficiency and effectiveness.
    """
    def __init__(self, dim):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        local_feat = self.local_branch(x)
        global_weight = self.global_branch(x)
        out = local_feat * global_weight
        out = self.fusion(out)
        return out + x