import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps (B, C, H, W).
    """
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class EdgeNeXtBackbone(nn.Module):
    """
    EdgeNeXt backbone for multi-scale feature extraction.
    """
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
            stage = nn.Sequential(*[
                nn.Conv2d(dims[i], dims[i], kernel_size=3, padding=1) for _ in range(depths[i])
            ])
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