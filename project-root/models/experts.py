import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import LayerNorm2d, SDTAEncoder


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


class ExpertRouter(nn.Module):
    """
    Dynamic router for selecting experts based on input features.
    """

    def __init__(self, dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Get routing weights
        weights = self.gate(x)  # [B, num_experts]
        return weights


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with dynamic routing.
    """

    def __init__(self, dim, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            TextureExpert(dim),
            AttentionExpert(dim),
            HybridExpert(dim)
        ])
        self.router = ExpertRouter(dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x):
        # Get routing weights
        routing_weights = self.router(x)  # [B, num_experts]

        # Process through each expert
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, C, H, W]

        # Apply routing weights
        routing_weights = routing_weights.view(x.size(0), self.num_experts, 1, 1, 1)
        output = (expert_outputs * routing_weights).sum(dim=1)

        # Calculate auxiliary loss (load balancing)
        aux_loss = self.calculate_aux_loss(routing_weights.squeeze(-1).squeeze(-1).squeeze(-1))

        return output, aux_loss

    def calculate_aux_loss(self, routing_weights):
        """
        Calculate auxiliary loss for load balancing.
        """
        # Encourage equal distribution across experts
        mean_routing = routing_weights.mean(dim=0)
        target_dist = torch.ones_like(mean_routing) / self.num_experts
        aux_loss = F.mse_loss(mean_routing, target_dist)
        return aux_loss * 0.01  # Scale down the auxiliary loss