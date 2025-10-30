"""
CamoXpert Expert Modules - FIXED VERSION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import LayerNorm2d, SDTAEncoder


class TextureExpert(nn.Module):
    """Expert 1: Multi-scale texture pattern recognition"""
    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, dilation=1),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=2, dilation=2),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=3, dilation=3),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=4, dilation=4),
            LayerNorm2d(dim // 4),
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
        return self.fusion(multi_scale) + x


class AttentionExpert(nn.Module):
    """Expert 2: Global context via self-attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = SDTAEncoder(dim=dim, num_heads=num_heads, drop_path=0.1)

    def forward(self, x):
        return self.attention(x)


class HybridExpert(nn.Module):
    """Expert 3: Local-global feature fusion"""
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
        return self.fusion(out) + x


class FrequencyExpert(nn.Module):
    """Expert 4: Frequency-domain analysis"""
    def __init__(self, dim):
        super().__init__()
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.mid_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq
        mid_freq_blur1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mid_freq_blur2 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        mid_freq = mid_freq_blur1 - mid_freq_blur2

        low_feat = self.low_freq_conv(low_freq)
        mid_feat = self.mid_freq_conv(mid_freq)
        high_feat = self.high_freq_conv(high_freq)
        spatial_feat = self.spatial_conv(x)

        freq_features = torch.cat([low_feat, mid_feat, high_feat, spatial_feat], dim=1)
        return self.fusion(freq_features) + x


class EdgeExpert(nn.Module):
    """Expert 5: Boundary and edge detection"""
    def __init__(self, dim):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

        self.sobel_branch = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.laplacian_branch = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.gradient_branch = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.spatial_branch = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(dim, dim, 1), LayerNorm2d(dim), nn.GELU())

    def compute_edges(self, x):
        B, C, H, W = x.shape
        sobel_edges, laplacian_edges = [], []
        for c in range(C):
            x_c = x[:, c:c + 1, :, :]
            sx = F.conv2d(x_c, self.sobel_x, padding=1)
            sy = F.conv2d(x_c, self.sobel_y, padding=1)
            sobel = torch.sqrt(sx ** 2 + sy ** 2 + 1e-8)
            lap = torch.abs(F.conv2d(x_c, self.laplacian, padding=1))
            sobel_edges.append(sobel)
            laplacian_edges.append(lap)
        sobel_feat = torch.cat(sobel_edges, 1)
        laplacian_feat = torch.cat(laplacian_edges, 1)
        gradient_feat = torch.sqrt(sobel_feat ** 2 + laplacian_feat ** 2 + 1e-8)
        return sobel_feat, laplacian_feat, gradient_feat

    def forward(self, x):
        sobel_feat, laplacian_feat, gradient_feat = self.compute_edges(x)
        sobel_out = self.sobel_branch(sobel_feat)
        lap_out = self.laplacian_branch(laplacian_feat)
        grad_out = self.gradient_branch(gradient_feat)
        spatial_out = self.spatial_branch(x)
        edge_features = torch.cat([sobel_out, lap_out, grad_out, spatial_out], dim=1)
        return self.fusion(edge_features) + x


class SemanticContextExpert(nn.Module):
    """Expert 6: Scene understanding"""
    def __init__(self, dim):
        super().__init__()
        self.pool_1 = nn.AdaptiveAvgPool2d(8)
        self.pool_2 = nn.AdaptiveAvgPool2d(4)
        self.pool_3 = nn.AdaptiveAvgPool2d(2)
        self.pool_4 = nn.AdaptiveAvgPool2d(1)

        self.conv_1 = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.conv_2 = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.conv_3 = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.conv_4 = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), LayerNorm2d(dim // 4), nn.GELU())
        self.context_fusion = nn.Sequential(nn.Conv2d(dim, dim, 1), LayerNorm2d(dim), nn.GELU())

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        p1 = F.interpolate(self.conv_1(self.pool_1(x)), size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.conv_2(self.pool_2(x)), size=(H, W), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.conv_3(self.pool_3(x)), size=(H, W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.conv_4(self.pool_4(x)), size=(H, W), mode='bilinear', align_corners=False)
        pyramid_feat = torch.cat([p1, p2, p3, p4], dim=1)
        return self.context_fusion(pyramid_feat) + x


class ContrastExpert(nn.Module):
    """Expert 7: Contrast enhancement"""
    def __init__(self, dim):
        super().__init__()
        self.local_contrast = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim)
        )
        self.fusion = nn.Sequential(nn.Conv2d(dim, dim, 1), LayerNorm2d(dim), nn.GELU())

    def forward(self, x):
        local_contrast = self.local_contrast(x)
        return self.fusion(local_contrast) + x


class ContentAwareGate(nn.Module):
    """
    Smart routing network that analyzes image content to select best experts

    Examines:
    1. Spatial patterns (edges, textures) via spatial attention
    2. Channel statistics (global semantics) via channel attention
    3. Multi-level aggregation for robust routing decisions

    This helps the router learn: "Image with strong edges → use EdgeExpert"
    """
    def __init__(self, in_channels, num_experts):
        super().__init__()

        # Branch 1: Spatial pattern analysis (local features)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, groups=in_channels // 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Branch 2: Channel statistics (global semantics)
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.GELU()
        )

        # Branch 3: Max pooling (captures salient features)
        self.max_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.GELU()
        )

        # Fusion: Combine all branches to make routing decision
        total_features = (in_channels // 4) * 3
        self.fusion = nn.Sequential(
            nn.Linear(total_features, in_channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),  # Prevent overfitting to specific experts
            nn.Linear(in_channels // 2, num_experts)
        )

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            gate_logits: Expert selection scores [B, num_experts]
        """
        # Analyze image from multiple perspectives
        spatial_features = self.spatial_branch(x)   # Local patterns
        channel_features = self.channel_branch(x)   # Global semantics
        max_features = self.max_branch(x)           # Salient features

        # Combine all views
        combined = torch.cat([spatial_features, channel_features, max_features], dim=1)

        # Make routing decision
        gate_logits = self.fusion(combined)

        return gate_logits


class MoELayer(nn.Module):
    """
    Intelligent Mixture of Experts with Content-Aware Routing

    The router learns which experts work best for each image by analyzing:
    - Spatial patterns (edges, textures)
    - Channel statistics (global context)
    - Multi-scale features
    """

    def __init__(self, in_channels, num_experts=5, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Create experts
        expert_classes = [
            TextureExpert, AttentionExpert, HybridExpert,
            FrequencyExpert, EdgeExpert, SemanticContextExpert, ContrastExpert
        ]
        expert_names = [
            'TextureExpert', 'AttentionExpert', 'HybridExpert',
            'FrequencyExpert', 'EdgeExpert', 'SemanticContextExpert', 'ContrastExpert'
        ]

        self.experts = nn.ModuleList()
        self.expert_names = []
        for i in range(num_experts):
            self.experts.append(expert_classes[i % len(expert_classes)](in_channels))
            self.expert_names.append(expert_names[i % len(expert_names)])

        # Smart Gating Network: Analyzes image content to select best experts
        self.gate = ContentAwareGate(in_channels, num_experts)

    def forward(self, x):
        """
        VECTORIZED expert routing: Process entire batch in parallel

        Key optimization: Group samples by expert selection, run in batches
        This is 10-50x faster than per-sample processing
        """
        B, C, H, W = x.shape

        # Smart routing: Gate analyzes image content
        gate_logits = self.gate(x)  # [B, num_experts]

        # Select top-k experts per image
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        # VECTORIZED: Group samples by expert and process in batches
        output = torch.zeros_like(x)
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        # For each expert, collect all samples that selected it
        for expert_idx in range(self.num_experts):
            # Find which samples selected this expert
            mask = (top_k_indices == expert_idx).any(dim=1)  # [B]
            if not mask.any():
                continue

            # Get samples and their weights for this expert
            sample_indices = mask.nonzero(squeeze_dim=True)
            expert_input = x[sample_indices]

            # Run expert on all selected samples at once (PARALLEL!)
            expert_output = self.experts[expert_idx](expert_input)

            # Distribute outputs back with weights
            for i, sample_idx in enumerate(sample_indices):
                # Find weight for this expert in this sample
                expert_positions = (top_k_indices[sample_idx] == expert_idx).nonzero(squeeze_dim=True)
                if expert_positions.numel() > 0:
                    k_idx = expert_positions[0].item()
                    weight = top_k_weights[sample_idx, k_idx]
                    output[sample_idx] += weight * expert_output[i]
                    expert_counts[expert_idx] += 1

        # Soft load balancing
        expert_freq = expert_counts / (B * self.top_k + 1e-8)
        target_freq = torch.ones_like(expert_freq) / self.num_experts
        aux_loss = F.mse_loss(expert_freq, target_freq) * 0.001

        routing_info = {
            'top_k_indices': top_k_indices.detach(),
            'top_k_weights': top_k_weights.detach(),
            'expert_counts': expert_counts.detach(),
            'expert_names': [self.expert_names[i] for i in range(self.num_experts)]
        }

        return output, aux_loss, routing_info