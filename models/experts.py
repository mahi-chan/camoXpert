"""
CamoXpert Expert Modules - 7 Experts with Top-4 Selection
PRODUCTION READY - Runs without errors
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
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=2, dilation=2),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=3, dilation=3),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=4, dilation=4),
            LayerNorm2d(dim // 4),  # FIXED
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
    Expert 2: Global context via self-attention
    Uses SDTA for long-range dependencies
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = SDTAEncoder(dim=dim, num_heads=num_heads, drop_path=0.1)

    def forward(self, x):
        return self.attention(x)


class HybridExpert(nn.Module):
    """
    Expert 3: Local-global feature fusion
    Combines depthwise convolution with global pooling
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


class FrequencyExpert(nn.Module):
    """Expert 4: Frequency-domain analysis"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.mid_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )

        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
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
        freq_weight = self.freq_attention(freq_features)
        freq_features = freq_features * freq_weight

        out = self.fusion(freq_features)
        return out + x


class EdgeExpert(nn.Module):
    """Expert 5: Boundary and edge detection"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

        self.sobel_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.laplacian_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )

        self.edge_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )

    def compute_edges(self, x):
        B, C, H, W = x.shape
        sobel_edges = []
        laplacian_edges = []

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
        edge_weight = self.edge_attention(edge_features)
        edge_features = edge_features * edge_weight

        out = self.fusion(edge_features)
        return out + x

class SemanticContextExpert(nn.Module):
    """
    Expert 6: High-level semantic scene understanding
    Based on Pyramid Pooling Module (PSPNet) and ASPP (DeepLab)
    Captures multi-scale context for scene-level understanding

    FIX: Uses LayerNorm2d instead of BatchNorm2d to support batch_size=1
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Multi-scale pyramid pooling (PSPNet-style)
        self.pool_1 = nn.AdaptiveAvgPool2d(8)  # 8x8 regions
        self.pool_2 = nn.AdaptiveAvgPool2d(4)  # 4x4 regions
        self.pool_3 = nn.AdaptiveAvgPool2d(2)  # 2x2 regions
        self.pool_4 = nn.AdaptiveAvgPool2d(1)  # Global pooling

        # Process each scale - FIXED: LayerNorm2d instead of BatchNorm2d
        self.conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4),  # FIXED
            nn.GELU()
        )

        # Context aggregation - FIXED
        self.context_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),  # FIXED
            nn.GELU()
        )

        # Semantic attention
        self.semantic_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Multi-scale pooling
        p1 = self.conv_1(self.pool_1(x))
        p2 = self.conv_2(self.pool_2(x))
        p3 = self.conv_3(self.pool_3(x))
        p4 = self.conv_4(self.pool_4(x))

        # Upsample all to original size
        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=False)

        # Concatenate multi-scale features
        pyramid_feat = torch.cat([p1, p2, p3, p4], dim=1)

        # Fuse context
        context_feat = self.context_fusion(pyramid_feat)

        # Apply semantic attention
        attention = self.semantic_attention(context_feat)
        context_feat = context_feat * attention

        return context_feat + x


class ContrastExpert(nn.Module):
    """Expert 7: Contrast enhancement for low-visibility regions"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.local_contrast = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            LayerNorm2d(dim),  # FIXED
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim)  # FIXED
        )

        self.local_mean_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            LayerNorm2d(dim)  # FIXED
        )

        self.local_std_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            LayerNorm2d(dim)  # FIXED
        )

        self.diff_enhance = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            LayerNorm2d(dim),  # FIXED
            nn.GELU()
        )

        self.contrast_boost = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.visibility_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            LayerNorm2d(dim),  # FIXED
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            LayerNorm2d(dim)  # FIXED
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        local_mean = self.local_mean_conv(x)
        local_std = self.local_std_conv(torch.abs(x - local_mean))
        local_contrast = self.local_contrast(x)
        diff = torch.abs(x - local_mean)

        contrast_feat = self.diff_enhance(torch.cat([local_contrast, diff], dim=1))
        boost_weight = self.contrast_boost(contrast_feat)
        contrast_feat = contrast_feat * boost_weight

        visibility_feat = self.visibility_enhance(contrast_feat)
        normalized_feat = visibility_feat / (local_std + 1e-6)
        out = self.fusion(normalized_feat)

        return out + x

class ExpertRouter(nn.Module):
    """
    Adaptive Expert Router with Top-K Selection
    Selects K most relevant experts for each input
    """

    def __init__(self, dim, num_experts=7, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 4, num_experts)
        )

        # Noise for exploration during training
        self.noise_std = 0.1

    def forward(self, x, training=True):
        """
        Select top-k experts based on input features

        Args:
            x: Input features [B, C, H, W]
            training: Whether in training mode

        Returns:
            top_k_indices: Selected expert indices [B, top_k]
            top_k_weights: Normalized weights [B, top_k]
        """
        # Compute routing logits
        logits = self.gate(x)  # [B, num_experts]

        # Add exploration noise during training
        if training and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Normalize weights via softmax
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_indices, top_k_weights


class MoELayer(nn.Module):
    """
    Mixture of 7 Experts with Top-4 Selection

    Experts:
    1. TextureExpert - Multi-scale texture patterns
    2. AttentionExpert - Global context via self-attention
    3. HybridExpert - Local-global fusion
    4. FrequencyExpert - Frequency-domain analysis
    5. EdgeExpert - Boundary detection
    6. SemanticContextExpert - Scene understanding (PSPNet/ASPP)
    7. ContrastExpert - Visibility enhancement (FROM PROPOSAL)

    Each input selects 4 most relevant experts (57% of total experts)
    """

    def __init__(self, dim, num_experts=7, top_k=4):
        super().__init__()

        # 7 Specialized Experts
        self.experts = nn.ModuleList([
            TextureExpert(dim),  # Expert 1
            AttentionExpert(dim),  # Expert 2
            HybridExpert(dim),  # Expert 3
            FrequencyExpert(dim),  # Expert 4
            EdgeExpert(dim),  # Expert 5
            SemanticContextExpert(dim),  # Expert 6
            ContrastExpert(dim)  # Expert 7 (FROM PROPOSAL)
        ])

        self.router = ExpertRouter(dim, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k

        print(f"✓ MoELayer initialized with {num_experts} experts, Top-{top_k} routing:")
        print("  1. TextureExpert (multi-scale dilated convolutions)")
        print("  2. AttentionExpert (SDTA self-attention)")
        print("  3. HybridExpert (local-global fusion)")
        print("  4. FrequencyExpert (frequency-domain analysis)")
        print("  5. EdgeExpert (boundary detection)")
        print("  6. SemanticContextExpert (pyramid pooling)")
        print("  7. ContrastExpert (visibility enhancement) ← FROM PROPOSAL")
        print(f"  → Computation: {top_k}/{num_experts} experts per input ({top_k / num_experts * 100:.1f}%)")

    def forward(self, x):
        """
        Forward pass with Top-K expert selection

        Args:
            x: Input features [B, C, H, W]

        Returns:
            output: Weighted combination of top-k expert outputs [B, C, H, W]
            aux_loss: Load balancing loss (scalar)
            routing_info: Dictionary with routing statistics
        """
        B = x.size(0)

        # Get top-k expert selection for each sample
        top_k_indices, top_k_weights = self.router(x, training=self.training)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each sample with its selected experts
        for b in range(B):
            sample_output = torch.zeros_like(x[b:b + 1])

            # Run only the top-k experts for this sample
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k].item()
                expert_weight = top_k_weights[b, k]

                # Execute selected expert
                expert_out = self.experts[expert_idx](x[b:b + 1])

                # Add weighted contribution
                sample_output = sample_output + expert_weight * expert_out

            output[b:b + 1] = sample_output

        # Load balancing loss (encourages even expert usage)
        aux_loss = self._compute_load_balance_loss(top_k_indices)

        # Routing statistics
        routing_info = {
            'top_k_indices': top_k_indices.detach(),
            'top_k_weights': top_k_weights.detach()
        }

        return output, aux_loss, routing_info

    def _compute_load_balance_loss(self, top_k_indices):
        """
        Load balancing loss to encourage even expert usage
        Prevents router from always selecting same experts
        """
        B = top_k_indices.size(0)

        # Count how many times each expert was selected
        expert_counts = torch.zeros(self.num_experts, device=top_k_indices.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (top_k_indices == expert_idx).sum().float()

        # Normalize to get usage frequency
        expert_freq = expert_counts / (B * self.top_k)

        # Target: uniform distribution (each expert used equally)
        target_freq = torch.ones_like(expert_freq) / self.num_experts

        # MSE loss between actual and target frequency
        aux_loss = F.mse_loss(expert_freq, target_freq) * 0.01

        return aux_loss


# Utility functions for analysis
@torch.no_grad()
def analyze_expert_routing(moe_layer, x):
    """
    Analyze which experts are being used for given inputs
    """
    B = x.size(0)
    top_k_indices, top_k_weights = moe_layer.router(x, training=False)

    # Count expert usage
    expert_usage = torch.zeros(moe_layer.num_experts)
    for expert_idx in range(moe_layer.num_experts):
        expert_usage[expert_idx] = (top_k_indices == expert_idx).sum().float() / (B * moe_layer.top_k)

    # Average weights per expert
    expert_avg_weights = torch.zeros(moe_layer.num_experts)
    for expert_idx in range(moe_layer.num_experts):
        mask = (top_k_indices == expert_idx)
        if mask.sum() > 0:
            expert_avg_weights[expert_idx] = top_k_weights[mask].mean()

    expert_names = ['Texture', 'Attention', 'Hybrid', 'Frequency', 'Edge', 'Semantic', 'Contrast']

    print("\n" + "=" * 70)
    print(f"EXPERT ROUTING ANALYSIS ({moe_layer.num_experts} experts, Top-{moe_layer.top_k})")
    print("=" * 70)
    print(f"Batch size: {B}")
    print(f"Selection: Top-{moe_layer.top_k}/{moe_layer.num_experts} experts per sample\n")
    print("Expert Usage Frequency:")
    print("-" * 70)

    for i, (name, usage, avg_weight) in enumerate(zip(expert_names, expert_usage, expert_avg_weights)):
        bar = '█' * int(usage * 50)
        print(f"  {i + 1}. {name:12s}: {usage:.3f} {bar}")
        if avg_weight > 0:
            print(f"      Avg weight when selected: {avg_weight:.3f}")

    print("-" * 70)
    print(f"Coverage: {(expert_usage > 0).sum().item()}/{moe_layer.num_experts} experts used")
    print("=" * 70 + "\n")

    return {
        'expert_usage': expert_usage.cpu().numpy(),
        'expert_avg_weights': expert_avg_weights.cpu().numpy(),
        'top_k_indices': top_k_indices.cpu().numpy(),
        'top_k_weights': top_k_weights.cpu().numpy()
    }


@torch.no_grad()
def visualize_expert_combinations(moe_layer, x):
    """
    Visualize which expert combinations are most common
    """
    top_k_indices, _ = moe_layer.router(x, training=False)
    B = top_k_indices.size(0)

    # Convert to tuples for counting
    combinations = []
    for b in range(B):
        combo = tuple(sorted(top_k_indices[b].cpu().tolist()))
        combinations.append(combo)

    # Count unique combinations
    from collections import Counter
    combo_counts = Counter(combinations)

    expert_names = ['Texture', 'Attention', 'Hybrid', 'Frequency', 'Edge', 'Semantic', 'Contrast']

    print("\n" + "=" * 70)
    print("TOP EXPERT COMBINATIONS")
    print("=" * 70)

    for combo, count in combo_counts.most_common(10):
        combo_names = [expert_names[i] for i in combo]
        freq = count / B * 100
        print(f"  {combo_names}")
        print(f"    Count: {count}/{B} ({freq:.1f}%)")

    print("=" * 70 + "\n")