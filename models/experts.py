import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import LayerNorm2d, SDTAEncoder


class TextureExpert(nn.Module):
    """
    Expert 1: Texture-focused using multi-scale dilated convolutions.
    Best for: Surface patterns, fine-grained textures
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
    Expert 2: Attention-based for global context.
    Best for: Scene understanding, long-range dependencies
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
    Expert 3: Hybrid local-global processing.
    Best for: Partially visible objects, occlusion handling
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
    """
    Expert 4: Frequency-domain analysis.
    Best for: Subtle texture differences, periodic patterns
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        self.mid_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
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

    def dct2d(self, x):
        """Approximate 2D DCT for frequency decomposition"""
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq
        mid_freq_blur1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mid_freq_blur2 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        mid_freq = mid_freq_blur1 - mid_freq_blur2

        return low_freq, mid_freq, high_freq

    def forward(self, x):
        low_freq, mid_freq, high_freq = self.dct2d(x)

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
    """
    Expert 5: Edge and boundary detection (NEW!)
    Best for: Object boundaries, contours, sharp transitions

    Critical for COD because camouflaged object boundaries are the
    hardest part to detect. Uses multi-scale edge detection with
    Sobel, Laplacian, and Canny-inspired filters.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Sobel edge detection (horizontal and vertical)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)

        # Laplacian edge detection (all directions)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=torch.float32).view(1, 1, 3, 3)

        # Diagonal edge detection
        diag1 = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
                             dtype=torch.float32).view(1, 1, 3, 3)
        diag2 = torch.tensor([[-1, 0, 1], [0, 0, 0], [1, 0, -1]],
                             dtype=torch.float32).view(1, 1, 3, 3)

        # Register as buffers (not trainable, but part of model state)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)
        self.register_buffer('diag1', diag1)
        self.register_buffer('diag2', diag2)

        # Edge processing branches
        self.sobel_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        self.laplacian_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        self.diagonal_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        # Gradient magnitude branch
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )

        # Edge attention to emphasize important boundaries
        self.edge_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

        # Multi-scale edge fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )

    def compute_edges(self, x):
        """
        Compute multi-directional edge maps

        Returns:
            sobel_edges: Horizontal + vertical edges
            laplacian_edges: All-direction edges
            diagonal_edges: Diagonal edges
            gradient_mag: Overall gradient magnitude
        """
        B, C, H, W = x.shape

        # Apply edge filters per channel, then reduce
        edges_list = []

        # Process each channel
        for c in range(C):
            x_c = x[:, c:c + 1, :, :]

            # Sobel edges (horizontal + vertical)
            sobel_h = F.conv2d(x_c, self.sobel_x, padding=1)
            sobel_v = F.conv2d(x_c, self.sobel_y, padding=1)
            sobel_edge = torch.sqrt(sobel_h ** 2 + sobel_v ** 2 + 1e-8)

            # Laplacian edges
            lap_edge = torch.abs(F.conv2d(x_c, self.laplacian, padding=1))

            # Diagonal edges
            diag_edge1 = torch.abs(F.conv2d(x_c, self.diag1, padding=1))
            diag_edge2 = torch.abs(F.conv2d(x_c, self.diag2, padding=1))
            diag_edge = torch.maximum(diag_edge1, diag_edge2)

            edges_list.append((sobel_edge, lap_edge, diag_edge))

        # Stack and reduce across channels
        sobel_edges = torch.cat([e[0] for e in edges_list], dim=1)
        laplacian_edges = torch.cat([e[1] for e in edges_list], dim=1)
        diagonal_edges = torch.cat([e[2] for e in edges_list], dim=1)

        # Compute overall gradient magnitude
        gradient_mag = torch.sqrt(sobel_edges ** 2 + laplacian_edges ** 2 + 1e-8)

        return sobel_edges, laplacian_edges, diagonal_edges, gradient_mag

    def forward(self, x):
        # Compute multi-directional edges
        sobel_edges, laplacian_edges, diagonal_edges, gradient_mag = self.compute_edges(x)

        # Process each edge type
        sobel_feat = self.sobel_branch(sobel_edges)
        laplacian_feat = self.laplacian_branch(laplacian_edges)
        diagonal_feat = self.diagonal_branch(diagonal_edges)
        gradient_feat = self.gradient_branch(gradient_mag)

        # Combine all edge features
        edge_features = torch.cat([sobel_feat, laplacian_feat,
                                   diagonal_feat, gradient_feat], dim=1)

        # Apply edge attention
        edge_weight = self.edge_attention(edge_features)
        edge_features = edge_features * edge_weight

        # Fuse and add residual
        out = self.fusion(edge_features)
        return out + x


class ExpertRouter(nn.Module):
    """
    Top-K Expert Router (K=3)

    Instead of using all experts with weighted sum, this router:
    1. Computes routing scores for all 5 experts
    2. Selects TOP 3 experts with highest scores
    3. Only runs those 3 experts (saves computation!)
    4. Normalizes their weights to sum to 1

    Benefits:
    - 40% less computation (3/5 experts vs 5/5)
    - Forces expert specialization
    - Better for diverse inputs (each image gets custom expert combination)
    """

    def __init__(self, dim, num_experts=5, top_k=3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Routing network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),  # Prevent overfitting to specific experts
            nn.Linear(dim // 4, num_experts)
        )

        # Add noise during training for exploration
        self.noise_std = 0.1

    def forward(self, x, training=True):
        """
        Args:
            x: Input features [B, C, H, W]
            training: Whether in training mode

        Returns:
            top_k_indices: Indices of top-k experts [B, top_k]
            top_k_weights: Normalized weights for top-k experts [B, top_k]
        """
        # Get routing logits
        logits = self.gate(x)  # [B, num_experts]

        # Add noise during training for exploration
        if training and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Normalize top-k weights (softmax only over top-k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_indices, top_k_weights


class MoELayer(nn.Module):
    """
    Sparse Mixture of 5 Experts with Top-3 Selection

    Experts:
    1. TextureExpert: Multi-scale texture patterns
    2. AttentionExpert: Global context via self-attention
    3. HybridExpert: Local-global feature fusion
    4. FrequencyExpert: Frequency-domain analysis
    5. EdgeExpert: Boundary and edge detection (NEW!)

    Key Innovation: Top-K Routing (K=3)
    - Each input only uses its 3 most relevant experts
    - Different images use different expert combinations
    - 40% computation savings vs using all 5 experts
    - Better specialization and generalization
    """

    def __init__(self, dim, num_experts=5, top_k=3):
        super().__init__()

        # 5 Specialized Experts
        self.experts = nn.ModuleList([
            TextureExpert(dim),  # Expert 1
            AttentionExpert(dim),  # Expert 2
            HybridExpert(dim),  # Expert 3
            FrequencyExpert(dim),  # Expert 4
            EdgeExpert(dim)  # Expert 5 (NEW!)
        ])

        self.router = ExpertRouter(dim, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k

        print(f"MoELayer initialized with {num_experts} experts, Top-{top_k} routing:")
        print("  1. TextureExpert (multi-scale dilated convolutions)")
        print("  2. AttentionExpert (self-attention)")
        print("  3. HybridExpert (local-global fusion)")
        print("  4. FrequencyExpert (frequency-domain analysis)")
        print("  5. EdgeExpert (boundary detection) ← NEW!")
        print(f"  → Computation: {top_k}/{num_experts} experts per input ({top_k / num_experts * 100:.0f}%)")

    def forward(self, x):
        """
        Forward pass with Top-K expert selection

        Args:
            x: Input features [B, C, H, W]

        Returns:
            output: Weighted combination of top-k expert outputs [B, C, H, W]
            aux_loss: Load balancing auxiliary loss (scalar)
            routing_info: Dictionary with routing statistics (for analysis)
        """
        B = x.size(0)

        # Get top-k expert indices and weights for each sample in batch
        top_k_indices, top_k_weights = self.router(x, training=self.training)
        # top_k_indices: [B, top_k], top_k_weights: [B, top_k]

        # Initialize output
        output = torch.zeros_like(x)

        # Process each sample in the batch
        for b in range(B):
            sample_output = torch.zeros_like(x[b:b + 1])

            # Run only the top-k experts for this sample
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k].item()
                expert_weight = top_k_weights[b, k]

                # Run the selected expert
                expert_out = self.experts[expert_idx](x[b:b + 1])

                # Add weighted expert output
                sample_output = sample_output + expert_weight * expert_out

            output[b:b + 1] = sample_output

        # Calculate auxiliary loss (load balancing)
        aux_loss = self.calculate_aux_loss(top_k_indices)

        # Gather routing info for analysis
        routing_info = {
            'top_k_indices': top_k_indices.detach(),
            'top_k_weights': top_k_weights.detach()
        }

        return output, aux_loss, routing_info

    def calculate_aux_loss(self, top_k_indices):
        """
        Load balancing loss to encourage even expert usage.

        Without this, the model might only use 2-3 experts and ignore others.
        We want all 5 experts to be useful for different inputs.
        """
        B = top_k_indices.size(0)

        # Count how many times each expert was selected
        expert_counts = torch.zeros(self.num_experts, device=top_k_indices.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (top_k_indices == expert_idx).sum().float()

        # Normalize to get usage frequency
        expert_freq = expert_counts / (B * self.top_k)

        # Target: each expert should be used equally (1/num_experts)
        target_freq = torch.ones_like(expert_freq) / self.num_experts

        # MSE loss between actual and target frequency
        aux_loss = F.mse_loss(expert_freq, target_freq) * 0.01

        return aux_loss


# Utility functions for analysis
@torch.no_grad()
def analyze_expert_routing(moe_layer, x):
    """
    Analyze which experts are being used for given inputs

    Args:
        moe_layer: MoELayer instance
        x: Input tensor [B, C, H, W]

    Returns:
        routing_stats: Dictionary with detailed routing statistics
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

    expert_names = ['Texture', 'Attention', 'Hybrid', 'Frequency', 'Edge']

    print("\n" + "=" * 70)
    print("EXPERT ROUTING ANALYSIS (Top-3 Selection)")
    print("=" * 70)
    print(f"\nBatch size: {B}")
    print(f"Top-K: {moe_layer.top_k}/{moe_layer.num_experts} experts per sample")
    print(f"\nExpert Usage Frequency:")
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

    Args:
        moe_layer: MoELayer instance
        x: Input tensor [B, C, H, W]
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

    expert_names = ['Texture', 'Attention', 'Hybrid', 'Frequency', 'Edge']

    print("\n" + "=" * 70)
    print("TOP EXPERT COMBINATIONS")
    print("=" * 70)

    for combo, count in combo_counts.most_common(10):
        combo_names = [expert_names[i] for i in combo]
        freq = count / B * 100
        print(f"  {combo_names}: {count}/{B} ({freq:.1f}%)")

    print("=" * 70 + "\n")