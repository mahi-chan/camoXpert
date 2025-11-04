"""
100% COD-Specialized Modules
Based on SOTA COD research (SINet, PraNet, ZoomNet, UGTR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryUncertaintyModule(nn.Module):
    """
    Estimates uncertainty at object boundaries
    Camouflaged boundaries are inherently ambiguous - model should know when uncertain
    """
    def __init__(self, dim):
        super().__init__()
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Predict mean (main prediction)
        self.mean_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1)
        )

        # Predict uncertainty (how confident we are)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        mean = self.mean_head(shared_feat)
        uncertainty = self.uncertainty_head(shared_feat)
        return mean, uncertainty


class SearchIdentificationModule(nn.Module):
    """
    Mimics visual search process for camouflaged objects (from SINet)
    1. Search: Where might objects be hiding?
    2. Identification: What are the object features?
    """
    def __init__(self, dim):
        super().__init__()
        # Search branch: Generate search map (where to look)
        self.search_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()  # Search confidence map [0, 1]
        )

        # Identification branch: Enhanced features at search locations
        self.identify_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),  # Pointwise
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        search_map = self.search_conv(x)  # Where are objects likely?
        features = self.identify_conv(x)  # What do they look like?
        # Focus identification on high-search areas
        searched_features = features * search_map
        return searched_features, search_map


class ReverseAttentionModule(nn.Module):
    """
    Learns to suppress background by modeling what is NOT the object (from PraNet)
    Key: Once you know what's NOT the object, what remains must be the object
    """
    def __init__(self, dim):
        super().__init__()
        # Predict background (inverse of foreground)
        self.background_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

        # Refine features by removing background
        self.refine = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Predict background probability
        bg_map = self.background_pred(x)

        # Foreground = not background
        fg_map = 1 - bg_map

        # Concatenate features with background map
        x_with_bg = torch.cat([x, bg_map], dim=1)
        refined = self.refine(x_with_bg)

        # Apply foreground attention
        output = refined * fg_map

        return output, fg_map


class ContrastEnhancementModule(nn.Module):
    """
    Enhances subtle differences between foreground and background
    Uses multi-scale contrast kernels specifically for COD
    """
    def __init__(self, dim):
        super().__init__()
        # Multi-scale contrast detection (depthwise separable for efficiency)
        self.contrast_3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim // 3, 1)
        )
        self.contrast_5x5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.Conv2d(dim, dim // 3, 1)
        )
        self.contrast_7x7 = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.Conv2d(dim, dim // 3, 1)
        )

        # Contrast fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Contrast amplification (learned)
        self.amplify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Detect contrast at multiple scales
        c3 = self.contrast_3x3(x)
        c5 = self.contrast_5x5(x)
        c7 = self.contrast_7x7(x)

        # Fuse multi-scale contrasts
        contrast = self.fusion(torch.cat([c3, c5, c7], dim=1))

        # Amplify contrast (channel-wise attention)
        contrast_weight = self.amplify(contrast)
        enhanced = x + contrast * contrast_weight

        return enhanced


class IterativeBoundaryRefinement(nn.Module):
    """
    Iteratively refine boundaries using uncertainty feedback
    Focus computation on uncertain regions (boundaries)
    """
    def __init__(self, dim, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations

        self.refinement_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim + 2, dim, 3, padding=1),  # +2 for pred + uncertainty
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_iterations)
        ])

        self.pred_heads = nn.ModuleList([
            nn.Conv2d(dim, 1, 1) for _ in range(num_iterations)
        ])

    def forward(self, features, initial_pred, initial_uncertainty):
        """
        Args:
            features: [B, C, H, W]
            initial_pred: [B, 1, H, W] logits
            initial_uncertainty: [B, 1, H, W] uncertainty map
        Returns:
            refinements: List of refined predictions
        """
        pred = initial_pred
        uncertainty = initial_uncertainty

        refinements = []

        for i in range(self.num_iterations):
            # Normalize uncertainty for attention
            uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)

            # Concatenate features, prediction, and uncertainty
            x = torch.cat([features, pred, uncertainty_norm], dim=1)

            # Refine features
            refined_features = self.refinement_blocks[i](x)

            # Apply attention (focus on boundaries = high uncertainty)
            refined_features = refined_features * (1 + uncertainty_norm)

            # New prediction
            pred = self.pred_heads[i](refined_features)

            refinements.append(pred)

        return refinements


class CODTextureExpert(nn.Module):
    """
    COD-optimized texture expert with adaptive dilation
    Detects multi-scale texture patterns critical for camouflage
    """
    def __init__(self, dim):
        super().__init__()
        # Adaptive dilations for texture at different scales
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=8, dilation=8),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        multi_scale = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fusion(multi_scale) + x


class CODFrequencyExpert(nn.Module):
    """
    COD-optimized frequency domain analysis
    Separates texture frequencies to detect camouflaged patterns
    """
    def __init__(self, dim):
        super().__init__()
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.mid_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Frequency separation via averaging
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq

        # Mid frequency = difference of Gaussians
        mid_freq_blur1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mid_freq_blur2 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        mid_freq = mid_freq_blur1 - mid_freq_blur2

        low_feat = self.low_freq_conv(low_freq)
        mid_feat = self.mid_freq_conv(mid_freq)
        high_feat = self.high_freq_conv(high_freq)
        spatial_feat = self.spatial_conv(x)

        freq_features = torch.cat([low_feat, mid_feat, high_feat, spatial_feat], dim=1)
        return self.fusion(freq_features) + x


class CODEdgeExpert(nn.Module):
    """
    COD-optimized edge detection
    Detects subtle boundaries that camouflaged objects try to hide
    """
    def __init__(self, dim):
        super().__init__()
        # Sobel and Laplacian kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x_base', sobel_x)
        self.register_buffer('sobel_y_base', sobel_y)
        self.register_buffer('laplacian_base', laplacian)
        self.dim = dim

        self.sobel_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.laplacian_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def compute_edges(self, x):
        B, C, H, W = x.shape
        device = x.device  # Get input device for DataParallel compatibility

        # Move buffers to input device before repeat
        sobel_x = self.sobel_x_base.to(device).repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y_base.to(device).repeat(C, 1, 1, 1)
        laplacian = self.laplacian_base.to(device).repeat(C, 1, 1, 1)

        # Make outputs contiguous to prevent misaligned address errors with DataParallel
        sx = F.conv2d(x, sobel_x, padding=1, groups=C).contiguous()
        sy = F.conv2d(x, sobel_y, padding=1, groups=C).contiguous()
        lap = F.conv2d(x, laplacian, padding=1, groups=C).contiguous()

        sobel_feat = torch.sqrt(sx ** 2 + sy ** 2 + 1e-8)
        laplacian_feat = torch.abs(lap)
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
