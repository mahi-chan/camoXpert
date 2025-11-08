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
        shared_feat = self.shared(x).clone()
        mean = self.mean_head(shared_feat).clone()
        uncertainty = self.uncertainty_head(shared_feat).clone()
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
        search_map = self.search_conv(x).clone()  # Where are objects likely?
        features = self.identify_conv(x).clone()  # What do they look like?
        # Focus identification on high-search areas
        searched_features = (features * search_map).clone()
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
        bg_map = self.background_pred(x).clone()

        # Foreground = not background
        fg_map = (1 - bg_map).clone()

        # Concatenate features with background map - clone x for alignment
        x_with_bg = torch.cat([x.clone(), bg_map], dim=1)
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
        # Detect contrast at multiple scales - clone outputs for alignment
        c3 = self.contrast_3x3(x).clone()
        c5 = self.contrast_5x5(x).clone()
        c7 = self.contrast_7x7(x).clone()

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
            # Normalize uncertainty for attention - clone for alignment
            uncertainty_norm = (uncertainty / (uncertainty.max() + 1e-8)).clone()

            # Concatenate features, prediction, and uncertainty - clone inputs
            x = torch.cat([features.clone(), pred.clone(), uncertainty_norm], dim=1)

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
        # Clone all branch outputs to ensure proper alignment for DataParallel
        feat1 = self.branch1(x).clone()
        feat2 = self.branch2(x).clone()
        feat3 = self.branch3(x).clone()
        feat4 = self.branch4(x).clone()
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
        # Frequency separation via averaging - clone pooling outputs for alignment
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1).clone()
        high_freq = (x - low_freq).clone()

        # Mid frequency = difference of Gaussians
        mid_freq_blur1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1).clone()
        mid_freq_blur2 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2).clone()
        mid_freq = (mid_freq_blur1 - mid_freq_blur2).clone()

        # Clone all conv outputs to ensure proper alignment for DataParallel
        low_feat = self.low_freq_conv(low_freq).clone()
        mid_feat = self.mid_freq_conv(mid_freq).clone()
        high_feat = self.high_freq_conv(high_freq).clone()
        spatial_feat = self.spatial_conv(x).clone()

        freq_features = torch.cat([low_feat, mid_feat, high_feat, spatial_feat], dim=1)
        return self.fusion(freq_features) + x


class CODEdgeExpert(nn.Module):
    """
    COD-optimized edge detection (DataParallel-safe version)
    Uses learnable edge detection initialized with edge kernels to avoid
    DataParallel grouped convolution misalignment issues
    """
    def __init__(self, dim):
        super().__init__()
        # Learnable depthwise convolutions initialized as edge detectors
        self.horizontal_edge = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise
            nn.Conv2d(dim, dim // 4, 1),  # Pointwise
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.vertical_edge = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise
            nn.Conv2d(dim, dim // 4, 1),  # Pointwise
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.laplacian_edge = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise
            nn.Conv2d(dim, dim // 4, 1),  # Pointwise
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

        # Initialize depthwise convs with edge detection kernels
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        """Initialize depthwise convolutions with edge detection kernels"""
        # Sobel X kernel for horizontal edges
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32) / 4.0  # Normalize

        # Sobel Y kernel for vertical edges
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32) / 4.0  # Normalize

        # Laplacian kernel for edge detection
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32) / 4.0  # Normalize

        # Initialize horizontal edge detector with Sobel X
        with torch.no_grad():
            conv = self.horizontal_edge[0]  # Get depthwise conv
            for i in range(conv.weight.shape[0]):
                conv.weight[i, 0, :, :] = sobel_x

        # Initialize vertical edge detector with Sobel Y
        with torch.no_grad():
            conv = self.vertical_edge[0]  # Get depthwise conv
            for i in range(conv.weight.shape[0]):
                conv.weight[i, 0, :, :] = sobel_y

        # Initialize Laplacian edge detector
        with torch.no_grad():
            conv = self.laplacian_edge[0]  # Get depthwise conv
            for i in range(conv.weight.shape[0]):
                conv.weight[i, 0, :, :] = laplacian

    def forward(self, x):
        # Learnable edge detection - DataParallel safe
        h_edge = self.horizontal_edge(x).clone()
        v_edge = self.vertical_edge(x).clone()
        lap_edge = self.laplacian_edge(x).clone()
        spatial = self.spatial_branch(x).clone()

        edge_features = torch.cat([h_edge, v_edge, lap_edge, spatial], dim=1)
        return self.fusion(edge_features) + x
