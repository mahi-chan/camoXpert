# cod_metrics.py

import torch

class CODMetrics:
    """
    Comprehensive metrics for camouflaged object detection evaluation.

    Implements:
    - MAE (Mean Absolute Error)
    - F-measure (Weighted F-score)
    - S-measure (Structure measure)
    - E-measure (Enhanced alignment measure)
    - IoU (Intersection over Union)
    """

    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return torch.mean(torch.abs(pred - target)).item()

    @staticmethod
    def f_measure(pred, target, beta=0.3):
        """
        Weighted F-measure.

        F_β = (1 + β²) * Precision * Recall / (β² * Precision + Recall)
        """
        pred_binary = (pred > 0.5).float()

        tp = torch.sum(pred_binary * target)
        fp = torch.sum(pred_binary * (1 - target))
        fn = torch.sum((1 - pred_binary) * target)

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        beta_sq = beta ** 2
        f_score = ((1 + beta_sq) * precision * recall) / \
                  (beta_sq * precision + recall + 1e-10)

        return f_score.item()

    @staticmethod
    def s_measure(pred, target, alpha=0.5):
        """
        Structure measure combining region and object similarity.

        S = α * S_r + (1 - α) * S_o
        """
        # Region similarity
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        correlation = torch.sum(pred_centered * target_centered) / \
                      (torch.sqrt(torch.sum(pred_centered ** 2) *
                                  torch.sum(target_centered ** 2)) + 1e-10)

        s_r = (correlation + 1) / 2

        # Object similarity
        intersection = torch.min(pred, target)
        s_o = torch.mean(intersection) / (torch.mean(torch.max(pred, target)) + 1e-10)

        return (alpha * s_r + (1 - alpha) * s_o).item()

    @staticmethod
    def e_measure(pred, target):
        """
        Enhanced alignment measure for boundary quality.
        """
        # Alignment matrix
        align_matrix = 2 * pred * target / (pred ** 2 + target ** 2 + 1e-10)

        # Enhanced mean
        e_measure = torch.mean(align_matrix).item()

        return e_measure

    @staticmethod
    def iou(pred, target, threshold=0.5):
        """
        Intersection over Union (Jaccard Index).
        """
        pred_binary = (pred > threshold).float()

        intersection = torch.sum(pred_binary * target)
        union = torch.sum(pred_binary) + torch.sum(target) - intersection

        iou = intersection / (union + 1e-10)

        return iou.item()

    @classmethod
    def compute_all(cls, pred, target):
        """
        Compute all metrics at once.

        Returns:
            dict: Dictionary containing all metric values
        """
        return {
            'MAE': cls.mae(pred, target),
            'F-measure': cls.f_measure(pred, target),
            'S-measure': cls.s_measure(pred, target),
            'E-measure': cls.e_measure(pred, target),
            'IoU': cls.iou(pred, target)
        }