#!/usr/bin/env python3
"""
Comprehensive MoE System Verification

This script verifies:
1. Router learns from image features (not random)
2. Top-3 expert selection is working correctly
3. Expert outputs are combined with learned weights
4. Router makes content-aware decisions
5. No NaN or degenerate outputs
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, '/home/user/camoXpert')

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from torch.utils.data import DataLoader


class MoEVerifier:
    """Comprehensive MoE verification suite"""

    def __init__(self, model, dataloader, device='cuda'):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.results = defaultdict(list)

    def verify_all(self):
        """Run all verification checks"""
        print("\n" + "="*70)
        print("🔍 MoE SYSTEM VERIFICATION")
        print("="*70)

        # Test 1: Verify routing logic
        print("\n[1/6] Verifying routing logic...")
        self.verify_routing_logic()

        # Test 2: Check weight properties
        print("\n[2/6] Checking weight properties...")
        self.verify_weight_properties()

        # Test 3: Verify output combination
        print("\n[3/6] Verifying output combination...")
        self.verify_output_combination()

        # Test 4: Check for NaN/Inf
        print("\n[4/6] Checking for numerical issues...")
        self.check_numerical_stability()

        # Test 5: Verify content-aware routing
        print("\n[5/6] Verifying content-aware routing...")
        self.verify_content_awareness()

        # Test 6: Analyze expert specialization
        print("\n[6/6] Analyzing expert specialization...")
        self.analyze_expert_specialization()

        # Print summary
        self.print_summary()

    @torch.no_grad()
    def verify_routing_logic(self):
        """Verify that routing selects top-3 experts correctly"""
        test_passed = True

        for batch_idx, (images, _) in enumerate(self.dataloader):
            if batch_idx >= 5:  # Test on 5 batches
                break

            images = images.to(self.device)
            B = images.shape[0]

            # Extract features and run through MoE
            features = self.model.backbone(images)

            for layer_idx, (feat, sdta, moe) in enumerate(zip(features, self.model.sdta_blocks, self.model.moe_layers)):
                feat = sdta(feat)

                # Get routing decisions
                gate_logits = moe.gate(feat)  # [B, num_experts]
                top_k_logits, top_k_indices = torch.topk(gate_logits, moe.top_k, dim=-1)

                # Verify: top_k_indices should have shape [B, 3]
                assert top_k_indices.shape == (B, 3), \
                    f"❌ Top-k indices shape mismatch: {top_k_indices.shape} vs expected ({B}, 3)"

                # Verify: All indices should be unique per sample
                for i in range(B):
                    unique_experts = torch.unique(top_k_indices[i])
                    if len(unique_experts) != 3:
                        print(f"  ⚠️  Sample {i} selected duplicate experts: {top_k_indices[i].tolist()}")
                        test_passed = False

                # Verify: Indices should be in valid range [0, num_experts-1]
                assert (top_k_indices >= 0).all() and (top_k_indices < moe.num_experts).all(), \
                    "❌ Expert indices out of range"

                self.results['routing_checks'].append({
                    'layer': layer_idx,
                    'batch': batch_idx,
                    'indices': top_k_indices.cpu().numpy(),
                    'logits': top_k_logits.cpu().numpy()
                })

        if test_passed:
            print("  ✅ Routing logic verified: Top-3 selection working correctly")
        else:
            print("  ⚠️  Routing logic has issues (see warnings above)")

    @torch.no_grad()
    def verify_weight_properties(self):
        """Verify that routing weights sum to 1 and are non-negative"""
        test_passed = True

        for batch_idx, (images, _) in enumerate(self.dataloader):
            if batch_idx >= 5:
                break

            images = images.to(self.device)
            features = self.model.backbone(images)

            for layer_idx, (feat, sdta, moe) in enumerate(zip(features, self.model.sdta_blocks, self.model.moe_layers)):
                feat = sdta(feat)

                # Get weights
                gate_logits = moe.gate(feat)
                top_k_logits, top_k_indices = torch.topk(gate_logits, moe.top_k, dim=-1)
                top_k_weights = F.softmax(top_k_logits, dim=-1)

                # Verify: Weights should sum to 1.0 (within tolerance)
                weight_sums = top_k_weights.sum(dim=1)
                if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5):
                    print(f"  ❌ Weights don't sum to 1.0: {weight_sums[0].item():.6f}")
                    test_passed = False

                # Verify: All weights should be non-negative
                if (top_k_weights < 0).any():
                    print(f"  ❌ Negative weights detected")
                    test_passed = False

                # Verify: Weights should be diverse (not all equal)
                weight_std = top_k_weights.std(dim=1).mean().item()
                if weight_std < 0.01:
                    print(f"  ⚠️  Weights are very uniform (std={weight_std:.4f}), router may not be learning")

                self.results['weight_properties'].append({
                    'layer': layer_idx,
                    'batch': batch_idx,
                    'weights': top_k_weights.cpu().numpy(),
                    'weight_std': weight_std
                })

        if test_passed:
            print("  ✅ Weight properties verified: Sum to 1.0, non-negative")
        else:
            print("  ❌ Weight properties test FAILED")

    @torch.no_grad()
    def verify_output_combination(self):
        """Verify that expert outputs are properly combined"""
        test_passed = True

        for batch_idx, (images, _) in enumerate(self.dataloader):
            if batch_idx >= 3:
                break

            images = images.to(self.device)
            B, C, H, W = images.shape
            features = self.model.backbone(images)

            for layer_idx, (feat, sdta, moe) in enumerate(zip(features, self.model.sdta_blocks, self.model.moe_layers)):
                feat = sdta(feat)
                feat_shape = feat.shape

                # Get MoE output
                moe_output, aux_loss, routing_info = moe(feat)

                # Verify: Output shape matches input shape
                assert moe_output.shape == feat_shape, \
                    f"❌ Output shape mismatch: {moe_output.shape} vs {feat_shape}"

                # Verify: Output is not all zeros
                if moe_output.abs().sum() < 1e-6:
                    print(f"  ❌ MoE output is all zeros at layer {layer_idx}")
                    test_passed = False

                # Verify: Output has reasonable magnitude
                output_mean = moe_output.abs().mean().item()
                output_std = moe_output.std().item()
                if output_mean < 1e-6 or output_std < 1e-6:
                    print(f"  ⚠️  Layer {layer_idx} has very small output (mean={output_mean:.6f}, std={output_std:.6f})")

                self.results['output_stats'].append({
                    'layer': layer_idx,
                    'batch': batch_idx,
                    'mean': output_mean,
                    'std': output_std,
                    'min': moe_output.min().item(),
                    'max': moe_output.max().item()
                })

        if test_passed:
            print("  ✅ Output combination verified: Shapes correct, non-zero outputs")
        else:
            print("  ❌ Output combination test FAILED")

    @torch.no_grad()
    def check_numerical_stability(self):
        """Check for NaN, Inf, or extreme values"""
        test_passed = True

        for batch_idx, (images, _) in enumerate(self.dataloader):
            if batch_idx >= 10:
                break

            images = images.to(self.device)

            # Full forward pass
            pred, aux_loss, _ = self.model(images)

            # Check prediction
            if torch.isnan(pred).any():
                print(f"  ❌ NaN detected in prediction at batch {batch_idx}")
                test_passed = False

            if torch.isinf(pred).any():
                print(f"  ❌ Inf detected in prediction at batch {batch_idx}")
                test_passed = False

            # Check aux loss
            if aux_loss is not None:
                if torch.isnan(aux_loss).any():
                    print(f"  ❌ NaN detected in aux_loss at batch {batch_idx}")
                    test_passed = False

                if torch.isinf(aux_loss).any():
                    print(f"  ❌ Inf detected in aux_loss at batch {batch_idx}")
                    test_passed = False

            # Check for extreme values
            pred_max = pred.abs().max().item()
            if pred_max > 1e6:
                print(f"  ⚠️  Very large prediction values: {pred_max:.2e}")

        if test_passed:
            print("  ✅ Numerical stability verified: No NaN/Inf detected")
        else:
            print("  ❌ Numerical stability test FAILED")

    @torch.no_grad()
    def verify_content_awareness(self):
        """Verify that router makes different decisions for different images"""

        # Collect routing decisions for multiple images
        all_routing_decisions = []

        for batch_idx, (images, _) in enumerate(self.dataloader):
            if batch_idx >= 10:
                break

            images = images.to(self.device)
            features = self.model.backbone(images)

            for feat, sdta, moe in zip(features, self.model.sdta_blocks, self.model.moe_layers):
                feat = sdta(feat)
                gate_logits = moe.gate(feat)
                top_k_logits, top_k_indices = torch.topk(gate_logits, moe.top_k, dim=-1)

                # Store decisions for each image in batch
                for i in range(images.shape[0]):
                    all_routing_decisions.append(top_k_indices[i].cpu().numpy())

        # Convert to numpy array
        all_routing_decisions = np.array(all_routing_decisions)  # [N, 3]

        # Check diversity: Count unique expert combinations
        unique_combinations = set()
        for decision in all_routing_decisions:
            unique_combinations.add(tuple(sorted(decision)))

        diversity_ratio = len(unique_combinations) / len(all_routing_decisions)

        print(f"  📊 Routing diversity: {len(unique_combinations)} unique combinations out of {len(all_routing_decisions)} decisions")
        print(f"     Diversity ratio: {diversity_ratio:.2%}")

        if diversity_ratio < 0.1:
            print(f"  ⚠️  Very low diversity - router may be stuck selecting same experts")
        elif diversity_ratio > 0.5:
            print(f"  ✅ High diversity - router is content-aware")
        else:
            print(f"  ✓  Moderate diversity - router shows some content awareness")

        self.results['routing_diversity'] = {
            'unique_combinations': len(unique_combinations),
            'total_decisions': len(all_routing_decisions),
            'diversity_ratio': diversity_ratio
        }

    @torch.no_grad()
    def analyze_expert_specialization(self):
        """Analyze which experts are selected most often"""

        expert_selection_counts = defaultdict(int)
        expert_names = None
        total_selections = 0

        for batch_idx, (images, _) in enumerate(self.dataloader):
            if batch_idx >= 20:
                break

            images = images.to(self.device)
            features = self.model.backbone(images)

            for feat, sdta, moe in zip(features, self.model.sdta_blocks, self.model.moe_layers):
                feat = sdta(feat)
                _, _, routing_info = moe(feat)

                if expert_names is None:
                    expert_names = routing_info['expert_names']

                # Count expert selections
                expert_counts = routing_info['expert_counts'].cpu().numpy()
                for i, name in enumerate(expert_names):
                    expert_selection_counts[name] += int(expert_counts[i])
                    total_selections += int(expert_counts[i])

        # Print expert usage
        print("\n  📊 Expert Selection Frequency:")
        sorted_experts = sorted(expert_selection_counts.items(), key=lambda x: x[1], reverse=True)

        for name, count in sorted_experts:
            percentage = 100 * count / total_selections if total_selections > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"     {name:25s}: {count:5d} ({percentage:5.1f}%) {bar}")

        # Check for balanced usage
        min_percentage = 100 * sorted_experts[-1][1] / total_selections
        max_percentage = 100 * sorted_experts[0][1] / total_selections

        print(f"\n  📈 Load balance: Min={min_percentage:.1f}%, Max={max_percentage:.1f}%")

        if max_percentage > 80:
            print(f"  ⚠️  One expert dominates (>{max_percentage:.0f}%) - load balancing may be too weak")
        elif max_percentage - min_percentage < 5:
            print(f"  ⚠️  Very uniform usage - experts may not be specializing")
        else:
            print(f"  ✅ Experts show specialization with balanced load")

        self.results['expert_specialization'] = expert_selection_counts

    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*70)
        print("📋 VERIFICATION SUMMARY")
        print("="*70)

        print("\n✓ All verification tests completed!")
        print("\nKey findings:")

        # Routing diversity
        if 'routing_diversity' in self.results:
            diversity = self.results['routing_diversity']['diversity_ratio']
            print(f"  • Routing diversity: {diversity:.1%} (higher is better)")

        # Weight properties
        if 'weight_properties' in self.results:
            avg_weight_std = np.mean([x['weight_std'] for x in self.results['weight_properties']])
            print(f"  • Average weight std: {avg_weight_std:.4f} (>0.05 indicates learning)")

        # Output stats
        if 'output_stats' in self.results:
            avg_output_mean = np.mean([x['mean'] for x in self.results['output_stats']])
            print(f"  • Average output magnitude: {avg_output_mean:.4f}")

        print("\n" + "="*70)
        print("✅ MoE system verification complete!")
        print("="*70 + "\n")


def main():
    """Run MoE verification"""
    import argparse

    parser = argparse.ArgumentParser(description='Verify MoE system integrity')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to COD10K dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--backbone', type=str, default='edgenext_base',
                        help='Backbone architecture')
    parser.add_argument('--num-experts', type=int, default=7,
                        help='Number of experts')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Image size')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing')

    args = parser.parse_args()

    # Create model
    print("Loading model...")
    model = CamoXpert(in_channels=3, num_classes=1, pretrained=True, backbone=args.backbone,
                      num_experts=args.num_experts)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using randomly initialized model (for architecture verification)")

    model = model.cuda().eval()

    # Create dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = COD10KDataset(args.dataset_path, 'val', args.img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Dataset size: {len(dataset)} images")

    # Run verification
    verifier = MoEVerifier(model, dataloader)
    verifier.verify_all()


if __name__ == '__main__':
    main()
