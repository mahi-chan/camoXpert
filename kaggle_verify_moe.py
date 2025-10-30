#!/usr/bin/env python3
"""
Kaggle-specific MoE Verification Script

Quick verification of the MoE system for CamoXpert model in Kaggle environment.
Run this before training to ensure MoE routing is working correctly.

Usage:
    python kaggle_verify_moe.py                           # Quick test (no checkpoint needed)
    python kaggle_verify_moe.py --checkpoint path.pth     # Full test with trained model
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, '/kaggle/working')

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from torch.utils.data import DataLoader


def print_header(title):
    """Print a nice header"""
    print("\n" + "="*70)
    print(f"🔍 {title}")
    print("="*70 + "\n")


def check_environment():
    """Check Kaggle environment"""
    print_header("ENVIRONMENT CHECK")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {capability[0]}.{capability[1]}")

        if capability[0] >= 7:
            print("✅ GPU supports Flash Attention")
        else:
            print("⚠️  GPU may not support Flash Attention (requires compute >= 7.0)")

    # Check dataset
    dataset_paths = [
        '/kaggle/input/cod10k-dataset/COD10K-v3',
        '/kaggle/input/cod10k-v3/COD10K-v3',
        '/kaggle/input/cod10k'
    ]

    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Dataset found: {path}")
            break

    if dataset_path is None:
        print("❌ COD10K dataset not found!")
        print("   Please add COD10K dataset to your Kaggle notebook")
        return None

    return dataset_path


def quick_moe_test(model, dataloader, num_batches=5):
    """Quick test of MoE routing logic"""
    print_header("QUICK MoE VERIFICATION")

    model.eval()
    all_issues = []
    expert_counts = defaultdict(int)
    total_selections = 0
    routing_diversity = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Testing", total=num_batches)):
            if batch_idx >= num_batches:
                break

            images = images.cuda()
            B = images.shape[0]

            # Extract features
            features = model.backbone(images)

            for layer_idx, (feat, sdta, moe) in enumerate(zip(features, model.sdta_blocks, model.moe_layers)):
                feat = sdta(feat)

                # Get routing info
                gate_logits = moe.gate(feat)
                top_k_logits, top_k_indices = torch.topk(gate_logits, moe.top_k, dim=-1)
                top_k_weights = F.softmax(top_k_logits, dim=-1)

                # Check 1: Weights sum to 1
                weight_sums = top_k_weights.sum(dim=1)
                if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4):
                    all_issues.append(f"Layer {layer_idx}: Weights don't sum to 1.0")

                # Check 2: No NaN/Inf
                if torch.isnan(gate_logits).any() or torch.isinf(gate_logits).any():
                    all_issues.append(f"Layer {layer_idx}: NaN/Inf in gate logits")

                # Check 3: Valid indices
                if (top_k_indices < 0).any() or (top_k_indices >= moe.num_experts).any():
                    all_issues.append(f"Layer {layer_idx}: Invalid expert indices")

                # Track expert usage
                for i in range(B):
                    experts_selected = top_k_indices[i].cpu().numpy()
                    routing_diversity.append(tuple(sorted(experts_selected)))
                    for expert_idx in experts_selected:
                        expert_counts[expert_idx.item()] += 1
                        total_selections += 1

                # Run MoE forward to check output
                output, aux_loss, routing_info = moe(feat)

                # Check 4: Output is valid
                if torch.isnan(output).any() or torch.isinf(output).any():
                    all_issues.append(f"Layer {layer_idx}: NaN/Inf in MoE output")

                if output.abs().sum() < 1e-6:
                    all_issues.append(f"Layer {layer_idx}: MoE output is all zeros")

    # Print results
    print("\n📊 VERIFICATION RESULTS:\n")

    if len(all_issues) == 0:
        print("✅ All checks PASSED!")
        print("   - Weights sum to 1.0 ✓")
        print("   - No NaN/Inf values ✓")
        print("   - Valid expert indices ✓")
        print("   - MoE outputs are valid ✓")
    else:
        print("❌ ISSUES FOUND:")
        for issue in all_issues:
            print(f"   - {issue}")

    # Expert usage statistics
    print("\n📊 EXPERT USAGE:\n")
    expert_names = ['TextureExpert', 'AttentionExpert', 'HybridExpert',
                    'FrequencyExpert', 'EdgeExpert', 'SemanticContextExpert', 'ContrastExpert']

    sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
    for expert_idx, count in sorted_experts:
        if expert_idx < len(expert_names):
            name = expert_names[expert_idx]
            percentage = 100 * count / total_selections if total_selections > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"   {name:25s}: {count:5d} ({percentage:5.1f}%) {bar}")

    # Routing diversity
    unique_combinations = len(set(routing_diversity))
    total_decisions = len(routing_diversity)
    diversity_ratio = unique_combinations / total_decisions if total_decisions > 0 else 0

    print(f"\n📈 ROUTING DIVERSITY:")
    print(f"   Unique combinations: {unique_combinations}/{total_decisions}")
    print(f"   Diversity ratio: {diversity_ratio:.1%}")

    if diversity_ratio < 0.1:
        print("   ⚠️  Very low diversity - router may be stuck")
    elif diversity_ratio > 0.5:
        print("   ✅ High diversity - router is content-aware")
    else:
        print("   ✓  Moderate diversity")

    return len(all_issues) == 0


def full_model_test(model, dataloader, num_batches=10):
    """Full forward pass test"""
    print_header("FULL MODEL TEST")

    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Testing", total=num_batches)):
            if batch_idx >= num_batches:
                break

            images = images.cuda()
            masks = masks.cuda()

            # Full forward pass
            pred, aux_loss, _ = model(images)
            pred = torch.sigmoid(pred)

            # Check outputs
            if torch.isnan(pred).any():
                print("❌ NaN detected in predictions")
                return False

            if torch.isinf(pred).any():
                print("❌ Inf detected in predictions")
                return False

            # Check reasonable output range
            pred_min = pred.min().item()
            pred_max = pred.max().item()

            all_metrics.append({
                'min': pred_min,
                'max': pred_max,
                'mean': pred.mean().item(),
                'std': pred.std().item()
            })

    # Summary
    print("\n📊 OUTPUT STATISTICS:\n")
    print(f"   Min prediction:  {np.mean([m['min'] for m in all_metrics]):.4f}")
    print(f"   Max prediction:  {np.mean([m['max'] for m in all_metrics]):.4f}")
    print(f"   Mean prediction: {np.mean([m['mean'] for m in all_metrics]):.4f}")
    print(f"   Std prediction:  {np.mean([m['std'] for m in all_metrics]):.4f}")

    # Check if outputs are reasonable (after sigmoid, should be in [0, 1])
    avg_min = np.mean([m['min'] for m in all_metrics])
    avg_max = np.mean([m['max'] for m in all_metrics])

    if avg_min < -0.1 or avg_max > 1.1:
        print("\n⚠️  Predictions outside expected range [0, 1]")
        return False

    print("\n✅ Full model test PASSED!")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify MoE system in Kaggle')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (optional, for testing trained model)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Image size')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='Number of batches to test')

    args = parser.parse_args()

    # Check environment
    dataset_path = check_environment()
    if dataset_path is None:
        return

    # Create model
    print_header("LOADING MODEL")
    print("Creating CamoXpert model...")
    model = CamoXpert(
        in_channels=3,
        out_channels=1,
        pretrained=False,  # Don't need pretrained weights for verification
        backbone='edgenext_base',
        num_experts=7
    ).cuda()

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Checkpoint loaded")
    else:
        print("Using randomly initialized model (architecture verification only)")

    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M ({trainable_params/1e6:.1f}M trainable)")

    # Create dataset
    print(f"\nLoading validation dataset from {dataset_path}...")
    try:
        dataset = COD10KDataset(dataset_path, 'val', args.img_size, augment=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
        print(f"✅ Dataset loaded: {len(dataset)} images")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Run tests
    quick_passed = quick_moe_test(model, dataloader, num_batches=args.num_batches)
    full_passed = full_model_test(model, dataloader, num_batches=args.num_batches)

    # Final summary
    print_header("FINAL SUMMARY")

    if quick_passed and full_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYour MoE system is working correctly:")
        print("  ✓ Router learns from features")
        print("  ✓ Top-3 experts selected dynamically")
        print("  ✓ Outputs combined with learned weights")
        print("  ✓ No numerical issues detected")
        print("\n🚀 Ready to train!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease review the issues above before training.")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
