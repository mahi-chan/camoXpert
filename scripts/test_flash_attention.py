#!/usr/bin/env python3
"""
Test Flash Attention vs Standard Attention
Verifies that outputs are identical (within numerical precision)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/user/camoXpert')

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from torch.utils.data import DataLoader


def test_attention_equivalence():
    """Test that Flash Attention produces identical outputs to standard attention"""

    print("\n" + "="*70)
    print("🔬 FLASH ATTENTION VERIFICATION TEST")
    print("="*70 + "\n")

    # Test at different scales
    test_configs = [
        {'name': 'Small', 'batch': 2, 'heads': 4, 'seq_len': 256, 'dim': 32},
        {'name': 'Medium', 'batch': 4, 'heads': 8, 'seq_len': 576, 'dim': 64},
        {'name': 'Large', 'batch': 2, 'heads': 8, 'seq_len': 1024, 'dim': 64},
    ]

    print("Testing Flash Attention at different scales...\n")

    for config in test_configs:
        B, H, N, D = config['batch'], config['heads'], config['seq_len'], config['dim']

        # Generate random Q, K, V
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

        # Compute with Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            out_flash = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)
        else:
            print("  ⚠️  Flash Attention not available - skipping test")
            continue

        # Compute with standard attention
        scale = D ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out_std = attn @ V

        # Compare outputs
        diff = (out_flash - out_std).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check if within float32 precision
        tolerance = 1e-4  # Float32 precision limit
        passed = max_diff < tolerance

        status = "✅" if passed else "❌"
        print(f"{status} {config['name']:8s} | "
              f"B={B}, H={H}, N={N:4d}, D={D:2d} | "
              f"Max: {max_diff:.2e}, Mean: {mean_diff:.2e}")

        if not passed:
            print(f"   ⚠️  FAILED: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")

    print("\n" + "="*70)


def test_model_with_flash_attention(dataset_path, checkpoint_path=None, num_samples=10):
    """Test full model with Flash Attention vs Standard Attention"""

    print("\n" + "="*70)
    print("🔬 MODEL-LEVEL FLASH ATTENTION TEST")
    print("="*70 + "\n")

    # Create dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = COD10KDataset(dataset_path, 'val', 320, augment=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # Test with Flash Attention enabled
    print("\n1️⃣  Creating model with Flash Attention ENABLED...")
    model_flash = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()
    model_flash.eval()

    # Test with Flash Attention disabled
    print("\n2️⃣  Creating model with Flash Attention DISABLED...")
    model_std = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()

    # Disable Flash Attention in all SDTAEncoder modules
    for module in model_std.modules():
        if hasattr(module, 'use_flash_attn'):
            module.use_flash_attn = False
            module.flash_attn_available = False

    model_std.eval()

    # Copy weights from flash model to std model
    model_std.load_state_dict(model_flash.state_dict())

    # Test on multiple samples
    print(f"\n3️⃣  Testing on {num_samples} samples...\n")

    all_diffs = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Testing", total=num_samples)):
            if batch_idx >= num_samples:
                break

            images = images.cuda()

            # Forward pass with Flash Attention
            pred_flash, _, _ = model_flash(images)

            # Forward pass with standard attention
            pred_std, _, _ = model_std(images)

            # Compare outputs
            diff = (pred_flash - pred_std).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            all_diffs.append(max_diff)

    # Summary
    max_diff_overall = max(all_diffs)
    mean_diff_overall = np.mean(all_diffs)

    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Tested {num_samples} batches")
    print(f"Max difference across all samples:  {max_diff_overall:.2e}")
    print(f"Mean difference across all samples: {mean_diff_overall:.2e}")

    tolerance = 1e-4
    if max_diff_overall < tolerance:
        print(f"\n✅ PASSED: Outputs are identical (within float32 precision {tolerance:.2e})")
        print("   Flash Attention can be safely used without accuracy loss.")
    else:
        print(f"\n❌ FAILED: Max difference {max_diff_overall:.2e} exceeds tolerance {tolerance:.2e}")
        print("   Flash Attention may have compatibility issues.")

    print("="*70 + "\n")


def benchmark_speed(iterations=50):
    """Benchmark Flash Attention vs Standard Attention speed"""

    print("\n" + "="*70)
    print("⚡ FLASH ATTENTION SPEED BENCHMARK")
    print("="*70 + "\n")

    if not hasattr(F, 'scaled_dot_product_attention'):
        print("⚠️  Flash Attention not available - skipping benchmark")
        return

    # Test configuration (typical for CamoXpert)
    B, H, N, D = 16, 8, 576, 64  # 24x24 feature map

    Q = torch.randn(B, H, N, D, device='cuda')
    K = torch.randn(B, H, N, D, device='cuda')
    V = torch.randn(B, H, N, D, device='cuda')

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Benchmark Flash Attention
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        out = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    time_flash = start.elapsed_time(end) / iterations

    # Warmup standard attention
    scale = D ** -0.5
    for _ in range(10):
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        _ = attn @ V
    torch.cuda.synchronize()

    # Benchmark standard attention
    start.record()
    for _ in range(iterations):
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ V
    end.record()
    torch.cuda.synchronize()

    time_std = start.elapsed_time(end) / iterations

    # Results
    speedup = time_std / time_flash

    print(f"Configuration: B={B}, H={H}, N={N}, D={D}")
    print(f"\nStandard Attention:  {time_std:.2f} ms")
    print(f"Flash Attention:     {time_flash:.2f} ms")
    print(f"Speedup:             {speedup:.2f}x")

    if speedup > 1.5:
        print(f"\n✅ Flash Attention is {speedup:.1f}x faster!")
    elif speedup > 1.0:
        print(f"\n✓  Flash Attention is slightly faster ({speedup:.2f}x)")
    else:
        print(f"\n⚠️  Flash Attention is slower - may not be optimal for this configuration")

    print("="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test Flash Attention implementation')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to COD10K dataset (for model-level test)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--skip-model-test', action='store_true',
                        help='Skip model-level test (only run unit tests)')

    args = parser.parse_args()

    # Test 1: Unit test - attention equivalence
    test_attention_equivalence()

    # Test 2: Speed benchmark
    benchmark_speed()

    # Test 3: Model-level test (optional, requires dataset)
    if args.dataset_path and not args.skip_model_test:
        test_model_with_flash_attention(args.dataset_path, args.checkpoint, args.num_samples)
    else:
        print("\nℹ️  Skipping model-level test (no dataset provided)")
        print("   Run with --dataset-path to test full model\n")

    print("✅ All tests completed!\n")


if __name__ == '__main__':
    main()
