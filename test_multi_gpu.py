#!/usr/bin/env python3
"""
Test script to verify multi-GPU detection and setup
"""

import torch
import torch.nn as nn

def test_multi_gpu_setup():
    """Test multi-GPU detection and DataParallel setup"""

    print("="*70)
    print("MULTI-GPU DETECTION TEST")
    print("="*70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        print("   Install CUDA-enabled PyTorch to use GPU training")
        return False

    print(f"✓ CUDA is available (version: {torch.version.cuda})")

    # Count GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n📊 Number of GPUs detected: {num_gpus}")

    if num_gpus == 0:
        print("❌ No GPUs found")
        return False

    # List all GPUs
    print("\nGPU Details:")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)
        gpu_memory = gpu_props.total_memory / 1e9
        compute_capability = f"{gpu_props.major}.{gpu_props.minor}"

        print(f"\n  GPU {i}:")
        print(f"    Name: {gpu_name}")
        print(f"    Memory: {gpu_memory:.1f} GB")
        print(f"    Compute Capability: {compute_capability}")
        print(f"    Multi-Processor Count: {gpu_props.multi_processor_count}")

    # Test DataParallel
    print("\n" + "="*70)
    print("TESTING DATAPARALLEL")
    print("="*70)

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 1, 1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.conv3(x)
            return x

    # Create model and move to GPU
    model = SimpleModel().cuda()
    print(f"\n✓ Model created and moved to GPU")

    # Wrap with DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"\n🔧 Wrapping model with DataParallel across {num_gpus} GPUs...")
        model = nn.DataParallel(model)
        print(f"✓ DataParallel enabled on GPUs: {list(model.device_ids)}")
    else:
        print(f"\n✓ Single GPU mode (DataParallel not needed)")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 8
    test_input = torch.randn(batch_size, 3, 224, 224).cuda()

    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        # Check memory usage
        print(f"\nGPU Memory Usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if num_gpus == 1:
        print(f"✓ Single GPU setup working correctly")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Ready for training")
    else:
        print(f"✓ Multi-GPU setup working correctly")
        print(f"  - {num_gpus} GPUs available")
        print(f"  - DataParallel enabled")
        print(f"  - Effective batch size will be {num_gpus}x larger")
        print(f"  - Training speed will be ~{num_gpus}x faster")

    print("="*70)

    return True


if __name__ == '__main__':
    success = test_multi_gpu_setup()

    if not success:
        print("\n⚠️  Multi-GPU setup test failed")
        print("   Please check your CUDA installation and GPU availability")
        exit(1)
    else:
        print("\n🎉 Multi-GPU setup test passed!")
        print("   Your system is ready for multi-GPU training")
        exit(0)
