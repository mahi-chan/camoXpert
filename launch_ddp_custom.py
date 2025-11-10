"""
Custom DDP launcher with memory-optimized configuration for Tesla T4
Run with: python launch_ddp_custom.py

Memory-safe settings:
- Batch size: 20 per GPU (40 total with 2 GPUs)
- Mixed precision (AMP) enabled for 40% memory savings
- Gradient checkpointing enabled
"""
import os
import torch

# Set DDP environment variables
ngpus = torch.cuda.device_count()
batch_size = 20  # Safe for Tesla T4 with AMP
stage2_batch = 16  # Even safer for stage 2

print(f"🚀 Launching DDP training with {ngpus} GPUs...")
print(f"   Total batch size: Stage 1 = {batch_size * ngpus}, Stage 2 = {stage2_batch * ngpus}")
print(f"   Mixed precision: Enabled (AMP)")
print(f"   Gradient checkpointing: Enabled")
print(f"   Memory optimization: Tesla T4 optimized\n")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Launch with memory-optimized configuration
cmd = f"""
torchrun --nproc_per_node={ngpus} --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size {batch_size} \
    --stage2-batch-size {stage2_batch} \
    --accumulation-steps 1 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.0008 \
    --stage2-lr 0.00055 \
    --scheduler cosine \
    --min-lr 0.0001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --gradient-checkpointing \
    --num-workers 4 \
    --use-cod-specialized
"""

print(f"\nCommand:\n{cmd}\n")
print("=" * 70)
os.system(cmd)
