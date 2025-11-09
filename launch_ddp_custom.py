"""
Custom DDP launcher with your exact configuration
Run with: python launch_ddp_custom.py
"""
import os
import torch

# Set DDP environment variables
ngpus = torch.cuda.device_count()
print(f"Launching DDP training with {ngpus} GPUs...")
print(f"Total batch size: Stage 1 = {32 * ngpus}, Stage 2 = {24 * ngpus}")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Launch with your exact configuration
cmd = f"""
torchrun --nproc_per_node={ngpus} --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 32 \
    --stage2-batch-size 24 \
    --accumulation-steps 1 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.0008 \
    --stage2-lr 0.00055 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --num-workers 4 \
    --no-amp \
    --use-cod-specialized
"""

print(f"\nCommand:\n{cmd}\n")
print("=" * 70)
os.system(cmd)
