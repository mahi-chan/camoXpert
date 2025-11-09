#!/bin/bash
# Custom DDP Training Script with Your Exact Configuration
# Run with: bash train_ddp_custom.sh

echo "=================================================="
echo "CAMOXPERT DDP TRAINING - CUSTOM CONFIGURATION"
echo "=================================================="
echo "GPUs: 2 × Tesla T4"
echo "Total Batch: Stage 1 = 64, Stage 2 = 48"
echo "Resolution: 352px"
echo "Epochs: 150 (30 decoder + 120 full)"
echo "=================================================="
echo ""

torchrun --nproc_per_node=2 --master_port=29500 train_ultimate.py train \
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

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
